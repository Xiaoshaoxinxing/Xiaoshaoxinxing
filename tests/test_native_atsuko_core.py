"""B3 input isolation, core-loss math, calibration guards and checkpoint integrity."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from test_native_omni_atsuko import Thinker
from mer_omni.native_atsuko_core import CoreNativeAtsuko, insert_affect_slots, core_auxiliary_loss
from mer_omni.calibration import set_calibration
from mer_omni.core_protocol import train_vocabulary, auxiliary_target, calibration_signature, apply_calibration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from mer_omni.sft import atsuko as runner


class CoreTests(unittest.TestCase):
    def fixture(self):
        cfg = json.loads((ROOT/'tests/fixtures/config/native_omni_atsuko_core_epoch1.json').read_text())
        spectral = cfg['bridge']['spectral']
        spectral.update(dim=8, heads=2, layers=1, sequence_lengths=[8,8,8], conv_kernel_sizes=[1,1,1])
        cfg['bridge']['auxiliary']['classes'] = 4
        model = CoreNativeAtsuko(Thinker(), spectral, 10, cfg['bridge']['auxiliary'])
        set_calibration(model.frontend, [torch.ones(8)]*3, 1)
        model.eval()
        batch = dict(input_ids=torch.tensor([[1,3,3,4,4,10,10,10,10,2]]),
            attention_mask=torch.ones(1,10,dtype=torch.long), input_features=torch.randn(1,2,8),
            pixel_values_videos=torch.randn(2,8), feature_attention_mask=torch.ones(1,8),
            video_grid_thw=torch.tensor([[1,2,4]]), bypass_subtitle_ids=torch.tensor([[7,8]]))
        return model,batch,cfg

    def test_four_masked_slots_recover_exact_original_and_answers(self):
        batch = dict(input_ids=torch.tensor([[1,9,5,6,9,7,8,9]]),
            attention_mask=torch.ones(1,8,dtype=torch.long), labels=torch.tensor([[-100]*6+[8,9]]))
        changed,pos=insert_affect_slots(batch,10,9)
        self.assertEqual(pos,4)
        self.assertTrue((changed['labels'][:,pos:pos+4] == -100).all())
        for k in batch:
            torch.testing.assert_close(torch.cat((changed[k][:,:pos],changed[k][:,pos+4:]),1),batch[k],rtol=0,atol=0)

    def test_collision_and_invalid_slot_count_rejected(self):
        with self.assertRaises(ValueError):
            insert_affect_slots(dict(input_ids=torch.tensor([[10,9,9]])),10,9)
        model,batch,_=self.fixture();batch['input_ids'][0,6]=7
        with self.assertRaisesRegex(ValueError,'four consecutive'):
            model.prepare_embeddings(batch)

    def test_media_preserved_encoded_once_inputs_not_mutated(self):
        model,batch,_=self.fixture(); original={k:v.clone() for k,v in batch.items()}
        result=model.prepare_embeddings(batch)
        torch.testing.assert_close(result[0,1:3],batch['input_features'][0],rtol=0,atol=0)
        torch.testing.assert_close(result[0,3:5],batch['pixel_values_videos'],rtol=0,atol=0)
        for i in (0,9):
            torch.testing.assert_close(result[:,i],model.thinker.embedding(batch['input_ids'][:,i]),rtol=0,atol=0)
        self.assertEqual(model.thinker.calls,[1,1])
        for key in batch:
            torch.testing.assert_close(batch[key],original[key],rtol=0,atol=0)

    def test_labels_never_enter_embeddings_or_native_forward_kwargs(self):
        model,batch,_=self.fixture()
        first=model.prepare_embeddings(dict(batch,aux_target=torch.zeros(1,4)))
        second=model.prepare_embeddings(dict(batch,aux_target=torch.ones(1,4)))
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertNotIn('aux_target',model.native_kwargs(dict(batch,aux_target=torch.ones(1,4))))

    def test_normalize_real_positions_then_zero_pad(self):
        model,_,_=self.fixture()
        seq=[torch.randn(2,8)*5+3]
        for norm in model.frontend.input_norms:
            norm.bias.data.fill_(2.)
        prepared=model.frontend.prepare_sequences((seq,seq,seq))
        for values in prepared:
            self.assertTrue(torch.equal(values[:,2:],torch.zeros_like(values[:,2:])))
            self.assertAlmostEqual(float(values[:,:2].mean()),2.,places=5)

    def test_post_attention_router_uses_attended_modalities(self):
        model,batch,_=self.fixture(); captured=[]
        hook=model.frontend.modality_router.register_forward_pre_hook(lambda m,args: captured.append(args[0].detach().clone()))
        sequences,_,_=model.encode_sequences(batch)
        result=model.frontend(sequences);hook.remove()
        torch.testing.assert_close(captured[0],result['modalities'].flatten(1),rtol=0,atol=0)

    def test_all_heads_normalizers_and_projectors_receive_gradients(self):
        model,batch,cfg=self.fixture(); model.train()
        embeddings,aux=model._prepare(batch)
        losses=core_auxiliary_loss(aux,torch.tensor([[1.,0.,1.,0.]]),cfg['bridge']['auxiliary'])
        upstream=torch.randn_like(embeddings[:,5:9])
        ((embeddings[:,5:9]*upstream).mean()+.5*losses['total']).backward()
        for name,p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
            else:
                self.assertIsNone(p.grad,name)
        for head in [*model.fusion_in.unimodal,*model.fusion_in.bimodal,model.fusion_in.fused]:
            self.assertGreater(float(head.classifier.weight.grad.abs().sum()),0)

    def test_token_path_alone_reaches_frontend_and_auxiliary_path_stays_off_projector(self):
        model,batch,cfg=self.fixture()
        embeddings=model.prepare_embeddings(batch)
        (embeddings[:,5:9]*torch.randn_like(embeddings[:,5:9])).sum().backward()
        for module in (model.frontend.input_norms,model.frontend.adapters,model.frontend.modality_router,
                       model.fusion_in.projector,model.fusion_in.output_norm):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0)
        # Classification-only layers have no causal role in the token representation.
        for head in [*model.fusion_in.unimodal,*model.fusion_in.bimodal,model.fusion_in.fused]:
            self.assertIsNone(head.classifier.weight.grad)
        model.zero_grad(set_to_none=True)
        _,aux=model._prepare(batch)
        core_auxiliary_loss(aux,torch.ones(1,4),cfg['bridge']['auxiliary'])['total'].backward()
        for module in (model.fusion_in.projector,model.fusion_in.output_norm,model.thinker):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertIsNone(model.fusion_in.slot_embedding.grad)
        self.assertGreater(float(model.frontend.input_norms[0].weight.grad.abs().sum()),0)

    def test_auxiliary_formula_and_mcm_order_stopgradient(self):
        model,_,cfg=self.fixture();settings=cfg['bridge']['auxiliary']
        logits={k:torch.zeros(2,4,requires_grad=True) for k in ('fusion','text','video','audio','va','la','lv')}
        logits['lv']=torch.full((2,4),-3.,requires_grad=True)  # no audio hurts positive-label prediction
        routing_logits=torch.zeros(2,3,requires_grad=True)
        aux=dict(logits=logits,weights=routing_logits.softmax(-1))
        target=torch.ones(2,4)
        result=core_auxiliary_loss(aux,target,settings)
        self.assertTrue((result['complementarity_target'].argmax(-1)==2).all())
        self.assertFalse(result['complementarity_target'].requires_grad)
        fused=F.binary_cross_entropy_with_logits(logits['fusion'],target)
        uni=torch.stack([F.binary_cross_entropy_with_logits(logits[k],target) for k in ('text','video','audio')]).mean()
        bi=torch.stack([F.binary_cross_entropy_with_logits(logits[k],target) for k in ('va','la','lv')]).mean()
        torch.testing.assert_close(result['total'],fused+uni+.3*bi+.01*result['complementarity_kl'])
        result['complementarity_kl'].backward()
        self.assertGreater(float(routing_logits.grad.abs().sum()),0)
        self.assertTrue(all(v.grad is None for v in logits.values()))

    def test_four_tokens_not_rescaled_by_router_after_projection(self):
        model,_,_=self.fixture()
        features=dict(modalities=torch.randn(1,3,8),fusion=torch.randn(1,8),
            modality_weights=torch.tensor([[.999,.0005,.0005]]),band_weights=torch.ones(1,3,3)/3)
        first,_=model.fusion_in(features)
        features['modality_weights']=torch.ones(1,3)/3
        second,_=model.fusion_in(features)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        rms=first.square().mean(-1).sqrt()
        torch.testing.assert_close(rms,torch.full_like(rms,model.initial_embedding_rms),rtol=.002,atol=.001)

    def test_training_only_vocabulary_deterministic_ties_and_targets(self):
        rows=[dict(id='a',dataset='MER-Caption+',split='train',openset=[' Happy ','happy','sad']),
              dict(id='b',dataset='MER-Caption+',split='train',openset=['angry'])]
        vocab=train_vocabulary(rows,'sha',2)
        self.assertEqual(vocab,train_vocabulary(list(reversed(rows)),'sha',2))
        self.assertEqual(vocab['vocab'],['angry','happy'])
        torch.testing.assert_close(auxiliary_target(['HAPPY','unknown'],vocab['vocab']),torch.tensor([[0.,1.]]))
        bad=copy.deepcopy(rows);bad[0]['split']='test'
        with self.assertRaisesRegex(ValueError,'training-only'):train_vocabulary(bad,'sha',2)

    def test_calibration_rejects_old_incomplete_and_changed_normalization(self):
        model,_,cfg=self.fixture()
        signature=calibration_signature(model,cfg,'sha',3)
        artifact=dict(format='native-atsuko-core-bands-v1',signature=signature,complete=True,processed=3,
            energy=[[1.]*8]*3,band_edges=[s.band_edges.tolist() for s in model.frontend.spectra])
        apply_calibration(model,artifact,signature)
        for bad in [dict(artifact,format='atsuko-training-bands-v1'),dict(artifact,processed=2),dict(artifact,complete=False)]:
            with self.assertRaises(ValueError):apply_calibration(model,bad,signature)
        model.frontend.input_norms[0].weight.data.add_(.1)
        changed=calibration_signature(model,cfg,'sha',3)
        self.assertNotEqual(signature,changed)
        with self.assertRaises(ValueError):apply_calibration(model,artifact,changed)

    def test_checkpoint_restores_parameters_nonpersistent_bases_optimizer_and_rng(self):
        model,batch,cfg=self.fixture();optimizer=runner.build_optimizer(model,cfg['train'])
        values,aux=model._prepare(batch)
        (values[:,5:9].sum()+core_auxiliary_loss(aux,torch.ones(1,4),cfg['bridge']['auxiliary'])['total']).backward()
        optimizer.step()  # tiny synthetic fixture only, not a pretrained-model training update
        initial=runner.state(model)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt';runner.save_checkpoint(path,model,optimizer,7,{'test':'identity'})
            expected_random=torch.rand(3)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:p.add_(1.)
                model.frontend.spectra[0].U.add_(1.)
            self.assertEqual(runner.restore_checkpoint(path,model,optimizer,{'test':'identity'}),7)
            torch.testing.assert_close(torch.rand(3),expected_random,rtol=0,atol=0)
            restored=runner.state(model)
            for name,value in initial.items():torch.testing.assert_close(value,restored[name],rtol=0,atol=0)
            with self.assertRaises(ValueError):runner.restore_checkpoint(path,model,optimizer,{'test':'wrong'})
            receipt_path=path.with_suffix('.receipt.json')
            receipt=json.loads(receipt_path.read_text());receipt['step']=8
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):runner.restore_checkpoint(path,model,optimizer,{'test':'identity'})

    def test_resume_next_update_matches_uninterrupted_with_fresh_optimizer_and_dropout(self):
        model,batch,cfg=self.fixture();model.train()
        optimizer=runner.build_optimizer(model,cfg['train'])
        def update_toy(net,opt):
            opt.zero_grad(set_to_none=True)
            embeddings,aux=net._prepare(batch)
            loss=(embeddings[:,5:9]*torch.randn_like(embeddings[:,5:9])).mean()
            loss=loss+.5*core_auxiliary_loss(aux,torch.ones(1,4),cfg['bridge']['auxiliary'])['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],.6)
            opt.step()
            return loss.detach()
        update_toy(model,optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt';runner.save_checkpoint(path,model,optimizer,1,{'test':'replay'})
            expected_loss=update_toy(model,optimizer)
            expected_state=runner.state(model)
            expected_optimizer=copy.deepcopy(optimizer.state_dict())
            fresh,_,_=self.fixture();fresh.train()
            # Both runs must load the same frozen base, as the real loader does.
            # Toy Thinker embeddings are random unless explicitly synchronized.
            fresh.thinker.load_state_dict(model.thinker.state_dict())
            fresh_optimizer=runner.build_optimizer(fresh,cfg['train'])
            runner.restore_checkpoint(path,fresh,fresh_optimizer,{'test':'replay'})
            actual_loss=update_toy(fresh,fresh_optimizer)
            torch.testing.assert_close(actual_loss,expected_loss,rtol=0,atol=0)
            for name,value in runner.state(fresh).items():
                torch.testing.assert_close(value,expected_state[name],rtol=0,atol=0)
            actual_optimizer=fresh_optimizer.state_dict()
            self.assertEqual(actual_optimizer['param_groups'],expected_optimizer['param_groups'])
            for key,values in expected_optimizer['state'].items():
                for name,value in values.items():
                    torch.testing.assert_close(actual_optimizer['state'][key][name],value,rtol=0,atol=0)

    def test_optimizer_membership_unique_and_norm_slot_no_decay(self):
        model,_,cfg=self.fixture();optimizer=runner.build_optimizer(model,cfg['train'])
        ids=[id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})
        for name,param in model.named_parameters():
            if not param.requires_grad:continue
            group=next(g for g in optimizer.param_groups if any(p is param for p in g['params']))
            self.assertEqual(group['base_lr'],1e-4)
            if 'norm' in name or name.endswith('slot_embedding'):
                self.assertEqual(group['weight_decay'],0.)

    def test_init_does_not_consume_global_rng(self):
        model,_,cfg=self.fixture();rng=torch.get_rng_state().clone()
        CoreNativeAtsuko(model.thinker,cfg['bridge']['spectral'],10,cfg['bridge']['auxiliary'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_accumulation_weights_CE_by_tokens_and_aux_by_samples(self):
        class ObjectiveModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frontend = torch.nn.Linear(1,1,bias=False)
                self.frontend.weight.data.fill_(.7)
                self.auxiliary_settings = {'weight':.5}
            def forward(self, x, labels, **unused):
                p=self.frontend.weight.squeeze()
                ce=(p-x.squeeze()).square()
                aux=(2*p+x.squeeze()).square()
                return SimpleNamespace(loss=ce, auxiliary_losses={k:aux for k in
                    ('total','fusion_bce','unimodal_bce','bimodal_bce','complementarity_kl')})
        model=ObjectiveModel()
        cfg=json.loads((ROOT/'tests/fixtures/config/native_omni_atsuko_core_epoch1.json').read_text())
        settings=dict(cfg['train'],steps_per_epoch=10443,grad_clip=1e6)
        optimizer=torch.optim.SGD([{'params':list(model.parameters()),'base_lr':1e-4}],lr=1e-4)
        items=[{'train':{'x':torch.tensor([1.]),'labels':torch.tensor([[-100,1]])}},
               {'train':{'x':torch.tensor([3.]),'labels':torch.tensor([[-100,1,2,3]])}}]
        p=torch.tensor(.7,requires_grad=True)
        ce=((p-1).square()+3*(p-3).square())/4
        aux=((2*p+1).square()+(2*p+3).square())/2
        (ce+.5*aux).backward()
        stats=runner.update(model,optimizer,items,settings,0)
        torch.testing.assert_close(model.frontend.weight.grad.squeeze(),p.grad)
        self.assertAlmostEqual(stats['loss'],float(ce),places=5)
        self.assertAlmostEqual(stats['total_loss'],float(ce+.5*aux),places=5)

    def test_calibration_interrupt_resume_matches_full_pass_and_never_uses_labels(self):
        model,batch,cfg=self.fixture()
        rows=[dict(id=str(i),text='subtitle',video='unused',audio='unused',
                   description='MUST NOT REACH CALIBRATION',openset=['secret']) for i in range(3)]
        metadata={r['id']:{} for r in rows}
        proc=SimpleNamespace(tokenizer=lambda *a,**kw:{'input_ids':batch['bypass_subtitle_ids']})
        calls=[]
        def prompt(row,*unused):
            self.assertEqual(set(row),{'id','text','video','audio'})
            calls.append(row['id'])
            if len(calls)==2:raise RuntimeError('simulated decoder interruption')
            return {k:v.clone() for k,v in batch.items()},{}
        def load(config,processor,device,**unused):
            return model,{'calibration_signature':calibration_signature(model,config,'test_sha',3)}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'report';out.mkdir()
            cfg['bridge']['calibration_file']=str(Path(directory)/'bands.json')
            with patch.object(runner,'verify',return_value=[]), patch.object(runner.a,'load_inputs',return_value=(rows,proc,metadata)), \
                    patch.object(runner,'load_model',side_effect=load), patch.object(runner.native_evaluation,'prompt_only',side_effect=prompt):
                with self.assertRaisesRegex(RuntimeError,'interruption'):runner.calibrate(cfg,Path(directory),out,'cpu')
                self.assertFalse(Path(cfg['bridge']['calibration_file']).exists())
                self.assertEqual(json.loads((out/'calibration-progress.json').read_text())['processed'],1)
                runner.calibrate(cfg,Path(directory),out,'cpu')
                resumed=json.loads(Path(cfg['bridge']['calibration_file']).read_text())
                self.assertEqual(calls,['0','1','1','2'])
                self.assertEqual(resumed['processed'],3)
                self.assertTrue(resumed['complete'])
                other=Path(directory)/'fresh';other.mkdir()
                cfg['bridge']['calibration_file']=str(Path(directory)/'fresh-bands.json')
                runner.calibrate(cfg,Path(directory),other,'cpu')
                fresh=json.loads(Path(cfg['bridge']['calibration_file']).read_text())
                self.assertEqual(resumed['energy'],fresh['energy'])
                self.assertEqual(resumed['band_edges'],fresh['band_edges'])


if __name__=='__main__':unittest.main()
