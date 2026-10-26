"""CPU checks of shared SFT targets, token accumulation and resumable state."""
import copy
import importlib.util
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import torch

from mer_omni.sft import native_inputs as exp


class Toy(torch.nn.Module):
    def __init__(self, random_noise=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.3, dtype=torch.float64))
        self.random_noise = random_noise

    def forward(self, x, labels, **kwargs):
        noise = torch.rand((), dtype=torch.float64) if self.random_noise else 1.
        return SimpleNamespace(loss=((self.weight*noise-x)**2).mean())


def item(x, n):
    return dict(train=dict(x=torch.full((n,), x, dtype=torch.float64), labels=torch.tensor([[-100]+[7]*n])))


class EpochOneTests(unittest.TestCase):
    def test_native_template_accepts_single_string_or_single_item_list(self):
        for output in ('rendered', ['rendered']):
            processor=SimpleNamespace(apply_chat_template=lambda *a, **kw: output)
            self.assertEqual(exp.render(processor,[],True),'rendered')
        processor=SimpleNamespace(apply_chat_template=lambda *a, **kw: ['a','b'])
        with self.assertRaises(TypeError): exp.render(processor,[],True)

    def settings(self):
        return dict(lr=.02, min_lr=.02, warmup_lr=.001, warmup_steps=2,
                    epochs=1, steps_per_epoch=3, grad_clip=100.)

    def optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.05)

    def test_plan_matches_old_torch_and_python_streams_without_global_rng_changes(self):
        rows = [dict(id=str(i)) for i in range(17)]
        py, pt = random.getstate(), torch.get_rng_state().clone()
        plan, state = exp.plan_for(rows, 42)
        order = torch.randperm(17, generator=torch.Generator().manual_seed(42)).tolist()
        rng = random.Random(42)
        self.assertEqual([p['index'] for p in plan], order)
        self.assertEqual([p['qtype'] for p in plan], [rng.choice(('description','ovlabel')) for _ in order])
        self.assertEqual(state,rng.getstate())
        self.assertEqual(py,random.getstate())
        self.assertTrue(torch.equal(pt,torch.get_rng_state()))

    def test_complete_epoch_keeps_single_sample_tail(self):
        plan,_ = exp.plan_for([dict(id=str(i)) for i in range(31327)],42)
        groups = exp.groups_for(plan)
        self.assertEqual(len(groups),10443)
        self.assertEqual(len(groups[-1]),1)
        self.assertEqual(len({p['index'] for g in groups for p in g}),31327)

    def test_token_weighted_accumulation_matches_physical_batch(self):
        a,b = Toy(),Toy()
        oa,ob = self.optimizer(a),self.optimizer(b)
        parts=[item(.4,1),item(2.,5),item(-1.,2)]
        combined=dict(train=dict(x=torch.cat([p['train']['x'] for p in parts]), labels=torch.tensor([[-100]+[7]*8])))
        ra=exp.update(a,oa,parts,self.settings(),0)
        rb=exp.update(b,ob,[combined],self.settings(),0)
        self.assertAlmostEqual(ra['loss'],rb['loss'],places=12)
        torch.testing.assert_close(a.weight,b.weight,rtol=1e-12,atol=1e-12)
        self.assertEqual(ra['supervised_tokens'],8)

    def test_gradient_checked_update_matches_regular_update(self):
        checked, regular = Toy(), Toy()
        checked_opt, regular_opt = self.optimizer(checked), self.optimizer(regular)
        parts = [item(.4,1),item(2.,5),item(-1.,2)]
        result = exp.update(checked,checked_opt,parts,self.settings(),0,check_gradients=True)
        control = exp.update(regular,regular_opt,parts,self.settings(),0)
        self.assertEqual(result['loss'],control['loss'])
        self.assertGreater(result['gradients']['lora_B']['norm'],0)
        self.assertEqual(result['gradients']['lora_B']['absent'],0)
        torch.testing.assert_close(checked.weight,regular.weight,rtol=0,atol=0)

    def test_gradient_check_rejects_disconnected_parameters_before_update(self):
        model = Toy()
        model.register_parameter('unused',torch.nn.Parameter(torch.tensor(.2,dtype=torch.float64)))
        optimizer = self.optimizer(model)
        original = {name:value.detach().clone() for name,value in model.named_parameters()}
        with self.assertRaisesRegex(ValueError,'Disconnected adapter gradients'):
            exp.update(model,optimizer,[item(2.,3)],self.settings(),0,check_gradients=True)
        for name,value in model.named_parameters():
            torch.testing.assert_close(value,original[name],rtol=0,atol=0)

    def test_resume_restores_optimizer_rng_and_single_sample_tail(self):
        exp.seed_all(42)
        model=Toy(random_noise=True); opt=self.optimizer(model)
        exp.update(model,opt,[item(.4,1),item(2.,5),item(-1.,2)],self.settings(),0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint.pt'; identity={'fixture':True}
            exp.save_checkpoint(p,model,opt,1,identity)
            expected_loss=exp.update(model,opt,[item(3.,4)],self.settings(),1)['loss']
            expected=model.weight.detach().clone()
            replay=Toy(random_noise=True); ropt=self.optimizer(replay)
            self.assertEqual(exp.restore_checkpoint(p,replay,ropt,identity),1)
            actual_loss=exp.update(replay,ropt,[item(3.,4)],self.settings(),1)['loss']
            self.assertEqual(expected_loss,actual_loss)
            torch.testing.assert_close(expected,replay.weight,rtol=0,atol=0)
            self.assertEqual(opt.state[model.weight]['step'],ropt.state[replay.weight]['step'])

    def test_restore_rejects_changed_identity_or_corrupt_file(self):
        model=Toy(); opt=self.optimizer(model)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint.pt'
            exp.save_checkpoint(p,model,opt,0,{'fixture':1})
            with self.assertRaisesRegex(ValueError,'identity/hash'):
                exp.restore_checkpoint(p,model,opt,{'fixture':2})
            with p.open('ab') as f: f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'identity/hash'):
                exp.restore_checkpoint(p,model,opt,{'fixture':1})

    def test_nonfinite_loss_does_not_update(self):
        model=Toy(); opt=self.optimizer(model); before=model.weight.detach().clone()
        with self.assertRaises(FloatingPointError):
            exp.update(model,opt,[item(float('nan'),1)],self.settings(),0)
        torch.testing.assert_close(before,model.weight)

    def test_exact_existing_answer_templates(self):
        row=dict(description='Detailed description.',openset=['happy','hopeful'])
        self.assertEqual(exp.target(row,'description')[1],row['description'])
        self.assertEqual(exp.target(row,'ovlabel')[1],"The character's emotional state is happy, hopeful.")


if __name__=='__main__': unittest.main()
