import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from ...singleTask.model.router import router
from ...singleTask.model.spectral_band import GraphSpectralBandDecomposer, BandRouter, apply_band_masking


class Atsuko(nn.Module):
    def __init__(self, args):
        super(Atsuko, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375

        elif args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500

        elif args.dataset_name == 'sims':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 39, 39, 39
            else:
                self.len_l, self.len_v, self.len_a = 39, 55, 400

        elif args.dataset_name == 'sims_v2':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 232, 925

        elif args.dataset_name == 'mintrec':
            # MIntRec: 支持 aligned 和 unaligned 两种模式
            if args.need_data_aligned:
                aligned_len = int(getattr(args, 'aligned_seq_len', 50))
                self.len_l, self.len_v, self.len_a = aligned_len, aligned_len, aligned_len
            else:
                # unaligned 模式：从配置读取 audio_max_len / video_max_len
                self.len_l = int(getattr(args, 'aligned_seq_len', 50))  # text 仍用对齐长度
                self.len_v = int(getattr(args, 'video_max_len', 120))
                self.len_a = int(getattr(args, 'audio_max_len', 220))

        else:
            raise ValueError(f"Unknown dataset_name: {args.dataset_name}")

        self.aligned = args.need_data_aligned
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.fusion_method = args.fusion_method

        # # output_dim:
        # # - regression: 1
        # # - classification (e.g., MIntRec): num_classes
        # if getattr(args, "train_mode", "regression") == "classification":
        #     output_dim = int(getattr(args, "output_dim", getattr(args, "num_classes", 0)))
        #     if output_dim <= 1:
        #         raise ValueError(f"classification requires output_dim/num_classes > 1, got {output_dim}")
        # else:
        #     output_dim = 1

        # output_dim:
        # - regression: 1
        # - softlabel: 7 (distribution over {-3,-2,-1,0,1,2,3})
        # - classification (e.g., MIntRec): num_classes
        train_mode = getattr(args, "train_mode", "regression")
        if train_mode == "classification":
            output_dim = int(getattr(args, "output_dim", getattr(args, "num_classes", 0)))
            if output_dim <= 1:
                raise ValueError(f"classification requires output_dim/num_classes > 1, got {output_dim}")
        elif train_mode == "softlabel":
            output_dim = int(getattr(args, "softlabel_num_classes", 7))
            if output_dim != 7:
                raise ValueError(f"softlabel expects output_dim=7, got {output_dim}")
        else:
            output_dim = 1


        self.args = args

        # ABCM 开关：是否使用消融式互补重要性计算
        self.enable_abcm = bool(getattr(self.args, "enable_abcm", True))

        # temperature（给 band_router / Router_spec 用）
        self.t = float(getattr(self.args, "temperature", 0.1))
        
        # Conv1d(padding=0) 会缩短时间长度：T' = T - k + 1
        k_l = int(getattr(self.args, "conv1d_kernel_size_l", 1))
        k_v = int(getattr(self.args, "conv1d_kernel_size_v", 1))
        k_a = int(getattr(self.args, "conv1d_kernel_size_a", 1))
        
        self.len_l_enc = self.len_l - k_l + 1
        self.len_v_enc = self.len_v - k_v + 1
        self.len_a_enc = self.len_a - k_a + 1
        
        if self.len_l_enc <= 0 or self.len_v_enc <= 0 or self.len_a_enc <= 0:
            raise ValueError(
                f"Invalid conv kernel sizes: got len_enc=({self.len_l_enc},{self.len_v_enc},{self.len_a_enc}) "
                f"from len=({self.len_l},{self.len_v},{self.len_a}) and k=({k_l},{k_v},{k_a})"
            )


        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        self.self_attentions_l = self.get_network(self_type='l')
        self.self_attentions_v = self.get_network(self_type='v')
        self.self_attentions_a = self.get_network(self_type='a')

        self.proj1_l = nn.Linear(self.d_l, self.d_l)
        self.proj2_l = nn.Linear(self.d_l, self.d_l)
        self.out_layer_l = nn.Linear(self.d_l, output_dim)
        self.proj1_v = nn.Linear(self.d_l, self.d_l)
        self.proj2_v = nn.Linear(self.d_l, self.d_l)
        self.out_layer_v = nn.Linear(self.d_l, output_dim)
        self.proj1_a = nn.Linear(self.d_l, self.d_l)
        self.proj2_a = nn.Linear(self.d_l, self.d_l)
        self.out_layer_a = nn.Linear(self.d_l, output_dim)

        if self.fusion_method == "sum": 
            self.proj1_c = nn.Linear(self.d_l, self.d_l)
            self.proj2_c = nn.Linear(self.d_l, self.d_l)
            self.out_layer_c = nn.Linear(self.d_l, output_dim)
        elif self.fusion_method == "concat":
            self.proj1_c = nn.Linear(self.d_l*3, self.d_l*3)
            self.proj2_c = nn.Linear(self.d_l*3, self.d_l*3)
            self.out_layer_c = nn.Linear(self.d_l*3, output_dim)

        # --- ABCM 改进：添加子模态融合头（仅当 enable_abcm=True 时创建）---
        if self.enable_abcm:
            # 无论主融合方法如何，我们都为子网络使用 'sum' 逻辑（输入 self.d_l）
            # 因为它们仅用于计算损失
            # 1. V+A (没有 L)
            self.proj1_c_va = nn.Linear(self.d_l, self.d_l)
            self.proj2_c_va = nn.Linear(self.d_l, self.d_l)
            self.out_layer_c_va = nn.Linear(self.d_l, output_dim)
            # 2. L+A (没有 V)
            self.proj1_c_la = nn.Linear(self.d_l, self.d_l)
            self.proj2_c_la = nn.Linear(self.d_l, self.d_l)
            self.out_layer_c_la = nn.Linear(self.d_l, output_dim)
            # 3. L+V (没有 A)
            self.proj1_c_lv = nn.Linear(self.d_l, self.d_l)
            self.proj2_c_lv = nn.Linear(self.d_l, self.d_l)
            self.out_layer_c_lv = nn.Linear(self.d_l, output_dim)
        # --- ABCM 改进结束 ---

        self.transfer_a_ali = nn.Linear(self.len_a, self.len_l)
        self.transfer_v_ali = nn.Linear(self.len_v, self.len_l)

        # ===================== Spectral-Band Hierarchical Routing (optional) =====================
        self.enable_spectral_bands = getattr(self.args, "enable_spectral_bands", False)
        if self.enable_spectral_bands:
            self.num_bands = int(getattr(self.args, "num_bands", 3))
            self.band_partition = getattr(self.args, "band_partition", "equal_energy")  # "equal_energy" | "ratio"
            self.band_edge_ratios = getattr(self.args, "band_edge_ratios", [1/3, 2/3])
            self.band_mask_rate = float(getattr(self.args, "band_mask_rate", 0.0))

            # (Recommended) once you precompute edges offline, set these 3 lists in config
            self.band_edges_l = getattr(self.args, "band_edges_l", None)
            self.band_edges_v = getattr(self.args, "band_edges_v", None)
            self.band_edges_a = getattr(self.args, "band_edges_a", None)

            # band decomposers: time length per modality
            self.spec_l = GraphSpectralBandDecomposer(
                length=self.len_l_enc,
                num_bands=self.num_bands,
                partition=self.band_partition,
                band_edges=self.band_edges_l,
                band_edge_ratios=self.band_edge_ratios,
                normalized_laplacian=True,
                basis_device="cpu",
            )
            self.spec_v = GraphSpectralBandDecomposer(
                length=self.len_v_enc,
                num_bands=self.num_bands,
                partition=self.band_partition,
                band_edges=self.band_edges_v,
                band_edge_ratios=self.band_edge_ratios,
                normalized_laplacian=True,
                basis_device="cpu",
            )
            self.spec_a = GraphSpectralBandDecomposer(
                length=self.len_a_enc,
                num_bands=self.num_bands,
                partition=self.band_partition,
                band_edges=self.band_edges_a,
                band_edge_ratios=self.band_edge_ratios,
                normalized_laplacian=True,
                basis_device="cpu",
            )

            # band routers (one per modality)
            self.band_router_l = BandRouter(embed_dim=self.d_l, num_bands=self.num_bands, temperature=self.t)
            self.band_router_v = BandRouter(embed_dim=self.d_l, num_bands=self.num_bands, temperature=self.t)
            self.band_router_a = BandRouter(embed_dim=self.d_l, num_bands=self.num_bands, temperature=self.t)

            # modality router (use pooled fused features; avoids huge flatten)
            self.Router_spec = router(self.d_l * 3, 3, self.t)
            self.spec_router_input = getattr(self.args, "spec_router_input", "pre_attn")  # "pre_attn" | "post_attn"
        else:
            self.Router = router(self.orig_d_l * self.len_l + self.orig_d_a * self.len_l + self.orig_d_v * self.len_l, 3, self.args.temperature)
        # =========================================================================================




    def get_network(self, self_type='l', layers=-1):
        if self_type == 'l':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type == 'v':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def get_net(self, name):
        return getattr(self, name)

    def forward(self, text, audio, video):
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        if not self.aligned:
            audio_ = self.transfer_a_ali(audio.permute(0, 2, 1)).permute(0, 2, 1)
            video_ = self.transfer_v_ali(video.permute(0, 2, 1)).permute(0, 2, 1)
            m_i = torch.cat((text, video_, audio_), dim=2)
        else:
            m_i = torch.cat((text, video, audio), dim=2)
        m_w = None if self.enable_spectral_bands else self.Router(m_i)

        # proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        # proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        # proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        k_l = int(getattr(self.args, "conv1d_kernel_size_l", 1))
        k_a = int(getattr(self.args, "conv1d_kernel_size_a", 1))
        k_v = int(getattr(self.args, "conv1d_kernel_size_v", 1))
        
        proj_x_l = self.proj_l(x_l) if (self.orig_d_l != self.d_l or k_l != 1) else x_l
        proj_x_a = self.proj_a(x_a) if (self.orig_d_a != self.d_a or k_a != 1) else x_a
        proj_x_v = self.proj_v(x_v) if (self.orig_d_v != self.d_v or k_v != 1) else x_v


        c_l = self.encoder_c(proj_x_l)  # [B, D, T_l]
        c_v = self.encoder_c(proj_x_v)  # [B, D, T_v]
        c_a = self.encoder_c(proj_x_a)  # [B, D, T_a]

        band_w_l = band_w_v = band_w_a = None  # for logging/regularization

        if self.enable_spectral_bands:
            # convert to [B, T, D] for spectral decomposition
            seq_l = c_l.transpose(1, 2)  # [B, Tl, D]
            seq_v = c_v.transpose(1, 2)  # [B, Tv, D]
            seq_a = c_a.transpose(1, 2)  # [B, Ta, D]

            # 1) split into bands
            bands_l = self.spec_l.split_bands(seq_l)  # list of [B,Tl,D]
            bands_v = self.spec_v.split_bands(seq_v)
            bands_a = self.spec_a.split_bands(seq_a)

            # 2) band router from pooled band embeddings
            pooled_l = torch.stack([b.mean(dim=1) for b in bands_l], dim=1)  # [B,M,D]
            pooled_v = torch.stack([b.mean(dim=1) for b in bands_v], dim=1)
            pooled_a = torch.stack([b.mean(dim=1) for b in bands_a], dim=1)

            band_w_l = self.band_router_l(pooled_l)  # [B,M]
            band_w_v = self.band_router_v(pooled_v)
            band_w_a = self.band_router_a(pooled_a)

            if self.training and self.band_mask_rate > 0:
                band_w_l = apply_band_masking(band_w_l, self.band_mask_rate)
                band_w_v = apply_band_masking(band_w_v, self.band_mask_rate)
                band_w_a = apply_band_masking(band_w_a, self.band_mask_rate)

            # 3) fuse bands back to one sequence per modality (SSR-style semantic-aware fusion)
            stack_l = torch.stack(bands_l, dim=1)  # [B,M,Tl,D]
            stack_v = torch.stack(bands_v, dim=1)  # [B,M,Tv,D]
            stack_a = torch.stack(bands_a, dim=1)  # [B,M,Ta,D]

            seq_l_fused = (stack_l * band_w_l.view(stack_l.size(0), self.num_bands, 1, 1)).sum(dim=1)  # [B,Tl,D]
            seq_v_fused = (stack_v * band_w_v.view(stack_v.size(0), self.num_bands, 1, 1)).sum(dim=1)  # [B,Tv,D]
            seq_a_fused = (stack_a * band_w_a.view(stack_a.size(0), self.num_bands, 1, 1)).sum(dim=1)  # [B,Ta,D]

            # 4) modality router on pooled fused features (stable, avoids huge flatten)
            f_l = seq_l_fused.mean(dim=1)  # [B,D]
            f_v = seq_v_fused.mean(dim=1)
            f_a = seq_a_fused.mean(dim=1)
            # m_w = self.Router_spec(torch.cat([f_l, f_v, f_a], dim=1))  # [B,3]
            if self.spec_router_input == "pre_attn":
                m_w = self.Router_spec(torch.cat([f_l, f_v, f_a], dim=1))  # [B,3]

            # 5) back to [T,B,D] for the original transformer blocks
            c_l = seq_l_fused.permute(1, 0, 2)
            c_v = seq_v_fused.permute(1, 0, 2)
            c_a = seq_a_fused.permute(1, 0, 2)
        else:
            # 原始路径：直接送入 Transformer
            c_l = c_l.permute(2, 0, 1)
            c_v = c_v.permute(2, 0, 1)
            c_a = c_a.permute(2, 0, 1)


        c_l_att = self.self_attentions_l(c_l)
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]
        c_v_att = self.self_attentions_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]
        c_a_att = self.self_attentions_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        # 保存原始 router 的输出（基于 raw 拼接输入 m_i）
        # m_w_raw = m_w

        # 若开启频带分解：用“模态级 Router_spec”产生最终用于融合的权重
        # if self.enable_spectral_bands:
        #     m_w = self.Router_spec(torch.cat([c_l_att, c_v_att, c_a_att], dim=1))
        if self.enable_spectral_bands and self.spec_router_input == "post_attn":
            m_w = self.Router_spec(torch.cat([c_l_att, c_v_att, c_a_att], dim=1))  # [B,3]

        if m_w is None:
            raise RuntimeError("m_w is None: please set spec_router_input to 'pre_attn' or 'post_attn'.")
        
        # === 模态缺失处理：根据 modality_mask 屏蔽 Router 权重 ===
        # 只有在存在被禁用的模态时才执行，全模态时完全跳过以保持原有行为
        modality_mask = getattr(self.args, 'modality_mask', {})
        use_text = modality_mask.get('text', True) if isinstance(modality_mask, dict) else True
        use_vision = modality_mask.get('vision', True) if isinstance(modality_mask, dict) else True
        use_audio = modality_mask.get('audio', True) if isinstance(modality_mask, dict) else True
        
        # 检查是否存在被禁用的模态
        has_masked_modality = not (use_text and use_vision and use_audio)
        
        if has_masked_modality:
            # 创建 mask tensor: [text, vision, audio] 对应 m_w 的 [0, 1, 2]
            mask_tensor = torch.tensor([use_text, use_vision, use_audio], dtype=m_w.dtype, device=m_w.device)
            m_w = m_w * mask_tensor.unsqueeze(0)  # [B, 3] * [1, 3]
            
            # 重新归一化（避免除零）
            m_w_sum = m_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
            m_w = m_w / m_w_sum
        # === 模态缺失处理结束 ===
            
        l_proj = self.proj2_l(
            F.dropout(F.relu(self.proj1_l(c_l_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        l_proj += c_l_att
        logits_l = self.out_layer_l(l_proj)
        v_proj = self.proj2_v(
            F.dropout(F.relu(self.proj1_v(c_v_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        v_proj += c_v_att
        logits_v = self.out_layer_v(v_proj)
        a_proj = self.proj2_a(
            F.dropout(F.relu(self.proj1_a(c_a_att), inplace=True), p=self.output_dropout,
                      training=self.training))
        a_proj += c_a_att
        logits_a = self.out_layer_a(a_proj)

        # --- ABCM 改进：计算子模态 logits（仅当 enable_abcm=True 时计算）---
        if self.enable_abcm:
            # (c_l_att, c_v_att, c_a_att 此时的 shape 都是 [batch_size, self.d_l])

            # 1. V+A (没有 L)
            c_fusion_va = c_v_att + c_a_att
            c_proj_va = self.proj2_c_va(
                F.dropout(F.relu(self.proj1_c_va(c_fusion_va), inplace=True), p=self.output_dropout,
                        training=self.training))
            c_proj_va += c_fusion_va
            logits_va = self.out_layer_c_va(c_proj_va)

            # 2. L+A (没有 V)
            c_fusion_la = c_l_att + c_a_att
            c_proj_la = self.proj2_c_la(
                F.dropout(F.relu(self.proj1_c_la(c_fusion_la), inplace=True), p=self.output_dropout,
                        training=self.training))
            c_proj_la += c_fusion_la
            logits_la = self.out_layer_c_la(c_proj_la)

            # 3. L+V (没有 A)
            c_fusion_lv = c_l_att + c_v_att
            c_proj_lv = self.proj2_c_lv(
                F.dropout(F.relu(self.proj1_c_lv(c_fusion_lv), inplace=True), p=self.output_dropout,
                        training=self.training))
            c_proj_lv += c_fusion_lv
            logits_lv = self.out_layer_c_lv(c_proj_lv)
        else:
            logits_va = logits_la = logits_lv = None
        # --- ABCM 改进结束 ---


        # if self.fusion_method == "sum": 
        #     for i in range(m_w.shape[0]):
        #         c_f = c_l_att[i] * m_w[i][0] + c_v_att[i] * m_w[i][1] + c_a_att[i] * m_w[i][2]
        #         if i == 0: c_fusion = c_f.unsqueeze(0)
        #         else: c_fusion = torch.cat([c_fusion, c_f.unsqueeze(0)], dim=0)   
        # elif self.fusion_method == "concat":        
        #     for i in range(m_w.shape[0]):
        #         c_f = torch.cat([c_l_att[i] * m_w[i][0], c_v_att[i] * m_w[i][1], c_a_att[i] * m_w[i][2]], dim=0) * 3
        #         if i == 0: c_fusion = c_f.unsqueeze(0)
        #         else: c_fusion = torch.cat([c_fusion, c_f.unsqueeze(0)], dim=0)   
        if self.fusion_method == "sum":
            # c_*_att: [B, d]
            # m_w: [B, 3]
            c_fusion = (
                c_l_att * m_w[:, 0:1] +
                c_v_att * m_w[:, 1:2] +
                c_a_att * m_w[:, 2:3]
            )
        elif self.fusion_method == "concat":
            c_fusion = torch.cat(
                [
                    c_l_att * m_w[:, 0:1],
                    c_v_att * m_w[:, 1:2],
                    c_a_att * m_w[:, 2:3],
                ],
                dim=1
            ) * 3
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")




        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)


        res = {
            'logits_c': logits_c,
            'logits_l': logits_l,
            'logits_v': logits_v,
            'logits_a': logits_a,

            # --- ABCM 改进：添加子模态 logits 到输出 ---
            'logits_va': logits_va,
            'logits_la': logits_la,
            'logits_lv': logits_lv,
            # --- ABCM 改进结束 ---
            
            'channel_weight': m_w,
            # 'raw_channel_weight': m_w_raw,

            # 频带级权重（若未开启则为 None）
            'band_weights_l': band_w_l,
            'band_weights_v': band_w_v,
            'band_weights_a': band_w_a,
            'c_proj': c_proj,
            'l_proj': l_proj,
            'v_proj': v_proj,
            'a_proj': a_proj,
            'c_fea': c_fusion,
        }
        return res
