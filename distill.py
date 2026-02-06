import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPDualMiLoss(nn.Module):
    """
    适配CLIP场景的双互信息损失模块。
    支持输入形状 [Templates, Classes, Dim] 的文本特征。
    """
    def __init__(self, visual_dim=768, text_dim=512, embed_dim=256, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
        # 投影层
        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        # 注意：如果输入的text特征已经是最终特征，这里只是一个进一步的映射
        self.text_proj = nn.Linear(text_dim, embed_dim)
        
        print("Initialized CLIPDualMiLoss (Adapted for Multi-Template).")

    def _compute_log_frobenius(self, gram_matrix):
        """计算 log2(||G||_F^2)"""
        trace = torch.trace(gram_matrix)
        if trace == 0:
            return torch.tensor(0.0, device=gram_matrix.device)
        gram_matrix_norm = gram_matrix / trace
        frobenius_norm_sq = torch.norm(gram_matrix_norm, p='fro') ** 2
        return torch.log2(frobenius_norm_sq + 1e-8)

    def compute_relation_loss(self, visual_emb, text_emb, relation_vec):
        """计算压缩损失 I(visual, text; relation)"""
        norm_visual = F.normalize(visual_emb, p=2, dim=-1)
        norm_text = F.normalize(text_emb, p=2, dim=-1)
        norm_relation = F.normalize(relation_vec, p=2, dim=-1)
        
        G_visual = torch.matmul(norm_visual, norm_visual.t())
        G_text = torch.matmul(norm_text, norm_text.t())
        G_relation = torch.matmul(norm_relation, norm_relation.t())
        
        G_tri = G_visual * G_text * G_relation
        
        loss_f = self._compute_log_frobenius(G_relation)
        loss_tri = self._compute_log_frobenius(G_tri)
        
        return -loss_f + loss_tri

    def compute_distill_loss(self, relation_t, relation_s):
        """计算蒸馏损失 I(relation_t; relation_s)"""
        norm_t = F.normalize(relation_t, p=2, dim=-1)
        norm_s = F.normalize(relation_s, p=2, dim=-1)

        G_t = torch.matmul(norm_t, norm_t.t())
        G_s = torch.matmul(norm_s, norm_s.t())
        G_ts = G_t * G_s

        loss_s = self._compute_log_frobenius(G_s)
        loss_t = self._compute_log_frobenius(G_t)
        loss_ts = self._compute_log_frobenius(G_ts)
        
        return -(-loss_t - loss_s + loss_ts)

    def forward(self, student_feats, teacher_feats, student_relation_model, teacher_relation_model):
        visual_feat_s, text_feat_s = student_feats
        visual_feat_t, text_feat_t = teacher_feats
        
        # 统一转为 float
        visual_feat_s = visual_feat_s.float()
        text_feat_s = text_feat_s.float()
        visual_feat_t = visual_feat_t.float()
        text_feat_t = text_feat_t.float()

        # --- 维度解析修正 ---
        # visual_feat_t: [Batch, N_tokens, Dim]
        B = visual_feat_t.shape[0]
        
        # text_feat_t: [Templates, Classes, Dim]
        # 我们需要获取类别数 C，它在第 1 维
        C = text_feat_t.shape[1]
        
        # --- 1. 计算关系向量 ---
        with torch.no_grad():
            # Teacher Model
            relation_t = teacher_relation_model(visual_feat_t, text_feat_t) # (B, C, D_rel)
        
        # Student Model
        relation_s = student_relation_model(visual_feat_s, text_feat_s) # (B, C, D_rel)
        
        # --- 2. 扁平化 ---
        # 现在的 reshape 是安全的，因为我们正确获取了 B 和 C
        flat_relation_t = relation_t.reshape(B * C, -1)
        flat_relation_s = relation_s.reshape(B * C, -1)
        
        # --- 3. 准备特征用于压缩损失 ---
        # 3.1 视觉特征: 取 [CLS] token
        visual_cls_t = visual_feat_t[:, 0, :] # (B, D_v)
        visual_emb_t = self.visual_proj(visual_cls_t) # (B, embed_dim)
        
        # 3.2 文本特征: 对所有模板求平均，得到每个类别的原型
        # input: [Templates, Classes, Dim] -> mean(0) -> [Classes, Dim]
        text_mean_t = text_feat_t.mean(dim=0) 
        text_emb_t = self.text_proj(text_mean_t) # (C, embed_dim)

        # 3.3 扩展以匹配 (B*C)
        # visual: (B, D) -> (B*C, D)
        flat_visual_t = visual_emb_t.unsqueeze(1).expand(-1, C, -1).reshape(B * C, -1)
        # text: (C, D) -> (B*C, D)
        flat_text_t = text_emb_t.unsqueeze(0).expand(B, -1, -1).reshape(B * C, -1)
        
        # --- 4. 计算损失 ---
        loss_r = self.compute_relation_loss(flat_visual_t, flat_text_t, flat_relation_t)
        loss_d = self.compute_distill_loss(flat_relation_t, flat_relation_s)
        
        total_loss = self.alpha * loss_r + self.beta * loss_d
        
        return total_loss, loss_r.item(), loss_d.item()