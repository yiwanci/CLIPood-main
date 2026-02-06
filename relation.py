import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPRelationModuleAgentV2(nn.Module):
    def __init__(self, visual_dim=768, text_dim=512, embed_dim=256, num_agent_tokens=16, num_heads=4):
        super().__init__()
        # ... (初始化部分保持不变) ...
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_agent_tokens = num_agent_tokens
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 投影层
        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        # 注意：如果输入的 text_feat 已经是 CLIP 的最终输出，可能不需要再 drastic 地投影，
        # 但为了对齐维度和训练稳定性，保留一个 Linear 层通常更好。
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # Agent Tokens
        self.agent_tokens = nn.Parameter(torch.randn(1, num_agent_tokens, embed_dim))

        # Norms
        self.visual_norm = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.agent_norm = nn.LayerNorm(embed_dim)

        # Scales & Params
        self.scale = self.head_dim ** -0.5
        self.pooling_scale = embed_dim ** -0.5
        self.gamma1 = nn.Parameter(torch.zeros(1).normal_(mean=0, std=0.1))
        self.gamma2 = nn.Parameter(torch.zeros(1).normal_(mean=0, std=0.1))
        self.gamma_init = 0.1

        # Attention Projections
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim)

    def agent_mask_pooling(self, source, scores):
        """
        source: (Batch, Len, Dim)
        scores: (Batch, Len, Num_Agents)
        """
        attn = F.softmax(scores, dim=1) 
        # (B, Len, K).T @ (B, Len, D) -> (B, K, D)
        aggregated = torch.einsum('blk,bld->bkd', attn, source)
        return aggregated

    def forward(self, visual_feat, text_feat):
        """
        Args:
            visual_feat: [B, N_v, visual_dim] (图像特征)
            text_feat:   [P, C, text_dim] (80个模板的特征) 
                         P=80 (Templates), C=Num_Classes
        """
        # --- 1. 维度调整 ---
        # 用户输入是 [80, C, Dim]，我们需要转置成 [C, 80, Dim]
        # 这样对于每个类别 C，我们有 80 个特征作为序列供 Agent 聚合
        text_feat = text_feat.permute(1, 0, 2)  # [P, C, D] -> [C, P, D]

        B, N_v, _ = visual_feat.shape
        C, P, _ = text_feat.shape
        K = self.num_agent_tokens

        # --- 2. 基础特征嵌入 ---
        visual_emb = self.visual_norm(self.visual_proj(visual_feat)) # (B, N_v, D)
        text_emb = self.text_norm(self.text_proj(text_feat))         # (C, P, D)
        base_agents = self.agent_norm(self.agent_tokens)             # (1, K, D)

        # --- 3. 双向聚合 (Double Aggregation) ---

        # === Side A: Text Aggregation ===
        # Agent 学习这 80 个模板中，哪些模板对当前类别更重要
        agents_for_text = base_agents.expand(C, -1, -1) # (C, K, D)
        
        # 计算相似度: (C, K, D) @ (C, P, D)^T -> (C, K, P) -> (C, P, K)
        scores_text = torch.einsum('ckd,cpd->cpk', agents_for_text, text_emb) * self.pooling_scale
        
        # 聚合: 此时 Agent 融合了 80 个模板的信息
        masked_text = self.agent_mask_pooling(text_emb, scores_text) # (C, K, D)

        # === Side B: Visual Aggregation ===
        # Agent 学习图像中哪些 Patch 更重要
        agents_for_vis = base_agents.expand(B, -1, -1) # (B, K, D)
        scores_vis = torch.einsum('bkd,bnd->bnk', agents_for_vis, visual_emb) * self.pooling_scale
        masked_vis = self.agent_mask_pooling(visual_emb, scores_vis) # (B, K, D)

        # === Side C: Fusion ===
        # 融合 Gamma 参数
        gamma = torch.exp(self.gamma1) - torch.exp(self.gamma2) + self.gamma_init
        
        # 广播加法生成 Informed Agents
        # (1, 1, K, D) + (1, C, K, D) + (B, 1, K, D) -> (B, C, K, D)
        informed_agents = (
            base_agents.unsqueeze(0) + 
            gamma * masked_text.unsqueeze(0) + 
            masked_vis.unsqueeze(1)
        ) 

        # --- 4. 最终查询 (Cross Attention) ---
        # 将 B 和 C 合并处理以加速: (B*C, ...)
        q_in = visual_emb.unsqueeze(1).expand(-1, C, -1, -1).reshape(B*C, N_v, -1)
        kv_in = informed_agents.reshape(B*C, K, -1)

        q = self.to_q(q_in)
        k = self.to_k(kv_in)
        v = self.to_v(kv_in)

        # Multi-head attention 
        q = q.view(B*C, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B*C, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B*C, K, self.num_heads, self.head_dim).transpose(1, 2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v) # (B*C, H, N_v, D_h)
        
        out = out.transpose(1, 2).reshape(B*C, N_v, -1)
        relation_feat = self.to_out(out)
        
        # 恢复维度 (B, C, N_v, D)
        relation_feat = relation_feat.view(B, C, N_v, -1)
        
        # Mean over patches -> (B, C, D)
        relation_feat = relation_feat.mean(dim=2)
        
        return F.normalize(relation_feat, p=2, dim=-1)


if __name__ == "__main__":
    # 简单的测试代码
    model = CLIPRelationModuleAgentV2()
    visual_feat = torch.randn(2, 49, 768)  # Batch of 2, 49 patches, 768-dim
    text_feat = torch.randn(80, 10, 512)    # 80 templates, 10 classes, 512-dim

    relation_feat = model(visual_feat, text_feat)
    print(relation_feat.shape)  # Expected: (2, 10, D_relation)