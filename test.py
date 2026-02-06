import torch
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

# ============================
# 配置区域
# ============================
IMAGE_PATH = "/data/luhongtao/dataset/PACS/images/photo/dog/n02106662_24786.jpg"

TEXT_PROMPTS = [
    "a photo of a dog",   # Index 0: 目标
    "a photo of a cat",
    "a photo of a car",
    "a photo of a tree",
    "a photo of a building"
]

MODEL_NAME = 'ViT-B/16'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# 关键修改：LAMBDA = 0
# 我们先只看“相似度”，彻底排除熵的干扰，看看 CLIP 到底认不认识狗
# ============================
LAMBDA = 0.0          
K_RATIO = 0.2
TAU = 0.01

print(f"Running on device: {DEVICE}")

# ... (加载模型部分不变) ...
print(f"Loading model {MODEL_NAME}...")
clip_model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model = clip_model.to(DEVICE)
model.eval()

# ... (图像处理部分不变) ...
try:
    raw_image = Image.open(IMAGE_PATH).convert('RGB')
except FileNotFoundError:
    print(f"Error: Cannot find image at {IMAGE_PATH}")
    sys.exit(1)

image_tensor = preprocess(raw_image).unsqueeze(0).to(DEVICE)
text_tokens = clip.tokenize(TEXT_PROMPTS).to(DEVICE)

# ============================
# 核心计算逻辑 (Pure Semantic Similarity)
# ============================
print("Executing Pure Similarity logic...")
with torch.no_grad(), torch.cuda.amp.autocast():
    # 1. 提取特征
    text_feat,_ = model.encode_text(text_tokens)
    text_feat = F.normalize(text_feat, dim=-1)

    _, image_feat_all = model.visual(image_tensor)
    patch_feat = image_feat_all[:, 1:, :] 
    patch_feat = F.normalize(patch_feat, dim=-1)
    B, N, D = patch_feat.shape

    # 2. 计算相似度 (Similarity)
    # 取第一个 prompt "a photo of a dog"
    target_text_feat = text_feat[0] 
    
    # [B, N] 纯相似度分数
    sim_score = patch_feat @ target_text_feat.unsqueeze(-1)
    sim_score = sim_score.squeeze(-1)

    # 3. 计算熵 (仅用于观察，不参与排名)
    logits = patch_feat @ text_feat.T
    p = F.softmax(logits / TAU, dim=-1)
    entropy = -(p * torch.log(p + 1e-6)).sum(dim=-1)

    # 4. 采样 (完全基于相似度)
    K = max(1, int(K_RATIO * N))
    
    # Foreground = 最像狗的
    fg_out = sim_score.topk(K, largest=True, dim=-1)
    fg_idx = fg_out.indices[0]
    
    # Background = 最不像狗的
    bg_out = sim_score.topk(K, largest=False, dim=-1)
    bg_idx = bg_out.indices[0]

    # 打印调试信息 (看看选出来的 Patch 到底像不像狗)
    # 相似度范围通常在 0.2 ~ 0.3 之间
    print(f"Top-K FG Avg Similarity: {fg_out.values[0].mean().item():.4f}")
    print(f"Top-K BG Avg Similarity: {bg_out.values[0].mean().item():.4f}")
    
    # 如果 FG 的相似度都很低 (<0.15)，说明 CLIP 可能根本没觉得这里有狗
    if fg_out.values[0].mean().item() < 0.15:
        print("[WARNING] Target similarity is very low. Model might be missing the object.")

# ... (Mask 准备和绘图部分不变，直接用下面的) ...
# ============================
# 绘图 (手动混合)
# ============================
grid_size = int(np.sqrt(N))
fg_map = torch.zeros((grid_size, grid_size), device=DEVICE)
bg_map = torch.zeros((grid_size, grid_size), device=DEVICE)
fg_map.view(-1)[fg_idx] = 1.0
bg_map.view(-1)[bg_idx] = 1.0

input_H, input_W = image_tensor.shape[2:]
fg_mask = F.interpolate(fg_map.unsqueeze(0).unsqueeze(0), size=(input_H, input_W), mode='nearest').squeeze().cpu().numpy()
bg_mask = F.interpolate(bg_map.unsqueeze(0).unsqueeze(0), size=(input_H, input_W), mode='nearest').squeeze().cpu().numpy()

mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1, 3, 1, 1)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1, 3, 1, 1)
img_denorm = image_tensor * std + mean
img_to_show = torch.clamp(img_denorm, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()

def blend_image(image, mask, color, alpha=0.5):
    out = image.copy()
    mask_expanded = mask[:, :, None]
    color_np = np.array(color).reshape(1, 1, 3)
    weight = mask_expanded * alpha
    out = out * (1 - weight) + color_np * weight
    return np.clip(out, 0, 1)

img_fg_vis = blend_image(img_to_show, fg_mask, color=[0, 1, 0], alpha=0.6)
img_bg_vis = blend_image(img_to_show, bg_mask, color=[1, 0, 0], alpha=0.6)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_to_show)
axes[0].set_title("Input")
axes[0].axis('off')

axes[1].imshow(img_fg_vis)
axes[1].set_title("Pure Similarity (Focus on 'Dog')")
axes[1].axis('off')

axes[2].imshow(img_bg_vis)
axes[2].set_title("Low Similarity (Background)")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("similarity_only.png")
print("Saved to similarity_only.png")
plt.show()