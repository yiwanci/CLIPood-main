import os
import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 遍历目录收集图像路径和标签
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_name)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], self.image_paths[idx]

def extract_clip_features(dataloader, device, model):
    """提取CLIP特征"""
    features, labels, image_paths = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, batch_labels, batch_paths) in enumerate(dataloader):
            images = images.to(device)
            # 提取图像特征
            image_features,_ = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu().numpy())
            labels.extend(batch_labels)
            image_paths.extend(batch_paths)
            if batch_idx % 10 == 0:
                print(f'Processed {batch_idx}/{len(dataloader)} batches')
    features = np.vstack(features)
    return features, labels, image_paths

def visualize_tsne(features, labels, image_paths, output_dir='tsne_results'):
    """使用t-SNE可视化特征"""
    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ 先标准化
    features = StandardScaler().fit_transform(features)
    # 2️⃣ 再用 PCA 降到 50 维去噪
    features = PCA(n_components=50).fit_transform(features)

    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=10,       # 更小的 perplexity 聚集度更高
        learning_rate=50,    # 适当降低学习率
        n_iter=2000,
        init="pca",
        random_state=42
    )
    features_2d = tsne.fit_transform(features)

    # 颜色映射
    unique_labels = sorted(set(labels))
    color_map = plt.cm.get_cmap('tab20', len(unique_labels))
    label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(15, 12))
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                    c=[label_to_color[label]],
                    label=label, s=20, alpha=0.8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('CLIP Features Visualization using t-SNE (tighter clusters)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clip_tsne_visualization.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # 保存坐标与标签
    np.save(os.path.join(output_dir, 'tsne_coordinates.npy'), features_2d)
    with open(os.path.join(output_dir, 'labels.txt'), 'w') as f:
        for path, label, coord in zip(image_paths, labels, features_2d):
            f.write(f"{path}\t{label}\t{coord[0]:.4f}\t{coord[1]:.4f}\n")

    print(f"Results saved to {output_dir}")
    return features_2d

def main():
    parser = argparse.ArgumentParser(description='CLIP特征 t-SNE 可视化（更紧凑聚类）')
    parser.add_argument('--dataset_root', type=str, default="/data/luhongtao/dataset/PACS/images/photo/",
                        help='数据集根路径（子目录为类别）')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_name', type=str, default='ViT-B/16',
                        choices=['RN50','RN101','RN50x4','RN50x16','RN50x64',
                                 'ViT-B/32','ViT-B/16','ViT-L/14'])
    parser.add_argument('--output_dir', type=str, default='tsne_results')
    args = parser.parse_args()

    if not os.path.exists(args.dataset_root):
        raise FileNotFoundError(f"数据集路径 {args.dataset_root} 不存在")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    print(f"加载 CLIP 模型: {args.model_name}")
    model, preprocess = clip.load(args.model_name, device=device)

    dataset = ImageDataset(args.dataset_root, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    print(f"数据集包含 {len(dataset)} 张图像，类别数: {len(set(dataset.labels))}")
    print("类别:", set(dataset.labels))

    print("开始提取 CLIP 特征...")
    features, labels, image_paths = extract_clip_features(dataloader, device, model)
    print(f"特征形状: {features.shape}")

    visualize_tsne(features, labels, image_paths, args.output_dir)

if __name__ == "__main__":
    main()
