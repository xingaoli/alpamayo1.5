import numpy as np
import json
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'].insert(0, "AR PL UMing CN")
plt.rcParams['axes.unicode_minus'] = False


class CocCluster:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.clusterer = None
        self.umap_embeddings = None

    def prepare_data(self, data_path):
        """加载数据，展开 coc_lists，构建样本列表和索引映射"""
        print("正在加载数据...")
        raw_data = json.load(open(data_path, 'r'))

        self.samples = []        # coc text
        self.clip_ids = []       # 每个 coc 对应的 clip_id
        self.frame_indices = []  # 每个 coc 对应的 frame index (key)
        self.raw_entries = []    # 原始 entry (用于回溯)

        for entry in raw_data:
            clip_id = entry['clip_id']
            for frame_idx, coc_text in entry['coc_lists'].items():
                self.samples.append(coc_text)
                self.clip_ids.append(clip_id)
                self.frame_indices.append(frame_idx)
                self.raw_entries.append(entry)

        print(f"总样本数: {len(self.samples)}, 来自 {len(set(self.clip_ids))} 个 clip")
        return self.samples

    def encode(self):
        print("正在生成语义向量...")
        self.embeddings = self.model.encode(
            self.samples, show_progress_bar=True, convert_to_tensor=False
        )
        return self.embeddings

    def reduce_dimensionality(self, n_neighbors=15, min_dist=0.1, n_components=5):
        print("正在降维...")
        n_neighbors = min(n_neighbors, len(self.samples) - 1)
        umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=min(n_components, len(self.samples) - 1),
            metric='cosine',
            random_state=42
        )
        self.umap_embeddings = umap_reducer.fit_transform(self.embeddings)
        return self.umap_embeddings

    def cluster(self, min_cluster_size=2, min_samples=1):
        print("正在进行语义聚类...")
        n_samples = len(self.umap_embeddings)

        # 自动计算合理参数 — 用更小的 min_cluster_size 让小簇也能被发现
        min_cluster_size = max(5, n_samples // 200)
        min_samples = max(2, min_cluster_size // 3)

        print(f"样本数量: {n_samples}")
        print(f"min_cluster_size: {min_cluster_size}")
        print(f"min_samples: {min_samples}")

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        self.cluster_labels = self.clusterer.fit_predict(self.umap_embeddings)

        unique_labels = set(self.cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(self.cluster_labels == -1)
        print(f"初始聚类数量: {n_clusters}")
        print(f"噪声点数量: {n_noise} ({n_noise / len(self.cluster_labels) * 100:.1f}%)")

        # 保留噪声点,不做二次分配
        if n_noise > 0:
            print(f"注意: 噪声点将保留,不强制分配到最近的簇")

        return self.cluster_labels

    def _assign_noise_to_nearest_cluster(self):
        """将噪声点分配到 UMAP 空间中最近的簇"""
        noise_mask = self.cluster_labels == -1
        n_noise = np.sum(noise_mask)
        if n_noise == 0:
            return

        valid_mask = ~noise_mask
        valid_labels = self.cluster_labels[valid_mask]
        unique_valid = set(valid_labels) - {-1}
        if not unique_valid:
            return

        # 计算每个簇的质心
        centroids = {}
        for lbl in unique_valid:
            mask = self.cluster_labels == lbl
            centroids[lbl] = self.umap_embeddings[mask].mean(axis=0)

        # 对每个噪声点找最近质心
        centroid_matrix = np.array([centroids[lbl] for lbl in sorted(unique_valid)])
        centroid_labels = sorted(unique_valid)

        noise_embeddings = self.umap_embeddings[noise_mask]
        # 计算距离矩阵: (n_noise, n_clusters)
        diffs = noise_embeddings[:, np.newaxis, :] - centroid_matrix[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)

        nearest_indices = np.argmin(dists, axis=1)
        assigned_labels = np.array([centroid_labels[i] for i in nearest_indices])

        self.cluster_labels[noise_mask] = assigned_labels

        # 统计分配了多少
        reassigned = {}
        for lbl in assigned_labels:
            reassigned[int(lbl)] = reassigned.get(int(lbl), 0) + 1

        n_clusters = len(set(self.cluster_labels))
        print(f"\n噪声点已全部分配到最近的簇")
        print(f"最终聚类数量: {n_clusters}")
        print(f"噪声点: 0")
        top_reassigned = sorted(reassigned.items(), key=lambda x: -x[1])[:5]
        for lbl, cnt in top_reassigned:
            print(f"  簇 {lbl}: 新增 {cnt} 个样本")

    def merge_clusters_by_centroid(self, similarity_threshold=0.85):
        """基于 embedding 质心余弦相似度合并语义相近的簇"""
        print(f"\n正在基于质心相似度合并簇 (阈值: {similarity_threshold})...")

        unique_labels = sorted(set(self.cluster_labels))
        # 用原始 embedding 算质心，语义更准确
        centroids = {}
        for lbl in unique_labels:
            mask = self.cluster_labels == lbl
            centroids[lbl] = self.embeddings[mask].mean(axis=0)

        # 建立并查集
        parent = {lbl: lbl for lbl in unique_labels}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                # 小簇并入大簇
                size_a = np.sum(self.cluster_labels == ra)
                size_b = np.sum(self.cluster_labels == rb)
                if size_a < size_b:
                    ra, rb = rb, ra
                parent[rb] = ra

        # 计算所有簇对之间的质心余弦相似度
        label_list = sorted(unique_labels)
        n = len(label_list)
        merge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                li, lj = label_list[i], label_list[j]
                ci, cj = centroids[li], centroids[lj]
                sim = np.dot(ci, cj) / (np.linalg.norm(ci) * np.linalg.norm(cj))
                if sim >= similarity_threshold:
                    union(li, lj)
                    merge_count += 1

        # 重新编号
        root_set = sorted(set(find(lbl) for lbl in unique_labels))
        old_to_new = {root: new_id for new_id, root in enumerate(root_set)}

        new_labels = np.array([old_to_new[find(lbl)] for lbl in self.cluster_labels])
        self.cluster_labels = new_labels

        print(f"合并了 {merge_count} 对簇")
        print(f"簇数量: {len(unique_labels)} → {len(root_set)}")

        # 打印合并详情（每个新簇由哪些旧簇组成）
        old_to_root = {}
        for lbl in unique_labels:
            root = find(lbl)
            old_to_root.setdefault(root, []).append(lbl)

        merged_groups = {r: groups for r, groups in old_to_root.items() if len(groups) > 1}
        if merged_groups:
            print("\n合并详情:")
            for root, groups in merged_groups.items():
                new_id = old_to_new[root]
                # 取代表样本
                mask = np.isin(self.cluster_labels, new_id)
                centroid = self.embeddings[mask].mean(axis=0)
                sims = [np.dot(centroid, e) / (np.linalg.norm(centroid) * np.linalg.norm(e))
                        for e in self.embeddings[mask]]
                rep_idx = np.where(mask)[0][np.argmax(sims)]
                rep_text = self.samples[rep_idx][:80]

                print(f"  新簇{new_id}: 旧簇[{', '.join(str(g) for g in groups)}] → {rep_text}")

        return self.cluster_labels

    def visualize_clusters(self, save_path):
        """t-SNE 可视化聚类结果"""
        if self.umap_embeddings is None:
            raise ValueError("请先执行降维")

        n_samples = len(self.umap_embeddings)
        print(f"正在可视化 {n_samples} 个样本...")

        if n_samples < 100:
            perplexity = min(30, max(5, n_samples // 3))
        elif n_samples < 1000:
            perplexity = 30
        elif n_samples < 10000:
            perplexity = 50
        else:
            perplexity = 100

        print(f"使用 t-SNE, perplexity={perplexity}")
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                        max_iter=1000, learning_rate='auto')
            vis_embeddings = tsne.fit_transform(self.umap_embeddings)
        except Exception as e:
            print(f"t-SNE 失败: {e}, 使用 PCA")
            pca = PCA(n_components=2, random_state=42)
            vis_embeddings = pca.fit_transform(self.umap_embeddings)

        fig, ax = plt.subplots(figsize=(18, 12))
        unique_labels = set(self.cluster_labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        cluster_sizes = {label: np.sum(self.cluster_labels == label) for label in unique_labels}

        for i, label in enumerate(unique_labels):
            if label == -1:
                color = 'gray'
                label_name = f'Noise ({cluster_sizes[label]})'
                marker = 'x'
                alpha, s = 0.4, 30
            else:
                color = colors[i]
                label_name = f'Cluster {label} ({cluster_sizes[label]})'
                marker = 'o'
                alpha, s = 0.7, 40

            mask = self.cluster_labels == label
            ax.scatter(vis_embeddings[mask, 0], vis_embeddings[mask, 1],
                       c=[color], label=label_name, alpha=alpha, s=s, marker=marker,
                       edgecolors='black', linewidth=0.3)

        ax.set_title(f'COC Semantic Clustering\n({n_samples} samples, '
                     f'{len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters)',
                     fontsize=14)
        ax.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=7, ncol=2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化图已保存: {save_path}")
        plt.close()

    def build_cluster_map(self):
        """构建聚类结果，每个样本能映射回 clip_id"""
        cluster_label2sample_idx = {}
        for label in set(self.cluster_labels):
            cluster_label2sample_idx[int(label)] = []

        for i, label in enumerate(self.cluster_labels):
            cluster_label2sample_idx[int(label)].append(i)

        # 构建带完整映射信息的聚类结果
        cluster_results = {}
        for label, indices in cluster_label2sample_idx.items():
            cluster_results[label] = [
                {
                    'sample_idx': idx,
                    'clip_id': self.clip_ids[idx],
                    'frame_index': self.frame_indices[idx],
                    'coc_text': self.samples[idx],
                }
                for idx in indices
            ]

        return cluster_results

    def analyze_and_print(self, cluster_results):
        """打印聚类摘要"""
        print("\n" + "=" * 60)
        print("聚类分析结果")
        print("=" * 60)

        for label in sorted(cluster_results.keys()):
            samples = cluster_results[label]
            if label == -1:
                print(f"\n噪声点 ({len(samples)}个):")
            else:
                # 计算簇内平均相似度
                indices = [s['sample_idx'] for s in samples]
                emb = self.embeddings[indices]
                centroid = emb.mean(axis=0)
                sims = [np.dot(centroid, e) / (np.linalg.norm(centroid) * np.linalg.norm(e)) for e in emb]
                avg_sim = np.mean(sims)

                # 找代表样本
                rep_idx = indices[np.argmax(sims)]
                rep_text = self.samples[rep_idx][:80]

                print(f"\n簇 {label} ({len(samples)}个样本, 平均相似度: {avg_sim:.3f})")
                print(f"  代表: {rep_text}")

                # 统计涉及的 clip 数量
                clip_count = len(set(s['clip_id'] for s in samples))
                print(f"  涉及 {clip_count} 个 clip")

                # 显示前几个样本
                for s in samples[:3]:
                    print(f"    - [{s['clip_id'][:8]}...] frame={s['frame_index']}: {s['coc_text'][:70]}...")
                if len(samples) > 3:
                    print(f"    ... 还有 {len(samples) - 3} 个样本")

        return cluster_results


def main():
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")

    # Default paths
    default_data_dir = "/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles"
    data_dir = os.getenv("ALPAMAYO_DATA_DIR", default_data_dir)

    parser = argparse.ArgumentParser(description="Cluster COC sentences using sentence embeddings")
    parser.add_argument("--input-file", type=str,
                        default=os.path.join(data_dir, "labels", "coc_train", "coc_train_change.json"),
                        help="Input JSON file path")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(data_dir, "labels", "coc_train"),
                        help="Output directory for cluster results")
    parser.add_argument("--model", type=str, default="ckpts/all-mpnet-base-v2",
                        help="Sentence transformer model name")
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="UMAP min_dist parameter")
    parser.add_argument("--n-components", type=int, default=5,
                        help="UMAP n_components parameter")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip visualization")

    args = parser.parse_args()

    print(f"Data directory: {data_dir}")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize clusterer
    coc = CocCluster(model_name=args.model)

    # 1. Load data
    coc.prepare_data(args.input_file)

    # 2. Encode
    coc.encode()

    # 3. Reduce dimensionality
    coc.reduce_dimensionality(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=min(args.n_components, len(coc.samples) - 1)
    )

    # 4. Cluster
    coc.cluster()

    # 5. Build cluster map
    cluster_results = coc.build_cluster_map()

    # 6. Print analysis
    coc.analyze_and_print(cluster_results)

    # 7. Visualize (optional)
    if not args.no_visualize:
        vis_path = os.path.join(args.output_dir, "coc_cluster_vis.png")
        coc.visualize_clusters(vis_path)

    # 8. Save results
    output_path = os.path.join(args.output_dir, "coc_cluster_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_results, f, indent=2, ensure_ascii=False)
    print(f"\n聚类结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
