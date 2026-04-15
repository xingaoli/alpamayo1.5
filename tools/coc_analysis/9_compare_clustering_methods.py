#!/usr/bin/env python3
"""
对比不同聚类方法的脚本
测试4种方法：
1. 默认UMAP + HDBSCAN (原始方法)
2. 调整UMAP参数 + HDBSCAN (推荐)
3. PCA + K-Means (更紧凑)
4. 直接HDBSCAN (不降维)

Usage:
    python tools/compare_clustering_methods.py --mode action
    python tools/compare_clustering_methods.py --mode factor
    python tools/compare_clustering_methods.py --mode both --min-cluster-size 10
"""

import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import warnings
import argparse
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'].insert(0, "AR PL UMing CN")
plt.rcParams['axes.unicode_minus'] = False


class ClusteringComparator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.samples = []
        self.metadata = []
        self.embeddings = None
        self.results = {}

    def load_data(self, data_path, mode='action'):
        """加载数据"""
        print(f"正在加载数据 (mode={mode})...")
        raw_data = json.load(open(data_path, 'r', encoding='utf-8'))

        self.samples = []
        self.metadata = []

        for entry in raw_data:
            clip_id = entry.get('clip_id', 'unknown')

            if mode == 'both':
                # 处理 actions
                action_lists = entry.get('action_lists', {})
                for frame_idx, actions in action_lists.items():
                    if isinstance(actions, list):
                        for action_text in actions:
                            if action_text.strip():
                                self.samples.append(action_text.strip())
                                self.metadata.append({
                                    'clip_id': clip_id,
                                    'frame_index': frame_idx,
                                    'source': 'action'
                                })

                # 处理 factors
                factor_lists = entry.get('factor_lists', {})
                for frame_idx, factors in factor_lists.items():
                    if isinstance(factors, list):
                        for factor_text in factors:
                            if factor_text.strip():
                                self.samples.append(factor_text.strip())
                                self.metadata.append({
                                    'clip_id': clip_id,
                                    'frame_index': frame_idx,
                                    'source': 'factor'
                                })
            else:
                list_key = f'{mode}_lists'
                target_lists = entry.get(list_key, {})
                for frame_idx, items in target_lists.items():
                    if isinstance(items, list):
                        for text in items:
                            if text.strip():
                                self.samples.append(text.strip())
                                self.metadata.append({
                                    'clip_id': clip_id,
                                    'frame_index': frame_idx,
                                    'source': mode
                                })

        print(f"总样本数: {len(self.samples)}")
        unique_clips = len(set(m['clip_id'] for m in self.metadata))
        print(f"来自 {unique_clips} 个 clip")

    def encode(self):
        """生成语义向量"""
        print("正在生成语义向量...")
        self.embeddings = self.model.encode(
            self.samples, show_progress_bar=True, convert_to_tensor=False
        )
        return self.embeddings

    def method1_default_umap_hdbscan(self, min_cluster_size=5):
        """方法1: 默认UMAP + HDBSCAN (原始方法)"""
        print("\n" + "="*80)
        print("方法1: 默认UMAP + HDBSCAN (原始方法)")
        print("="*80)

        n_samples = len(self.samples)
        n_neighbors = min(15, n_samples - 1)
        n_components = min(5, n_samples - 1)
        min_cluster_size = max(min_cluster_size, n_samples // 200)
        min_samples = max(2, min_cluster_size // 3)

        # UMAP降维
        print(f"UMAP参数: n_neighbors={n_neighbors}, min_dist=0.1, n_components={n_components}")
        umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_components=n_components,
            metric='cosine',
            random_state=42
        )
        umap_embeddings = umap_reducer.fit_transform(self.embeddings)

        # HDBSCAN聚类
        print(f"HDBSCAN参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',  # 注意：这里和UMAP的cosine不一致
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(umap_embeddings)

        self.results['method1_default'] = {
            'name': '默认UMAP+HDBSCAN',
            'labels': labels,
            'embeddings_2d': umap_embeddings,
            'method': 'umap_hdbscan'
        }

        return labels

    def method2_tuned_umap_hdbscan(self, min_cluster_size=5):
        """方法2: 调整UMAP参数 + HDBSCAN (推荐)"""
        print("\n" + "="*80)
        print("方法2: 调整UMAP参数 + HDBSCAN (推荐)")
        print("="*80)

        n_samples = len(self.samples)
        # 调整参数：增大n_neighbors，减小min_dist
        n_neighbors = min(50, n_samples - 1)
        n_components = min(5, n_samples - 1)
        min_cluster_size = max(min_cluster_size, n_samples // 200)
        min_samples = max(3, min_cluster_size // 2)

        print(f"UMAP参数: n_neighbors={n_neighbors}, min_dist=0.05, n_components={n_components}")
        umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.05,  # 减小min_dist，让相似点更紧密
            n_components=n_components,
            metric='cosine',
            random_state=42
        )
        umap_embeddings = umap_reducer.fit_transform(self.embeddings)

        # HDBSCAN聚类：统一使用cosine度量
        print(f"HDBSCAN参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='leaf'  # 使用leaf方法，倾向于更多小簇
        )
        labels = clusterer.fit_predict(umap_embeddings)

        # 保留噪声点，不强制分配（噪声点可能是稀有样本）
        n_noise = np.sum(labels == -1)
        if n_noise > 0:
            print(f"噪声点数量: {n_noise} ({n_noise/len(labels)*100:.1f}%) - 保留为噪声")

        self.results['method2_tuned'] = {
            'name': '调优UMAP+HDBSCAN',
            'labels': labels,
            'embeddings_2d': umap_embeddings,
            'method': 'tuned_umap_hdbscan'
        }

        return labels

    def method3_pca_kmeans(self, min_cluster_size=5):
        """方法3: PCA + K-Means (更紧凑的聚类)"""
        print("\n" + "="*80)
        print("方法3: PCA + K-Means (更紧凑)")
        print("="*80)

        n_samples = len(self.samples)
        n_components = min(50, n_samples - 1)

        # PCA降维，保留主要方差
        print(f"PCA参数: n_components={n_components}")
        pca = PCA(n_components=n_components, random_state=42)
        pca_embeddings = pca.fit_transform(self.embeddings)
        explained_var = sum(pca.explained_variance_ratio_)
        print(f"保留的方差比例: {explained_var:.3f}")

        # 使用肘部法则估计K值
        k_min = max(3, min_cluster_size)
        k_max = min(50, n_samples // min_cluster_size)
        inertias = []
        K_range = range(k_min, k_max + 1, 5)
        
        print(f"正在搜索最优K值 (范围: {k_min}-{k_max})...")
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pca_embeddings)
            inertias.append(kmeans.inertia_)
        
        # 找到肘部（二阶导数最大点）
        if len(inertias) > 2:
            second_derivative = np.diff(inertias, 2)
            optimal_k_idx = np.argmax(second_derivative) + 1
            optimal_k = list(K_range)[optimal_k_idx]
        else:
            optimal_k = k_min

        print(f"最优K值: {optimal_k}")

        # 使用最优K进行聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pca_embeddings)

        # 为了可视化，使用PCA降到2D
        pca_2d = PCA(n_components=2, random_state=42)
        embeddings_2d = pca_2d.fit_transform(pca_embeddings)

        self.results['method3_pca_kmeans'] = {
            'name': 'PCA+K-Means',
            'labels': labels,
            'embeddings_2d': embeddings_2d,
            'method': 'pca_kmeans',
            'k': optimal_k
        }

        return labels

    def method4_direct_hdbscan(self, min_cluster_size=5):
        """方法4: 直接HDBSCAN (不降维)"""
        print("\n" + "="*80)
        print("方法4: 直接HDBSCAN (不降维)")
        print("="*80)

        n_samples = len(self.samples)
        min_cluster_size = max(min_cluster_size, n_samples // 200)
        min_samples = max(2, min_cluster_size // 3)

        print(f"HDBSCAN参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        # 使用预计算的余弦距离矩阵
        from sklearn.metrics.pairwise import cosine_distances
        # 只采样部分样本计算距离矩阵，避免内存爆炸
        if n_samples > 5000:
            print(f"样本数较多 ({n_samples})，使用随机采样5000个样本进行近似聚类")
            indices = np.random.RandomState(42).choice(n_samples, 5000, replace=False)
            dist_matrix = cosine_distances(self.embeddings[indices]).astype(np.float64)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='precomputed',
                cluster_selection_method='eom'
            )
            labels_partial = clusterer.fit_predict(dist_matrix)
            # 对其他样本使用最近邻分配
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, metric='cosine')
            nn.fit(self.embeddings[indices])
            _, nearest_idx = nn.kneighbors(self.embeddings)
            labels = np.full(n_samples, -1, dtype=int)
            labels[indices] = labels_partial
            for i in range(n_samples):
                if i not in set(indices):
                    labels[i] = labels_partial[nearest_idx[i][0]]
        else:
            dist_matrix = cosine_distances(self.embeddings).astype(np.float64)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='precomputed',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(dist_matrix)

        # 为了可视化，使用PCA降到2D
        pca_2d = PCA(n_components=2, random_state=42)
        embeddings_2d = pca_2d.fit_transform(self.embeddings)

        self.results['method4_direct_hdbscan'] = {
            'name': '直接HDBSCAN',
            'labels': labels,
            'embeddings_2d': embeddings_2d,
            'method': 'direct_hdbscan'
        }

        return labels

    def evaluate_methods(self):
        """评估所有方法"""
        print("\n" + "="*80)
        print("聚类方法评估对比")
        print("="*80)

        evaluation = {}

        for method_key, result in self.results.items():
            labels = result['labels']
            name = result['name']

            print(f"\n{name}:")
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1)
            noise_ratio = n_noise / len(labels) * 100

            stats = {
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': noise_ratio,
                'n_samples': len(labels)
            }

            # 计算评估指标（只对非噪声点和至少有2个簇的情况）
            if n_clusters >= 2 and n_noise < len(labels):
                valid_mask = labels != -1
                valid_embeddings = self.embeddings[valid_mask]
                valid_labels = labels[valid_mask]

                if len(set(valid_labels)) >= 2:
                    silhouette = silhouette_score(valid_embeddings, valid_labels, metric='cosine')
                    calinski = calinski_harabasz_score(valid_embeddings, valid_labels)
                    stats['silhouette_score'] = silhouette
                    stats['calinski_harabasz'] = calinski
                    print(f"  簇数量: {n_clusters}")
                    print(f"  噪声点: {n_noise} ({noise_ratio:.1f}%)")
                    print(f"  轮廓系数: {silhouette:.3f} (越高越好，范围[-1,1])")
                    print(f"  Calinski-Harabasz: {calinski:.1f} (越高越好)")

                    # 计算簇内紧密度
                    cluster_densities = []
                    for lbl in unique_labels:
                        if lbl == -1:
                            continue
                        mask = labels == lbl
                        cluster_embs = self.embeddings[mask]
                        centroid = cluster_embs.mean(axis=0)
                        avg_sim = np.mean([
                            np.dot(centroid, e) / (np.linalg.norm(centroid) * np.linalg.norm(e) + 1e-8)
                            for e in cluster_embs
                        ])
                        cluster_densities.append(avg_sim)

                    avg_density = np.mean(cluster_densities)
                    stats['avg_cluster_density'] = avg_density
                    print(f"  平均簇内紧密度: {avg_density:.3f} (越高越紧凑)")
                else:
                    print(f"  簇数量: {n_clusters} (不足以计算指标)")
            else:
                print(f"  簇数量: {n_clusters}")
                print(f"  噪声点: {n_noise} ({noise_ratio:.1f}%)")

            evaluation[method_key] = stats

            # 打印簇大小分布
            if n_clusters > 0:
                cluster_sizes = []
                for lbl in unique_labels:
                    if lbl != -1:
                        cluster_sizes.append(np.sum(labels == lbl))
                cluster_sizes.sort(reverse=True)
                print(f"  簇大小: Top5={cluster_sizes[:5]}, Min={cluster_sizes[-1] if cluster_sizes else 0}, "
                      f"Median={np.median(cluster_sizes):.0f}")

        self.evaluation = evaluation
        return evaluation

    def visualize_comparison(self, save_path, mode='action'):
        """可视化对比所有方法"""
        print(f"\n正在生成可视化对比图...")

        n_methods = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        for idx, (method_key, result) in enumerate(self.results.items()):
            ax = axes[idx]
            labels = result['labels']
            embeddings_2d = result['embeddings_2d']
            name = result['name']

            unique_labels = set(labels)
            if len(unique_labels) > 1:
                # 使用更大的perplexity进行t-SNE可视化
                n_samples = len(labels)
                perplexity = min(30, max(5, n_samples // 3))

                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                                max_iter=1000, learning_rate='auto')
                    vis_embeddings = tsne.fit_transform(embeddings_2d)
                except:
                    pca = PCA(n_components=2, random_state=42)
                    vis_embeddings = pca.fit_transform(embeddings_2d)
            else:
                vis_embeddings = embeddings_2d[:, :2]

            # 绘制聚类
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            colors = plt.cm.Spectral(np.linspace(0, 1, max(len(unique_labels), 2)))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                if label == -1:
                    color = 'gray'
                    label_name = f'Noise'
                    marker, alpha, s = 'x', 0.3, 20
                else:
                    color = colors[i]
                    label_name = f'C{label}'
                    marker, alpha, s = 'o', 0.6, 30

                ax.scatter(vis_embeddings[mask, 0], vis_embeddings[mask, 1],
                          c=[color], label=label_name, alpha=alpha, s=s,
                          marker=marker, edgecolors='none')

            stats = self.evaluation.get(method_key, {})
            title = f"{name}\n"
            title += f"簇={n_clusters}, 噪声={stats.get('n_noise', 0)}"
            if 'silhouette_score' in stats:
                title += f", 轮廓={stats['silhouette_score']:.3f}"

            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(f'{mode.capitalize()} 聚类方法对比', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比可视化已保存: {save_path}")
        plt.close()

    def print_recommendation(self):
        """打印推荐方案"""
        print("\n" + "="*80)
        print("推荐方案")
        print("="*80)

        if not hasattr(self, 'evaluation'):
            print("请先执行评估")
            return

        # 根据不同标准推荐
        best_silhouette = None
        best_density = None
        least_noise = None
        most_clusters = None

        for method_key, stats in self.evaluation.items():
            name = self.results[method_key]['name']

            if 'silhouette_score' in stats:
                if best_silhouette is None or stats['silhouette_score'] > best_silhouette[1]:
                    best_silhouette = (name, stats['silhouette_score'])

            if 'avg_cluster_density' in stats:
                if best_density is None or stats['avg_cluster_density'] > best_density[1]:
                    best_density = (name, stats['avg_cluster_density'])

            if least_noise is None or stats['noise_ratio'] < least_noise[1]:
                least_noise = (name, stats['noise_ratio'])

            if most_clusters is None or stats['n_clusters'] > most_clusters[1]:
                most_clusters = (name, stats['n_clusters'])

        print(f"轮廓系数最高 (聚类质量): {best_silhouette[0]} ({best_silhouette[1]:.3f})")
        print(f"簇内最紧密 (语义一致性): {best_density[0]} ({best_density[1]:.3f})")
        print(f"噪声点最少 (鲁棒性): {least_noise[0]} ({least_noise[1]:.1f}%)")
        print(f"簇数量最多 (细粒度): {most_clusters[0]} ({most_clusters[1]}个)")

        print("\n💡 建议:")
        print("  - 如果需要紧凑的聚类（相近语义聚集）：选择 PCA+K-Means")
        print("  - 如果需要自动发现聚类数量：选择 调优UMAP+HDBSCAN")
        print("  - 如果想保留原始高维信息：选择 直接HDBSCAN")


def main():
    # Load environment variables
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)

    # Default paths
    default_data_dir = "/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles"
    data_dir = os.getenv("ALPAMAYO_DATA_DIR", default_data_dir)

    parser = argparse.ArgumentParser(description="对比不同聚类方法")
    parser.add_argument("--input-file", type=str,
                        default=os.path.join(data_dir, "labels", "coc_train", "coc_train_action_factor.json"),
                        help="输入JSON文件路径")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(data_dir, "labels", "coc_train"),
                        help="输出目录")
    parser.add_argument("--model", type=str, default="ckpts/all-mpnet-base-v2",
                        help="Sentence transformer模型")
    parser.add_argument("--mode", type=str, default="action",
                        choices=["action", "factor", "both"],
                        help="聚类模式")
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="最小簇大小")

    args = parser.parse_args()

    print(f"数据目录: {data_dir}")
    print(f"输入文件: {args.input_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"模式: {args.mode}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化对比器
    comparator = ClusteringComparator(model_name=args.model)

    # 1. 加载数据
    comparator.load_data(args.input_file, mode=args.mode)

    if len(comparator.samples) == 0:
        print("错误: 没有找到可聚类的样本!")
        return

    # 2. 编码
    comparator.encode()

    # 3. 运行4种方法
    comparator.method1_default_umap_hdbscan(min_cluster_size=args.min_cluster_size)
    comparator.method2_tuned_umap_hdbscan(min_cluster_size=args.min_cluster_size)
    comparator.method3_pca_kmeans(min_cluster_size=args.min_cluster_size)
    comparator.method4_direct_hdbscan(min_cluster_size=args.min_cluster_size)

    # 4. 评估
    comparator.evaluate_methods()

    # 5. 可视化对比
    vis_path = os.path.join(args.output_dir, f"coc_{args.mode}_clustering_comparison.png")
    comparator.visualize_comparison(vis_path, mode=args.mode)

    # 6. 打印推荐
    comparator.print_recommendation()

    # 7. 保存所有结果
    output_file = os.path.join(args.output_dir, f"coc_{args.mode}_clustering_comparison.json")
    comparison_results = {
        method_key: {
            'name': result['name'],
            'labels': [int(x) for x in result['labels'].tolist()],
            'evaluation': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                          for k, v in comparator.evaluation.get(method_key, {}).items()}
        }
        for method_key, result in comparator.results.items()
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    print(f"\n对比结果已保存到: {output_file}")


if __name__ == "__main__":
    import os
    # 设置使用GPU2
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()
