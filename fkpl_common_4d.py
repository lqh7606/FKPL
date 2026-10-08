"""fkpl_4domain 公共模块。

被 diagnose_beta.py（阶段 1/2：SVD 主角度诊断 + beta 重标定）与
fkpl_domainnet4d.py（正式训练）共同使用。

由 fkpl_3domain/fkpl_common_3d.py 派生：数据/子空间/聚类/beta 标定逻辑逐段沿用，
只把默认域集合改为「4 个域全用」，并把客户端布局的默认档位扩到 4/3/2/1。

抽成公共模块的目的：
1. 两个脚本走同一段数据划分代码，np.random 的调用序列完全一致，
   因此诊断阶段产出的 svd_preprocess_stats.csv 可以与正式运行的
   svd_preprocess_stats.csv 做 MD5 比对（沿用工程内既有的锁定验收方法）；
2. 不再使用任何「按客户端索引硬编码的簇」，簇的生成方式只有两种：
   - build_fixed_domain_clusters()：按「域」动态生成域纯簇（默认）；
   - hierarchical_clustering()：用重标定的 beta 在线聚类（--use_hc_for_training）。

与既有 paper-aligned 脚本的差异说明：
- 数据根为 domainnet_subsetA（<domain>/<class>/*.jpg 的扁平结构，与 PACS /
  Office-Caltech 相同），因此沿用 i % 10 <= 7 的确定性 8:2 切分；
- subsetA 的 10 个类别在 4 个域中目录名完全一致（构建子集时已要求每域
  >= 150 张），逐域 ImageFolder 的字母序编号天然对齐；这里仍然显式构建
  全局 class_to_idx 并校验，避免以后换数据时静默错位；
- 自适应基向量分配（traindata_cls_ratio -> K）保持 paper-aligned fixed 版
  的语义（label_key 为 int，字典 .get 生效），即每客户端基向量总数受
  --budget 约束按类样本占比分配，不再复现 MINI_FKPL_v2 中因类型不匹配
  而失效的死代码。
"""
import os
import csv
import copy
import time
from collections import Counter

import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler

from hierarchical_clustering import calculating_adjacency, hierarchical_clustering, round_to


# ==================== 数据集常量 ====================
ALL_DOMAINS = ['clipart', 'painting', 'real', 'sketch']
DEFAULT_TRAIN_DOMAINS = list(ALL_DOMAINS)  # 4 域全用，无留出国
DEFAULT_DATA_ROOT = './data/domainnet_subsetA/'
SUBSET_TRAIN_NUM = 7
SUBSET_CAPACITY = 10
MEAN_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET = (0.229, 0.224, 0.225)

SVD_PREPROCESS_FIELDNAMES = [
    'client_id', 'domain', 'num_samples', 'num_labels', 'u_cols_total', 'u_rows',
    'u_bytes', 'u_megabytes', 'svd_time_sec', 'client_total_preprocess_time_sec',
]


# ==================== 基础工具 ====================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


class CsvLogger:
    def __init__(self, out_csv: str, fieldnames):
        self.out_csv = out_csv
        ensure_dir(os.path.dirname(out_csv))
        self.fieldnames = list(fieldnames)
        self._initialized = False

    def write_row(self, row: dict):
        mode = 'a' if self._initialized and os.path.exists(self.out_csv) else 'w'
        with open(self.out_csv, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._initialized or mode == 'w':
                writer.writeheader()
            self._initialized = True
            writer.writerow(row)


def write_single_row_csv(out_csv: str, row: dict):
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def tensor_nbytes(x: torch.Tensor) -> int:
    return int(x.numel() * x.element_size())


def numpy_nbytes(x: np.ndarray) -> int:
    return int(x.nbytes)


def bytes_to_mb(x: float) -> float:
    return float(x) / (1024.0 * 1024.0)


# ==================== 数据加载 ====================
def list_domain_classes(data_root: str, domains):
    """返回 {domain: sorted(class_names)}，只统计含图片的类别目录。"""
    exts = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    out = {}
    for domain in domains:
        domain_dir = os.path.join(data_root, domain)
        classes = []
        for cls_name in sorted(os.listdir(domain_dir)):
            cls_dir = os.path.join(domain_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            n = sum(1 for f in os.listdir(cls_dir) if f.lower().endswith(exts))
            if n > 0:
                classes.append(cls_name)
        out[domain] = classes
    return out


def build_global_class_to_idx(data_root: str, domains):
    """跨域统一标签空间：类别名取并集后按字母序编号。"""
    class_sets = list_domain_classes(data_root, domains)
    union = set()
    for cls_list in class_sets.values():
        union.update(cls_list)
    classes = sorted(union)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for domain in domains:
        if class_sets[domain] != classes:
            missing = sorted(union - set(class_sets[domain]))
            extra = sorted(set(class_sets[domain]) - union)
            print(f'[WARN] domain {domain} label space differs from global union: '
                  f'missing={missing} extra={extra}; samples of unmatched classes will be dropped.')
    return class_to_idx, class_sets


class ImageFolderCustom(torch.utils.data.Dataset):
    """<root>/<domain>/<class>/*.jpg，按 i % subset_capacity <= subset_train_num 切 8:2。

    标签用全局 class_to_idx 重映射（不用逐域 ImageFolder 自带的编号）。
    """

    def __init__(self, data_name, root, train, transform, class_to_idx,
                 subset_train_num=SUBSET_TRAIN_NUM, subset_capacity=SUBSET_CAPACITY):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        imagefolder_obj = datasets.ImageFolder(root=os.path.join(root, data_name), transform=transform)

        samples = []
        dropped = 0
        for path, cls_id in imagefolder_obj.samples:
            cls_name = imagefolder_obj.classes[cls_id]
            if cls_name in class_to_idx:
                samples.append((path, class_to_idx[cls_name]))
            else:
                dropped += 1
        if dropped:
            print(f'[{data_name}] dropped {dropped} samples outside the global label space.')
        self.samples = samples
        self.loader = imagefolder_obj.loader

        self.train_index_list = []
        self.test_index_list = []
        for i in range(len(self.samples)):
            if i % subset_capacity <= subset_train_num:
                self.train_index_list.append(i)
            else:
                self.test_index_list.append(i)

    def __len__(self):
        return len(self.train_index_list) if self.train else len(self.test_index_list)

    def __getitem__(self, index):
        used = self.train_index_list if self.train else self.test_index_list
        path, target = self.samples[used[index]]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def labels_of_train_pool(self):
        return np.array([int(self.samples[i][1]) for i in self.train_index_list], dtype=np.int64)


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET),
    ])


def record_net_data_stats(y_train, net_dataidx):
    unq, unq_cnt = np.unique(np.asarray(y_train)[np.asarray(net_dataidx)], return_counts=True)
    return {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}


def partition_domain_skew_loaders(train_datasets, test_datasets, percent_dict, batch_size=64,
                                  test_batch_size=256):
    """与既有 FKPL / Comparied 脚本一致的域偏斜划分：同域多客户端依次从该域
    未使用索引池中取 int(percent * 域训练池大小) 个样本（同域 percent 之和 <= 1）。
    """
    init_len_dict, not_used_index_dict = {}, {}
    net_dataidx_map, traindata_cls_counts = {}, {}
    train_loaders, test_loaders = [], []

    for ds in train_datasets:
        d = ds.data_name
        if d not in not_used_index_dict:
            not_used_index_dict[d] = np.arange(len(ds.train_index_list))
            init_len_dict[d] = len(ds.train_index_list)

    for ds in test_datasets:
        test_loaders.append(DataLoader(ds, batch_size=test_batch_size, shuffle=False))

    for i, ds in enumerate(train_datasets):
        d = ds.data_name
        percent = percent_dict[d]
        idxs = np.random.permutation(not_used_index_dict[d])
        cut = int(percent * init_len_dict[d])
        selected_idxs = idxs[0:cut]
        not_used_index_dict[d] = idxs[cut:]

        train_loaders.append(DataLoader(ds, batch_size=batch_size, sampler=SubsetRandomSampler(selected_idxs)))
        traindata_cls_counts[i] = record_net_data_stats(ds.labels_of_train_pool(), selected_idxs)
        net_dataidx_map[i] = selected_idxs

    return traindata_cls_counts, net_dataidx_map, train_loaders, test_loaders


# ==================== 子空间构建 ====================
def compute_basis_budget(traindata_cls_counts, num_clients, budget=10):
    """按类样本占比把每客户端的基向量总预算 budget 分配到各类。"""
    traindata_cls_ratio = {}
    for i in range(num_clients):
        counts = traindata_cls_counts[i]
        total_sum = sum(counts.values())
        base = 1 / len(counts)
        temp_ratio = {}
        for k, v in counts.items():
            ratio = v / total_sum
            temp_ratio[k] = v if ratio >= base else ratio
        sub_sum = sum(temp_ratio.values())
        for k in list(temp_ratio.keys()):
            temp_ratio[k] = (temp_ratio[k] / sub_sum) * budget
        rounded = round_to(list(temp_ratio.values()), budget)
        for idx, k in enumerate(list(temp_ratio.keys())):
            temp_ratio[k] = rounded[idx]
        traindata_cls_ratio[i] = temp_ratio
    return traindata_cls_ratio


def build_U_clients(train_dataset_list, net_dataidx_map, traindata_cls_ratio, selected_domain_list,
                    num_clients, n_basis=5, budget_mode='budget', verbose=True):
    """逐客户端按类做像素矩阵 SVD，取前 K 个左奇异向量拼成子空间基。

    返回 (U_clients, stats_rows)。K 的取法与 paper-aligned fixed 版一致：
    budget 模式下 K 来自 traindata_cls_ratio（每客户端总数 ~= --budget）；
    original 模式复现 MINI_FKPL_v2 中恒为 n_basis 的行为，仅用于对照。
    """
    U_clients, stats_rows = [], []
    for idx in range(num_clients):
        client_start = time.perf_counter()
        dataidxs = net_dataidx_map[idx]
        train_data = train_dataset_list[idx]

        sample_list, label_list = [], []
        for dataidx in dataidxs:
            x, y = train_data[int(dataidx)]
            sample_list.append(x)
            label_list.append(int(y))

        train_data_np = torch.stack(sample_list, dim=0).numpy().astype(np.float32)
        labels = np.array(label_list)
        idxs_local = np.arange(len(sample_list))
        idxs_labels_local = np.vstack((idxs_local, labels))
        idxs_labels_local = idxs_labels_local[:, idxs_labels_local[1, :].argsort()]
        idxs_local = idxs_labels_local[0, :]
        labels_local = idxs_labels_local[1, :]
        uni_labels, cnt_labels = np.unique(labels_local, return_counts=True)

        cnt, U_temp, svd_time_acc, total_cols, u_rows = 0, [], 0.0, 0, 0
        for j in range(len(uni_labels)):
            local_ds = train_data_np[idxs_local[cnt:cnt + cnt_labels[j]]]
            local_ds = local_ds.reshape(cnt_labels[j], -1).T
            u_rows = local_ds.shape[0]

            if budget_mode == 'budget':
                label_key = int(uni_labels[j])
                K = int(traindata_cls_ratio[idx].get(label_key, n_basis))
            else:
                K = int(n_basis)

            if K > 0:
                svd_start = time.perf_counter()
                u1_temp, _, _ = np.linalg.svd(local_ds, full_matrices=False)
                svd_time_acc += time.perf_counter() - svd_start
                u1_temp = u1_temp / np.linalg.norm(u1_temp, ord=2, axis=0)
                u1_temp = u1_temp[:, 0:K].astype(np.float32)
                U_temp.append(u1_temp)
                total_cols += int(u1_temp.shape[1])
            cnt += int(cnt_labels[j])

        U_client = np.hstack(U_temp).astype(np.float32)
        U_clients.append(copy.deepcopy(U_client))
        u_bytes = numpy_nbytes(U_client)
        stats_rows.append({
            'client_id': idx,
            'domain': str(selected_domain_list[idx]),
            'num_samples': len(dataidxs),
            'num_labels': int(len(uni_labels)),
            'u_cols_total': int(total_cols),
            'u_rows': int(u_rows),
            'u_bytes': int(u_bytes),
            'u_megabytes': round(bytes_to_mb(u_bytes), 6),
            'svd_time_sec': round(svd_time_acc, 6),
            'client_total_preprocess_time_sec': round(time.perf_counter() - client_start, 6),
        })
        if verbose:
            print(f'Client {idx:02d} | domain={str(selected_domain_list[idx]):>10s} | '
                  f'n={len(dataidxs):4d} | labels={int(len(uni_labels))} | U={U_client.shape}')
    return U_clients, stats_rows


def compute_adjacency(U_clients):
    clients_idxs = np.arange(len(U_clients))
    return calculating_adjacency(clients_idxs, U_clients)


# ==================== 簇结构 ====================
def build_fixed_domain_clusters(selected_domain_list, domain_order=None):
    """按「域」动态生成域纯簇（不依赖任何客户端索引硬编码）。"""
    by_domain = {}
    for client_id, domain_name in enumerate(selected_domain_list):
        by_domain.setdefault(str(domain_name), []).append(int(client_id))
    order = list(domain_order) if domain_order is not None else sorted(by_domain.keys())
    return [[int(c) for c in by_domain[d]] for d in order if by_domain.get(d)]


def cluster_domain_report(clusters, selected_domain_list):
    """每个簇的域组成与（加权）域纯度。"""
    detail, correct, total = [], 0, 0
    for cid, cluster in enumerate(clusters):
        domains = [str(selected_domain_list[i]) for i in cluster]
        counter = Counter(domains)
        correct += max(counter.values())
        total += len(cluster)
        detail.append({
            'cluster_id': cid,
            'clients': [int(c) for c in cluster],
            'domains': sorted(counter.items(), key=lambda kv: kv[0]),
            'pure': len(counter) == 1,
        })
    purity = correct / total if total else 0.0
    return detail, purity


def distance_stats(adj_mat, clusters):
    intra, inter = [], []
    for cluster in clusters:
        for a in range(len(cluster)):
            for b in range(a + 1, len(cluster)):
                intra.append(float(adj_mat[cluster[a], cluster[b]]))
    for ca in range(len(clusters)):
        for cb in range(ca + 1, len(clusters)):
            for i in clusters[ca]:
                for j in clusters[cb]:
                    inter.append(float(adj_mat[i, j]))
    intra_mean = float(np.mean(intra)) if intra else float('nan')
    inter_mean = float(np.mean(inter)) if inter else float('nan')
    gap = inter_mean - intra_mean if not (np.isnan(intra_mean) or np.isnan(inter_mean)) else float('nan')
    return intra_mean, inter_mean, gap


def fmt_clusters(clusters):
    return '; '.join(['[' + ','.join(map(str, c)) + ']' for c in clusters])


# ==================== 域间几何与客户端布局 ====================
def domain_pair_distances(adj_mat, selected_domain_list):
    """{（域a, 域b）: 平均主角度距离}，跨域客户端对取均值；同域对单独统计。"""
    domains = [str(d) for d in selected_domain_list]
    intra, inter = {}, {}
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            d = float(adj_mat[i, j])
            if domains[i] == domains[j]:
                intra.setdefault(domains[i], []).append(d)
            else:
                inter.setdefault(tuple(sorted((domains[i], domains[j]))), []).append(d)
    intra_mean = {k: float(np.mean(v)) for k, v in intra.items()}
    inter_mean = {f'{a}|{b}': float(np.mean(v)) for (a, b), v in inter.items()}
    return intra_mean, inter_mean


def domain_centrality(adj_mat, selected_domain_list, domains):
    """每个域到其它选中域的平均主角度距离；越小越「居中」。"""
    inter, _ = domain_pair_distances(adj_mat, selected_domain_list)
    centrality = {}
    for d in domains:
        vals = [v for k, v in inter.items() if d in k.split('|')]
        centrality[d] = float(np.mean(vals)) if vals else float('nan')
    return centrality


def resolve_client_layout(centrality, client_counts=(4, 3, 2, 1), percents=(0.25, 0.30, 0.50, 1.0)):
    """越居中的域分配越多客户端（每客户端份额越小），越孤立的域客户端越少（份额越大），
    复现 Office-Caltech 4/3/2/1 + 0.25/0.25/0.5/1.0 的域偏斜形态。
    """
    ordered = sorted(centrality.keys(), key=lambda d: (centrality[d], d))  # 居中 -> 孤立
    layout = {}
    for i, d in enumerate(ordered):
        k = min(i, len(client_counts) - 1)
        layout[d] = {'num_clients': int(client_counts[k]), 'percent': float(percents[k])}
    for d in ordered[len(client_counts):]:
        layout[d] = {'num_clients': int(client_counts[-1]), 'percent': float(percents[-1])}
    return ordered, layout


# ==================== beta 扫描 ====================
def scan_beta(adj_mat, selected_domain_list, betas, linkage='average'):
    rows = []
    for beta in betas:
        clusters = hierarchical_clustering(copy.deepcopy(adj_mat), thresh=float(beta), linkage=linkage)
        detail, purity = cluster_domain_report(clusters, selected_domain_list)
        intra_mean, inter_mean, gap = distance_stats(adj_mat, clusters)
        rows.append({
            'beta_degree': round(float(beta), 4),
            'linkage': linkage,
            'V': len(clusters),
            'cluster_sizes': str([len(c) for c in clusters]),
            'weighted_domain_purity': round(purity, 4),
            'all_clusters_pure': bool(all(d['pure'] for d in detail)),
            'mean_intra_angle': round(intra_mean, 4) if not np.isnan(intra_mean) else '',
            'mean_inter_angle': round(inter_mean, 4) if not np.isnan(inter_mean) else '',
            'inter_minus_intra': round(gap, 4) if not np.isnan(gap) else '',
            'clusters': fmt_clusters(clusters),
            'domain_composition': '; '.join(
                [f"C{d['cluster_id']}[" + ','.join(f'{k}:{v}' for k, v in d['domains']) + ']' for d in detail]),
        })
    return rows


def select_beta(rows, v_min=2, v_max=4, require_pure=True):
    """beta 选取准则（两级）。

    1. 先取 V in [v_min, v_max] 的全部候选；
    2. 若存在「簇全域纯」的候选，则只在它们中选（沿用 PACS/Office 的口径）；
       否则退化为在 V 窗口内按 (域纯度 desc, inter-intra gap desc) 选最优，
       并在 meta 里记 note，提醒该数据集上 HC 无法给出完全域纯的簇；
    3. 以选中的 beta 为心，向两侧扩展到 (V, purity) 相同的连续 beta 平台，
       取平台中点（四舍五入到 0.1），避免落在平台边缘导致簇结构不稳定。
    """
    cand = [r for r in rows if v_min <= int(r['V']) <= v_max]
    if not cand:
        return {'beta': None, 'plateau': [], 'V': None, 'purity': None,
                'all_clusters_pure': False, 'note': f'V in [{v_min},{v_max}] 区间内无候选 beta'}

    pure_cands = [r for r in cand if r['all_clusters_pure']]
    if require_pure and pure_cands:
        pool, note = pure_cands, '在 V 窗口内存在完全域纯的 beta，按域纯准则选取'
    else:
        pool = cand
        note = ('在 V 窗口内不存在完全域纯的 beta，退化为「纯度优先、gap 次之」（best-effort）'
                if require_pure else '未要求域纯，按纯度/gap 选取')

    def _gap(r):
        try:
            return float(r['inter_minus_intra'])
        except (TypeError, ValueError):
            return float('-inf')

    best = max(pool, key=lambda r: (float(r['weighted_domain_purity']), _gap(r)))
    key = (int(best['V']), float(best['weighted_domain_purity']))
    same = sorted(float(r['beta_degree']) for r in pool
                  if (int(r['V']), float(r['weighted_domain_purity'])) == key)
    beta_best = float(best['beta_degree'])
    left = [b for b in same if b <= beta_best]
    right = [b for b in same if b >= beta_best]
    plateau_left = _contiguous_left(left, beta_best)
    plateau_right = _contiguous_right(right, beta_best)
    plateau = sorted(set(plateau_left + plateau_right))
    chosen = round(float(np.mean([plateau[0], plateau[-1]])), 1)
    return {'beta': chosen, 'plateau': [plateau[0], plateau[-1]], 'V': key[0],
            'purity': key[1], 'all_clusters_pure': bool(best['all_clusters_pure']),
            'beta_at_max_purity': beta_best, 'note': note}


def _step_size(betas):
    betas = sorted(betas)
    diffs = [round(betas[i + 1] - betas[i], 6) for i in range(len(betas) - 1)]
    diffs = [d for d in diffs if d > 0]
    return min(diffs) if diffs else 0.0


def _contiguous_left(sorted_desc_candidates, center):
    vals = sorted(v for v in sorted_desc_candidates if v <= center)
    if not vals:
        return [center]
    step = _step_size(vals + [center])
    out = [center]
    for v in reversed(vals[:-1] if vals[-1] == center else vals):
        if abs(out[0] - v) <= step * 1.5:
            out.insert(0, v)
        else:
            break
    return out


def _contiguous_right(sorted_asc_candidates, center):
    vals = sorted(v for v in sorted_asc_candidates if v >= center)
    if not vals:
        return [center]
    step = _step_size(vals + [center])
    out = [center]
    for v in vals[1:]:
        if abs(out[-1] - v) <= step * 1.5:
            out.append(v)
        else:
            break
    return out


# ==================== 特征 / 原型工具 ====================
def agg_features(features):
    """{label: [tensor, ...]} -> {label: 类原型 tensor}。"""
    protos = {}
    for label, label_features in features.items():
        if len(label_features) > 1:
            proto = 0 * label_features[0].data
            for feature in label_features:
                proto += feature.data
            protos[label] = proto / len(label_features)
        else:
            protos[label] = label_features[0].data
    return protos


def proto_aggregation(selected, local_protos_dict):
    """把 selected 客户端的同类原型求平均（三域版按「簇内」聚合，与 v2 语义一致）。"""
    agg = dict()
    for idx in selected:
        local_protos = local_protos_dict[idx]
        for label in local_protos.keys():
            if label in agg:
                agg[label].append(local_protos[label].cpu().numpy())
            else:
                agg[label] = [local_protos[label].cpu().numpy()]
    for label, protos_list in agg.items():
        mean_proto = np.mean(protos_list, axis=0, keepdims=True)
        agg[label] = torch.tensor(mean_proto, dtype=torch.float32)
    return agg


def get_fusion_features(encoder, train_dl, device):
    local_class_fusion_features = {}
    status = encoder.training
    encoder.eval()
    for input_x, target in train_dl:
        input_x, target = input_x.to(device), target.to(device)
        with torch.no_grad():
            fusion_features = encoder(input_x)
            for i in range(len(target)):
                label = int(target[i].item())
                if label in local_class_fusion_features:
                    local_class_fusion_features[label].append(fusion_features[i, :].detach().cpu())
                else:
                    local_class_fusion_features[label] = [fusion_features[i, :].detach().cpu()]
    encoder.train(status)
    return local_class_fusion_features


def get_fusion_protos(encoder, train_dls, num_clients, device):
    total_fusion_protos = {}
    for net_id in range(num_clients):
        local_features = get_fusion_features(encoder, train_dls[net_id], device)
        total_fusion_protos[net_id] = agg_features(local_features)
    return total_fusion_protos
