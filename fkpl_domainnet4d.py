"""FKPL 四域版（DomainNet-subsetA，4 个训练域）主训练脚本。

由 fkpl_3domain/fkpl_domainnet3d.py 派生，整条 FKPL 流水线（SVD 子空间 -> 层次聚类 ->
簇内 FedAvg -> 簇 encoder 特征拼接 -> DPCR 原型平移分类器重训）逐段沿用，差异集中在：

1. 【训练域 = 4 个域全集】clipart / painting / real / sketch 全部参与训练与测试，
   无留出国；协议仍为「仅训练域内部测试」（协议 B 风格），输出 mean4 / std4。
   指标键名随域数动态生成（mean{K}/std{K}），K = len(train_domains)。
2. 【去掉按索引硬编码的簇】簇有两种来源（与三域版一致）：
   - 默认：build_fixed_domain_clusters()，按「域」动态生成 4 个域纯簇；
   - --use_hc_for_training：用重标定的 beta（读自 fkpl_diagnosis/beta_calibration.json，
     或 --cluster_alpha 显式指定）在线层次聚类。
3. 【域偏斜布局】客户端在各域的分配与每客户端采样比例默认读自 SVD 主角度诊断产物
   （fkpl_diagnosis/client_layout.json，4 域不等分配）；也可用 --client_counts /
   --percents 直接指定布局（例如对齐同机基线的 5/5/5/5 + 0.2）。

用法：
  python fkpl_domainnet4d.py --run_encoder_stage --data_seed 7
  python fkpl_domainnet4d.py --use_hc_for_training --data_seed 7 --seed 1
  python fkpl_domainnet4d.py --client_counts 5 5 5 5 --percents 0.2 0.2 0.2 0.2
  python fkpl_domainnet4d.py --encoder_round 1 --local_epochs_encoder 1 --retrain_epochs 5 --device cpu
"""
import os
import json
import time
import copy
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import resnet10Encoder, SimpleClassifier, CombineModel, CombineAllModel
import fkpl_common_4d as C

try:
    from tSNE import *  # 与既有工程保持一致，可选
except Exception:
    pass


# ==================== 配置解析 ====================
DEFAULT_LAYOUT = {
    'train_domains': ['clipart', 'painting', 'real', 'sketch'],
    'client_counts': {'clipart': 4, 'painting': 3, 'real': 2, 'sketch': 1},
    'percent_per_client': {'clipart': 0.25, 'painting': 0.30, 'real': 0.50, 'sketch': 1.0},
}
DEFAULT_CLUSTER_ALPHA = 3.0
DEFAULT_LINKAGE = 'average'


def load_json(path):
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def resolve_layout(args):
    """客户端布局：命令行显式布局 > 诊断产物 > 按域数的内置默认。"""
    payload = None if args.client_counts else load_json(args.layout_file)
    if args.client_counts:
        if len(args.client_counts) != len(args.percents):
            raise ValueError('--client_counts 与 --percents 长度必须一致')
        train_domains = list(args.train_domains) if args.train_domains else list(DEFAULT_LAYOUT['train_domains'])
        if len(args.client_counts) != len(train_domains):
            raise ValueError(f'--client_counts 长度 {len(args.client_counts)} 与训练域数 '
                             f'{len(train_domains)} 不一致')
        client_counts = {d: int(c) for d, c in zip(train_domains, args.client_counts)}
        percents = {d: float(p) for d, p in zip(train_domains, args.percents)}
        source = 'cli_layout'
    elif payload is None:
        train_domains = list(args.train_domains) if args.train_domains else list(DEFAULT_LAYOUT['train_domains'])
        if args.train_domains:
            # 无诊断产物时按域数镜像 Office-Caltech 的不等分配形态
            default_counts = [4, 3, 2, 1][:len(train_domains)]
            default_percents = [0.25, 0.30, 0.50, 1.0][:len(train_domains)]
            client_counts = {d: c for d, c in zip(train_domains, default_counts)}
            percents = {d: p for d, p in zip(train_domains, default_percents)}
        else:
            client_counts = dict(DEFAULT_LAYOUT['client_counts'])
            percents = dict(DEFAULT_LAYOUT['percent_per_client'])
        source = 'builtin_default'
    else:
        train_domains = list(payload.get('chosen_domains') or payload.get('train_domains'))
        client_counts = {d: int(v) for d, v in payload['client_counts'].items() if d in train_domains}
        percents = {d: float(v) for d, v in payload['percent_per_client'].items() if d in train_domains}
        source = f'layout_file:{args.layout_file}'
        if args.train_domains:
            train_domains = list(args.train_domains)
            client_counts = {d: v for d, v in client_counts.items() if d in train_domains}
            percents = {d: v for d, v in percents.items() if d in train_domains}

    for d in train_domains:
        if d not in client_counts or d not in percents:
            raise ValueError(f'域 {d} 缺少客户端数或采样比例配置：请检查 {args.layout_file} 或 --train_domains')
    num_clients = int(sum(client_counts.values()))
    if args.num_clients is not None and args.num_clients != num_clients:
        raise ValueError(f'--num_clients={args.num_clients} 与布局 {client_counts} 的总数 {num_clients} 不一致')
    for d in train_domains:
        total = client_counts[d] * percents[d]
        if total > 1.0 + 1e-9:
            raise ValueError(f'域 {d} 的 percent 之和 {total:.2f} > 1，最后的客户端将拿不到数据')

    # beta 是在 Stage B 落盘的那份「客户端->域序列」的邻接矩阵上标定的。
    # 关键：不能直接把产物的 selected_domain_list 拿来用！那样会少一次
    # np.random.permutation 的随机数消耗，下游每客户端取样的图片子集会变，
    # 进而使邻接矩阵与标定时不同（实测：簇纯度对上了但 V 从 4 变成 5）。
    # 正确做法是用与 Stage B 相同的展开顺序（centrality 序）去 permutation，
    # 这样既能逐位复现标定时的布局，又能保持随机流一致。
    layout_order = None
    raw_order = (payload or {}).get('domain_order_central_to_isolated')
    if raw_order and not args.train_domains and not args.client_counts:
        cand_order = [str(x) for x in raw_order]
        if sorted(cand_order) == sorted(train_domains):
            layout_order = cand_order
    layout_selected = None
    raw_selected = (payload or {}).get('selected_domain_list')
    if raw_selected and not args.train_domains and not args.client_counts:
        cand = [str(x) for x in raw_selected]
        cnt = Counter(cand)
        if len(cand) == num_clients and all(cnt.get(d, 0) == client_counts[d] for d in train_domains):
            layout_selected = cand
        else:
            print(f'[WARN] {args.layout_file} 的 selected_domain_list 与 client_counts 不自洽，'
                  f'退回按 data_seed 自行洗牌')

    return (train_domains, client_counts, percents, num_clients, source,
            layout_order, layout_selected)


def resolve_beta(args):
    payload = load_json(args.calibration_file)
    linkage = str(args.linkage or (payload or {}).get('linkage') or DEFAULT_LINKAGE)
    if args.cluster_alpha is not None:
        return float(args.cluster_alpha), linkage, 'cli'
    if payload is not None and payload.get('cluster_alpha') is not None:
        return float(payload['cluster_alpha']), linkage, f'calibration_file:{args.calibration_file}'
    return float(DEFAULT_CLUSTER_ALPHA), linkage, 'builtin_default'


# ==================== 阶段无关的通用逻辑 ====================
def build_selected_domain_list(train_domains, client_counts, order_by_domain=None):
    expanded = []
    order = order_by_domain or sorted(client_counts.keys(), key=lambda d: -client_counts[d])
    for d in order:
        if d in client_counts:
            expanded += [d] * client_counts[d]
    return expanded


def build_control_clusters(grouping, base_clusters, num_clients, seed):
    """同容量集成对照（回应审稿人「增益只是容量增加」的质疑）。

    下游一切都以 clusters 为键：num_K（即独立训练的提取器个数与参数预算）、哪些
    客户端训练哪个提取器、以及喂给 DPCR 的逐簇域原型。因此固定组大小就同时固定了
    容量、每轮计算量与每轮通信量——每个客户端只属于一个组、每轮只上传下载一次，
    与组是怎么划出来的无关。

      'hc'         -> 原样返回 base_clusters（被测方法）
      'random'     -> 随机划分但保留 base_clusters 的组大小：num_K 与参数预算完全
                      一致，只是分组不再携带域信息
      'roundrobin' -> 确定性的非域感知划分，组大小同样一致
      'singleton'  -> 每客户端一个提取器（num_K = num_clients）：容量放大探针。若
                      「仅靠容量」就能解释增益，这个严格更大的集成不可能更差
    """
    if grouping == 'hc':
        return [list(g) for g in base_clusters]
    if grouping == 'singleton':
        return [[c] for c in range(num_clients)]
    sizes = [len(g) for g in base_clusters]
    if grouping == 'random':
        # 刻意用局部 RandomState：抽划分不能推进全局 numpy 随机流，否则客户端参与
        # 序列与 DPCR 的 alpha 采样都会偏离同配置的 hc 运行，对照就不再是单变量。
        order = np.random.RandomState(seed).permutation(num_clients)
    elif grouping == 'roundrobin':
        order = np.arange(num_clients)
    else:
        raise ValueError(f'unknown --grouping {grouping!r}')
    out, k = [], 0
    for s in sizes:
        out.append(sorted(int(c) for c in order[k:k + s]))
        k += s
    return out


def make_test_loaders(args, train_domains, class_to_idx):
    """协议 B：测试集只覆盖参与训练的 K 个域的测试池。"""
    test_transform = C.get_test_transform()
    test_datasets = [C.ImageFolderCustom(d, args.data_root, False, test_transform, class_to_idx)
                     for d in train_domains]
    test_dls = [DataLoader(ds, batch_size=args.test_batch_size, shuffle=False) for ds in test_datasets]
    sizes = {d: len(ds) for d, ds in zip(train_domains, test_datasets)}
    return test_dls, sizes


def global_evaluate(model, test_dls, device):
    status = model.training
    model.eval()
    accs = []
    for test_dl in test_dls:
        total, top1 = 0.0, 0.0
        for images, labels in test_dl:
            with torch.no_grad():
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = torch.topk(outputs, 1, dim=-1)
                top1 += (labels.view(-1, 1) == pred).sum().item()
                total += labels.size(0)
        accs.append(round(100 * top1 / max(1, total), 2))
    model.train(status)
    return accs


def infer_encoder_out_dim(encoder, train_dls, device):
    status = encoder.training
    encoder.eval()
    with torch.no_grad():
        for dl in train_dls:
            for x, _ in dl:
                out = encoder(x.to(device))
                encoder.train(status)
                return int(out.shape[1])
    encoder.train(status)
    raise RuntimeError('无法推断 encoder 输出维度：所有 train dataloader 均为空')


def build_client_to_cluster_map(clusters):
    mapping = {}
    for cluster_id, cluster_clients in enumerate(clusters):
        for cid in cluster_clients:
            mapping[int(cid)] = int(cluster_id)
    return mapping


# ==================== 簇内 FedAvg（encoder + 小分类头） ====================
def train_net_encoder_classifier(net_id, net_encoder, net_classifier, train_dl, epochs, lr, device,
                                 epoch_logger=None, stage_tag='encoder_cluster'):
    optimizer = torch.optim.SGD(
        [{'params': net_encoder.parameters()}, {'params': net_classifier.parameters()}],
        lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)

    total_train_time = 0.0
    local_class_features = {}
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_start = time.perf_counter()
        for x, target in train_dl:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            encoded = net_encoder(x)
            scores = net_classifier(encoded)
            loss = criterion(scores, target)
            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(float(loss.item()))

            with torch.no_grad():
                for i in range(len(target)):
                    label = int(target[i].item())
                    if label not in local_class_features:
                        local_class_features[label] = []
                    local_class_features[label].append(encoded[i, :].detach().cpu())

        epoch_time = time.perf_counter() - epoch_start
        total_train_time += epoch_time
        epoch_loss = sum(epoch_loss_collector) / max(1, len(epoch_loss_collector))
        print(f'[{stage_tag}] net={net_id} epoch={epoch} loss={epoch_loss:.6f} time={epoch_time:.4f}s')
        if epoch_logger is not None:
            epoch_logger.write_row({
                'stage': stage_tag, 'net_id': net_id, 'epoch': epoch,
                'epoch_loss': round(epoch_loss, 8), 'epoch_time_sec': round(epoch_time, 6),
                'num_batches': len(train_dl),
                'num_samples': len(train_dl.sampler) if hasattr(train_dl, 'sampler') else len(train_dl.dataset),
            })
    return C.agg_features(local_class_features), total_train_time


def local_train_net_encoder_classifier(encoder_list, classifier_list, selected, net_dataidx_map,
                                       domain_list, train_dls, local_epochs, device='cpu',
                                       epoch_logger=None):
    local_normal_protos, per_client_train_time = {}, {}
    for net_id in selected:
        dataidxs = net_dataidx_map[net_id]
        print(f'Training network {net_id}, domain: {domain_list[net_id]}, '
              f'number of training samples {len(dataidxs)}.')
        encoder_list[net_id].to(device)
        classifier_list[net_id].to(device)
        protos, client_time = train_net_encoder_classifier(
            net_id=net_id, net_encoder=encoder_list[net_id], net_classifier=classifier_list[net_id],
            train_dl=train_dls[net_id], epochs=local_epochs, lr=0.01, device=device,
            epoch_logger=epoch_logger, stage_tag='encoder_cluster')
        encoder_list[net_id].to('cpu')
        classifier_list[net_id].to('cpu')
        local_normal_protos[net_id] = protos
        per_client_train_time[net_id] = client_time

    individual_domain_protos = (C.proto_aggregation(selected, local_normal_protos)
                                if len(selected) > 0 else {})
    return per_client_train_time, individual_domain_protos


def collect_virtual_features_for_client(train_dl, encoder, proto_pool_by_class, alpha_low, alpha_high, device):
    uploaded_features, uploaded_labels = [], []
    encoder_status = encoder.training
    encoder.eval()
    for input_x, target in train_dl:
        input_x, target = input_x.to(device), target.to(device)
        with torch.no_grad():
            raw_features = encoder(input_x)
        for i in range(len(target)):
            label = int(target[i].item())
            if label not in proto_pool_by_class:
                continue
            local_feat = raw_features[i].detach().cpu()
            for domain_proto in proto_pool_by_class[label]:
                alpha = float(np.random.uniform(alpha_low, alpha_high))
                uploaded_features.append((1.0 - alpha) * local_feat + alpha * domain_proto)
                uploaded_labels.append(label)
    encoder.train(encoder_status)
    if len(uploaded_features) == 0:
        return None, None, 0, 0
    X = torch.stack(uploaded_features, dim=0)
    y = torch.tensor(uploaded_labels, dtype=torch.long)
    return X, y, C.tensor_nbytes(X) + C.tensor_nbytes(y), len(uploaded_labels)


def state_dict_nbytes(state_dict) -> int:
    return int(sum(C.tensor_nbytes(v) for v in state_dict.values() if isinstance(v, torch.Tensor)))


def proto_dict_nbytes(proto_dict: dict) -> int:
    return int(sum(C.tensor_nbytes(v) for v in proto_dict.values() if isinstance(v, torch.Tensor)))


# ==================== encoder 组装与 DPCR（主评估与逐轮 probe 共用同一实现） ====================
def assemble_encoder_all(cluster_paras, num_classes, device):
    """把每簇的全局 encoder 参数装回独立 encoder，再用 CombineAllModel 拼成 hidden_dim*num_K 维。"""
    nets = []
    for state_dict in cluster_paras:
        enc = resnet10Encoder(nclasses=num_classes)
        enc.load_state_dict(state_dict)
        enc.to(device)
        nets.append(enc)
    model = CombineAllModel(nets)
    model.to(device)
    return model


def build_proto_pool(encoder_all, train_dls, num_clients, clusters, device):
    """DPCR 原型池：逐客户端融合特征 -> 类原型 -> 按「簇」聚合 -> 类到原型列表。"""
    total_fusion_protos = C.get_fusion_protos(encoder_all, train_dls, num_clients, device)
    cluster_domain_prototypes = {
        int(cid): C.proto_aggregation([int(x) for x in members], total_fusion_protos)
        for cid, members in enumerate(clusters)}
    proto_pool_by_class = defaultdict(list)
    for _, proto_dict in cluster_domain_prototypes.items():
        for label, proto in proto_dict.items():
            proto_pool_by_class[int(label)].append(proto.detach().cpu().squeeze(0))
    return total_fusion_protos, cluster_domain_prototypes, proto_pool_by_class


def collect_virtual_pool(encoder_all, train_dls, total_fusion_protos, proto_pool_by_class,
                         num_clients, alpha_low, alpha_high, device):
    """逐客户端做原型平移采样，返回拼接后的服务器端训练池与逐客户端通信明细。"""
    x_list, y_list = [], []
    per_local_proto_bytes, per_virtual_bytes, per_num_virtual = {}, {}, {}
    start = time.perf_counter()
    for client_id in range(num_clients):
        local_proto_bytes = proto_dict_nbytes(total_fusion_protos[client_id])
        X_client, y_client, virtual_upload_bytes, num_virtual = collect_virtual_features_for_client(
            train_dl=train_dls[client_id], encoder=encoder_all, proto_pool_by_class=proto_pool_by_class,
            alpha_low=alpha_low, alpha_high=alpha_high, device=device)
        if X_client is not None:
            x_list.append(X_client)
            y_list.append(y_client)
        per_local_proto_bytes[client_id] = local_proto_bytes
        per_virtual_bytes[client_id] = virtual_upload_bytes
        per_num_virtual[client_id] = num_virtual
    if len(x_list) == 0:
        raise RuntimeError('DPCR 未生成任何虚拟特征：请检查原型池与 alpha_low/alpha_high')
    return {
        'X': torch.cat(x_list, dim=0),
        'y': torch.cat(y_list, dim=0),
        'per_local_proto_bytes': per_local_proto_bytes,
        'per_virtual_bytes': per_virtual_bytes,
        'per_num_virtual': per_num_virtual,
        'collect_time_sec': time.perf_counter() - start,
    }


def train_dpcr_classifier(X, y, in_dim, output_dim, epochs, lr, batch_size, device,
                          epoch_logger=None, log_every=20):
    """在虚拟特征池上重训全局线性分类头（与最终 DPCR 协议同一实现）。"""
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    clf = SimpleClassifier(hidden_dim=in_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    total_time = 0.0
    for epoch in range(epochs):
        losses, epoch_start = [], time.perf_counter()
        for feat, target in loader:
            feat, target = feat.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(clf(feat), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_time = time.perf_counter() - epoch_start
        total_time += epoch_time
        epoch_loss = sum(losses) / max(1, len(losses))
        if epoch_logger is not None:
            epoch_logger.write_row({'epoch': epoch, 'num_batches': len(loader),
                                    'loss': round(epoch_loss, 8), 'epoch_time_sec': round(epoch_time, 6)})
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f'[dpcr] epoch={epoch} loss={epoch_loss:.6f}')
    return clf, total_time


def format_domain_accs(train_domains, accs):
    """协议 B 的统口径：仅 K 个训练域，mean{K} / std{K}（同时给出与域数无关的 mean_acc/std_acc）。"""
    vals = np.asarray(accs, dtype=np.float64)
    k = len(train_domains)
    mean_val = round(float(vals.mean()), 3)
    std_val = round(float(vals.std(ddof=0)), 3) if len(vals) > 1 else 0.0
    payload = {'acc_per_domain': {d: float(a) for d, a in zip(train_domains, accs)},
               'mean_acc': mean_val,
               'std_acc': std_val}
    payload[f'mean{k}'] = mean_val
    payload[f'std{k}'] = std_val
    return payload


def probe_round_accs(encoder_all, train_dls, clusters, num_clients, test_dls, train_domains,
                     args, device, in_dim, output_dim):
    """逐轮 probe：走与最终评估完全相同的 DPCR 协议，只把重训轮数压到 --probe_epochs。

    用随机初始化的线性头直接评估会得到无意义的数值（无法与最终 mean_acc 或 Comparied
    基线的逐轮曲线对齐），因此这里复用 build_proto_pool / train_dpcr_classifier。
    注意：虚拟特征采样会消耗 np.random 流，故开启 --eval_every 会改变后续轮次的
    客户端参与采样（与 --eval_every 0 的运行不可比，需要曲线时统一固定开启）。
    """
    total_fusion_protos, _, proto_pool_by_class = build_proto_pool(
        encoder_all, train_dls, num_clients, clusters, device)
    pool = collect_virtual_pool(encoder_all, train_dls, total_fusion_protos, proto_pool_by_class,
                                num_clients, args.alpha_low, args.alpha_high, device)
    clf, retrain_time = train_dpcr_classifier(
        pool['X'], pool['y'], in_dim, output_dim, epochs=args.probe_epochs, lr=args.server_retrain_lr,
        batch_size=args.server_retrain_batch_size, device=device, log_every=0)
    accs = global_evaluate(CombineModel(encoder_all, clf), test_dls, device)
    item = format_domain_accs(train_domains, accs)
    item['probe_retrain_sec'] = round(retrain_time, 4)
    item['num_virtual_samples'] = int(len(pool['y']))
    clf.cpu()
    del total_fusion_protos, proto_pool_by_class, pool
    return accs, item


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description='FKPL on DomainNet-subsetA (4 training domains, protocol B)')
    parser.add_argument('--data_root', type=str, default=C.DEFAULT_DATA_ROOT)
    parser.add_argument('--layout_file', type=str, default='./fkpl_diagnosis/client_layout.json')
    parser.add_argument('--calibration_file', type=str, default='./fkpl_diagnosis/beta_calibration.json')
    parser.add_argument('--train_domains', type=str, nargs='*', default=None,
                        help='覆盖诊断产物给出的训练域（默认 4 个域全用）')
    parser.add_argument('--client_counts', type=int, nargs='*', default=None,
                        help='直接指定每域客户端数（按 --train_domains 顺序），给定后忽略 layout_file；'
                             '例如 5 5 5 5 可对齐同机基线的等客户端布局')
    parser.add_argument('--percents', type=float, nargs='*', default=None,
                        help='与 --client_counts 一一对应的每客户端采样比例，同域之和需 <= 1')
    parser.add_argument('--outdir', type=str, default=None,
                        help='默认 ./runs/domainnet4d_seed<seed>')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--data_seed', type=int, default=None,
                        help='仅用于数据划分（客户端->域映射与每客户端采样）的种子；'
                             '默认沿用 --seed。传固定值（如 7）可锁定划分，让 --seed 只影响初始化与训练随机性。')
    parser.add_argument('--num_clients', type=int, default=None, help='仅作校验，实际由布局决定')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--budget', type=int, default=10)
    parser.add_argument('--n_basis', type=int, default=5)
    parser.add_argument('--basis_mode', type=str, default='budget', choices=['budget', 'original'])
    parser.add_argument('--cluster_alpha', type=float, default=None,
                        help='层次聚类阈值 beta（度）；默认读 calibration_file')
    parser.add_argument('--linkage', type=str, default=None, choices=['average', 'complete', 'minimum'])
    parser.add_argument('--use_hc_for_training', action='store_true',
                        help='训练簇改用在线层次聚类结果（默认用按域动态生成的域纯簇）')
    parser.add_argument('--profile_hc_once', action='store_true', help='仅诊断：即使不用于训练也算一次 HC')
    parser.add_argument('--grouping', type=str, default='hc',
                        choices=['hc', 'random', 'roundrobin', 'singleton'],
                        help='客户端如何划分到各独立训练的提取器。hc 为被测方法；'
                             'random / roundrobin 是同容量集成对照——保留组大小，因此 num_K、参数预算、'
                             '每轮计算量与每轮通信量都与 hc 运行一致，只有分组规则不同；'
                             'singleton 每客户端一个提取器（num_K = num_clients），作为容量放大探针。')
    parser.add_argument('--group_seed', type=int, default=None,
                        help='--grouping 随机划分的种子，默认沿用 --seed；'
                             '经私有 RandomState 消费，不扰动本次运行的其它随机流。')
    parser.add_argument('--stop_after_clustering', action='store_true',
                        help='只跑到 SVD -> 邻接矩阵 -> 层次聚类诊断就退出，用于核查 beta 标定一致性（不训练）')
    parser.add_argument('--encoder_round', type=int, default=100)
    parser.add_argument('--local_epochs_encoder', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--output_dim', type=int, default=None, help='默认取训练集类别数（subsetA=10）')
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--run_encoder_stage', action='store_true', default=True)
    parser.add_argument('--no_encoder_stage', dest='run_encoder_stage', action='store_false')
    parser.add_argument('--encoder_ckpt', type=str, default='',
                        help='不训练 encoder 时加载的 CombineAllModel checkpoint 路径')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='每多少 encoder 轮做一次 probe 评估（与最终同协议）；<=0 关闭')
    parser.add_argument('--probe_epochs', type=int, default=30,
                        help='逐轮 probe 重训 DPCR 线性头的轮数（最终评估始终用 --retrain_epochs）')
    parser.add_argument('--last_k_rounds', type=int, default=5,
                        help='汇总逐轮曲线最后 K 个评估点的 mean_acc（与 Comparied 基线同口径）')
    parser.add_argument('--retrain_epochs', type=int, default=200)
    parser.add_argument('--alpha_low', type=float, default=0.2)
    parser.add_argument('--alpha_high', type=float, default=0.5)
    parser.add_argument('--server_retrain_lr', type=float, default=0.01)
    parser.add_argument('--server_retrain_batch_size', type=int, default=64)
    args = parser.parse_args()

    train_domains, client_counts, percents, num_clients, layout_source, \
        layout_order, layout_selected = resolve_layout(args)
    cluster_alpha, linkage, beta_source = resolve_beta(args)
    mean_key, std_key = f'mean{len(train_domains)}', f'std{len(train_domains)}'

    if args.outdir is None:
        args.outdir = f'./runs/domainnet4d_seed{args.seed}'
    final_dir = os.path.join(args.outdir, 'final')
    C.ensure_dir(args.outdir)
    C.ensure_dir(final_dir)

    data_seed = args.seed if args.data_seed is None else args.data_seed
    C.set_seed(data_seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print(f'FKPL / DomainNet-subsetA / {len(train_domains)} training domains (protocol B)')
    print('=' * 70)
    print(f'train_domains       = {train_domains}')
    print(f'client_counts       = {client_counts}  (num_clients={num_clients})')
    print(f'percent_per_client  = {percents}')
    print(f'layout source       = {layout_source}')
    print(f'beta (cluster_alpha)= {cluster_alpha}  linkage={linkage}  source={beta_source}')
    print(f'clusters used for training = {"HC" if args.use_hc_for_training else "domain-dynamic"}')
    print(f'seed={args.seed}, data_seed={data_seed} (data split locked)'
          if args.data_seed is not None else f'seed={args.seed}, data_seed=None -> reuse seed')
    print(f'device={device}, outdir={args.outdir}')

    # ---------------- 数据 ----------------
    class_to_idx, class_sets = C.build_global_class_to_idx(args.data_root, train_domains)
    num_classes = len(class_to_idx)
    output_dim = int(args.output_dim or num_classes)
    if output_dim != num_classes:
        print(f'[WARN] --output_dim={output_dim} 与训练集类别数 {num_classes} 不一致')
    print(f'global label space = {num_classes} classes shared across {train_domains}')

    # 展开顺序优先用诊断产物的 centrality 序，以逐位复现 beta 标定时的布局与随机流
    expanded = build_selected_domain_list(train_domains, client_counts,
                                         order_by_domain=layout_order or train_domains)
    selected_domain_list = np.random.permutation(expanded)
    print(f'selected_domain_list = {list(selected_domain_list)}')
    if layout_order:
        print(f'expand order = {layout_order} ( centrality 序，与 Stage B/C 标定一致 )')
    if layout_selected is not None:
        same = [str(d) for d in selected_domain_list] == list(layout_selected)
        print(f'layout vs 诊断产物 selected_domain_list 一致 = {same}')
        if not same:
            print('[WARN] 实际布局与诊断标定布局不一致，beta 标定的簇纯度不再可直接引用；'
                  '请确认 client_layout.json 与数据/代码版本是否同步')
    print(f'domain counts = {dict(Counter([str(d) for d in selected_domain_list]))}')

    train_transform = C.get_train_transform()
    train_dataset_list = [C.ImageFolderCustom(str(d), args.data_root, True, train_transform, class_to_idx)
                          for d in selected_domain_list]
    test_dls, test_sizes = make_test_loaders(args, train_domains, class_to_idx)
    traindata_cls_counts, net_dataidx_map, train_dls, _ = C.partition_domain_skew_loaders(
        train_dataset_list, [], percents, batch_size=args.batch_size)
    print(f'test pools (protocol B, training domains only) = {test_sizes}')

    # 划分完成后重新播种：encoder 初始化、DPCR 的 alpha 采样、增强与 shuffle 随 --seed 变化
    C.set_seed(args.seed)

    train_total = {str(d): 0 for d in train_domains}
    for i, idxs in net_dataidx_map.items():
        train_total[str(selected_domain_list[i])] += len(idxs)
    per_client_samples = {i: len(v) for i, v in net_dataidx_map.items()}
    print(f'train samples per domain = {train_total}')
    print(f'per-client samples: min={min(per_client_samples.values())} max={max(per_client_samples.values())} '
          f'mean={np.mean(list(per_client_samples.values())):.1f}')
    if min(per_client_samples.values()) < 50:
        print('[WARN] 存在样本数 <50 的客户端，簇内 FedAvg 可能不稳定')

    # ---------------- 阶段 1/2：子空间 + 簇 ----------------
    preprocess_start = time.perf_counter()
    svd_logger = C.CsvLogger(os.path.join(args.outdir, 'svd_preprocess_stats.csv'), C.SVD_PREPROCESS_FIELDNAMES)
    traindata_cls_ratio = C.compute_basis_budget(traindata_cls_counts, num_clients, budget=args.budget)
    U_clients, svd_rows = C.build_U_clients(
        train_dataset_list, net_dataidx_map, traindata_cls_ratio, selected_domain_list, num_clients,
        n_basis=args.n_basis, budget_mode=args.basis_mode, verbose=True)
    for row in svd_rows:
        svd_logger.write_row(row)
    per_client_u_bytes = {int(r['client_id']): int(r['u_bytes']) for r in svd_rows}
    per_client_svd_time = {int(r['client_id']): float(r['svd_time_sec']) for r in svd_rows}

    adj_mat = C.compute_adjacency(U_clients)
    np.savetxt(os.path.join(args.outdir, 'adjacency_degrees.csv'), adj_mat, delimiter=',', fmt='%.6f')
    adj_logger = C.CsvLogger(os.path.join(args.outdir, 'adjacency_scores.csv'),
                             ['client_i', 'domain_i', 'client_j', 'domain_j', 'angle_degrees'])
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            adj_logger.write_row({'client_i': i, 'domain_i': str(selected_domain_list[i]),
                                  'client_j': j, 'domain_j': str(selected_domain_list[j]),
                                  'angle_degrees': round(float(adj_mat[i, j]), 6)})

    hc_start = time.perf_counter()
    hc_clusters = C.hierarchical_clustering(copy.deepcopy(adj_mat), thresh=float(cluster_alpha), linkage=linkage)
    hc_time = time.perf_counter() - hc_start
    domain_clusters = C.build_fixed_domain_clusters(selected_domain_list, domain_order=train_domains)

    # 簇来源只有两种（无任何按客户端索引硬编码的簇）：
    #   --use_hc_for_training -> 重标定 beta 下的在线层次聚类结果；
    #   默认                 -> 按「域」动态生成的域纯簇。
    # --profile_hc_once 不改变训练簇来源，仅表示即使走默认路径也已记录一份 HC 诊断。
    base_clusters = [list(c) for c in (hc_clusters if args.use_hc_for_training else domain_clusters)]
    group_seed = args.seed if args.group_seed is None else args.group_seed
    clusters = build_control_clusters(args.grouping, base_clusters, num_clients, group_seed)
    hc_detail, hc_purity = C.cluster_domain_report(hc_clusters, selected_domain_list)
    dp_detail, dp_purity = C.cluster_domain_report(domain_clusters, selected_domain_list)
    used_detail, used_purity = C.cluster_domain_report(clusters, selected_domain_list)
    # 审计轨：记下每个提取器组实际包含什么，归档的运行能自陈分组规则、域纯度与样本预算
    group_audit = []
    for gid, members in enumerate(clusters):
        doms = sorted({str(selected_domain_list[c]) for c in members})
        group_audit.append({
            'group_id': gid,
            'clients': [int(c) for c in members],
            'size': len(members),
            'domains': doms,
            'domain_pure': len(doms) == 1,
            'num_samples': int(sum(len(net_dataidx_map[c]) for c in members)),
        })
    print(f'HC clusters(beta={cluster_alpha})   = {C.fmt_clusters(hc_clusters)} purity={hc_purity:.4f}')
    print(f'domain-dynamic clusters       = {C.fmt_clusters(domain_clusters)} purity={dp_purity:.4f}')
    print(f'grouping={args.grouping} (seed={group_seed}) base={C.fmt_clusters(base_clusters)}')
    print(f'clusters USED for training    = {C.fmt_clusters(clusters)} purity={used_purity:.4f}')

    clustering_summary = {
        'num_clients': num_clients,
        'train_domains': train_domains,
        'selected_domain_list': [str(d) for d in selected_domain_list],
        'cluster_source': 'hc' if args.use_hc_for_training else 'domain_dynamic',
        'grouping': args.grouping,
        'group_seed': int(group_seed),
        'num_K': int(len(clusters)),
        'cluster_alpha': cluster_alpha,
        'beta_source': beta_source,
        'linkage': linkage,
        'layout_source': layout_source,
        'client_counts': client_counts,
        'percent_per_client': percents,
        'clusters_used_for_training': [[int(x) for x in c] for c in clusters],
        'base_clusters_before_grouping': [[int(x) for x in c] for c in base_clusters],
        'group_audit': group_audit,
        'clusters_used_domain_purity': round(used_purity, 4),
        'clusters_used_detail': used_detail,
        'clusters_from_hc': [[int(x) for x in c] for c in hc_clusters],
        'clusters_from_hc_domain_purity': round(hc_purity, 4),
        'clusters_from_hc_detail': hc_detail,
        'domain_dynamic_clusters': [[int(x) for x in c] for c in domain_clusters],
        'domain_dynamic_domain_purity': round(dp_purity, 4),
        'hc_matches_domain_dynamic': C.fmt_clusters(hc_clusters) == C.fmt_clusters(domain_clusters),
        'train_samples_per_client': {int(k): int(v) for k, v in per_client_samples.items()},
        'train_samples_per_domain': {k: int(v) for k, v in train_total.items()},
        'test_samples_per_domain': {k: int(v) for k, v in test_sizes.items()},
        'total_subspace_upload_bytes': int(sum(per_client_u_bytes.values())),
        'total_subspace_upload_megabytes': round(C.bytes_to_mb(sum(per_client_u_bytes.values())), 6),
        'avg_subspace_upload_bytes_per_client': int(np.mean(list(per_client_u_bytes.values()))),
        'avg_client_svd_time_sec': round(float(np.mean(list(per_client_svd_time.values()))), 6),
        'server_hc_time_sec': round(hc_time, 6),
        'one_shot_clustering_total_time_sec': round(time.perf_counter() - preprocess_start, 6),
        'seed': args.seed,
        'data_seed': data_seed,
    }
    with open(os.path.join(args.outdir, 'clustering_once_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(clustering_summary, f, indent=2, ensure_ascii=False)
    write_row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                 for k, v in clustering_summary.items()}
    C.write_single_row_csv(os.path.join(args.outdir, 'clustering_once_summary.csv'), write_row)

    if args.stop_after_clustering:
        calib = load_json(args.calibration_file) or {}
        calib_purity = calib.get('hc_purity_at_chosen_beta')
        calib_sdl = calib.get('selected_domain_list')
        calib_v = calib.get('V_at_chosen_beta')
        calib_hc = calib.get('hc_clusters_at_chosen_beta')
        adj_cal = calib.get('adjacency_degrees')
        adj_diff = ('n/a' if not adj_cal else
                    round(float(np.max(np.abs(np.array(adj_cal, dtype=float) - adj_mat))), 6))
        print('\n' + '=' * 70)
        print('[stop_after_clustering] 已完成 SVD -> 邻接矩阵 -> HC 诊断，提前退出（未训练 encoder）')
        print('=' * 70)
        print(f'  本次 selected_domain_list      = {[str(d) for d in selected_domain_list]}')
        print(f'  标定文件 selected_domain_list  = {calib_sdl}')
        print(f'  layout 一致   = {[str(d) for d in selected_domain_list] == [str(d) for d in (calib_sdl or [])]}')
        print(f'  邻接矩阵最大差异 = {adj_diff} 度   (0.0 表示与标定完全同流)')
        print(f'  本次 HC@beta={cluster_alpha} = {C.fmt_clusters(hc_clusters)}  V={len(hc_clusters)}  纯度={hc_purity:.4f}')
        print(f'  标定 HC@beta      = {calib_hc}  V={calib_v}  纯度={calib_purity}')
        print(f'  产物 -> {os.path.join(args.outdir, "clustering_once_summary.json")}')
        return

    # ---------------- 阶段 3：簇内 FedAvg 训练 encoder ----------------
    num_K = len(clusters)
    fusion_dim = args.hidden_dim * num_K
    encoder_list = [resnet10Encoder(nclasses=num_classes) for _ in range(num_clients)]
    classifier_list = [SimpleClassifier(hidden_dim=args.hidden_dim, output_dim=output_dim)
                       for _ in range(num_clients)]
    encoder_global_para = [copy.deepcopy(encoder_list[0].state_dict()) for _ in range(num_K)]
    classifier_global_para = [copy.deepcopy(classifier_list[0].state_dict()) for _ in range(num_K)]

    small_encoder_bytes = state_dict_nbytes(encoder_list[0].state_dict())
    small_classifier_bytes = state_dict_nbytes(classifier_list[0].state_dict())
    small_model_bytes = small_encoder_bytes + small_classifier_bytes

    encoder_stage_summary, round_rows = None, []
    if args.run_encoder_stage:
        print('\n' + '=' * 70)
        print(f'Stage: cluster-wise FedAvg for encoders ({args.encoder_round} rounds, {num_K} clusters)')
        print('=' * 70)
        round_logger = C.CsvLogger(
            os.path.join(args.outdir, 'encoder_stage_round_stats.csv'),
            ['cluster_id', 'round', 'num_selected', 'selected_clients',
             'downlink_bytes', 'downlink_megabytes', 'uplink_bytes', 'uplink_megabytes',
             'total_comm_bytes', 'total_comm_megabytes', 'avg_comm_bytes_per_client',
             'avg_comm_megabytes_per_client', 'local_train_time_sec', 'aggregation_time_sec', 'round_time_sec'])
        epoch_logger = C.CsvLogger(
            os.path.join(args.outdir, 'encoder_stage_epoch_stats.csv'),
            ['stage', 'net_id', 'epoch', 'epoch_loss', 'epoch_time_sec', 'num_batches', 'num_samples'])

        overall_comm, overall_time = 0, 0.0
        round_global_comm = [0 for _ in range(args.encoder_round)]
        round_global_time = [0.0 for _ in range(args.encoder_round)]
        round_global_local = [0.0 for _ in range(args.encoder_round)]
        round_global_agg = [0.0 for _ in range(args.encoder_round)]
        eval_rows = []

        for r in range(args.encoder_round):
            print(f'\n====== comm round {r} ======')
            top = max(1, int(num_clients * args.sample_ratio))
            participation = np.random.permutation(np.arange(num_clients))[:top]

            for cluster_id in range(num_K):
                selected = [int(cid) for cid in clusters[cluster_id] if cid in participation]
                if len(selected) == 0:
                    print(f'no participating client in cluster {cluster_id}, skip')
                    continue
                round_start = time.perf_counter()
                downlink_bytes = len(selected) * small_model_bytes

                for idx in selected:
                    encoder_list[idx].load_state_dict(encoder_global_para[cluster_id])
                    classifier_list[idx].load_state_dict(classifier_global_para[cluster_id])

                local_train_start = time.perf_counter()
                local_train_net_encoder_classifier(
                    encoder_list, classifier_list, selected, net_dataidx_map,
                    selected_domain_list, train_dls, local_epochs=args.local_epochs_encoder,
                    device=device, epoch_logger=epoch_logger)
                local_train_time = time.perf_counter() - local_train_start

                agg_start = time.perf_counter()
                total_data_samples = sum([len(net_dataidx_map[cid]) for cid in selected])
                fed_avg_freqs = [len(net_dataidx_map[cid]) / total_data_samples for cid in selected]
                # encoder 与簇内小分类头都要做样本量加权 FedAvg（与基座脚本一致）
                for pos, client_id in enumerate(selected):
                    net_para = encoder_list[client_id].cpu().state_dict()
                    for key in net_para:
                        if pos == 0:
                            encoder_global_para[cluster_id][key] = net_para[key] * fed_avg_freqs[pos]
                        else:
                            encoder_global_para[cluster_id][key] += net_para[key] * fed_avg_freqs[pos]
                for pos, client_id in enumerate(selected):
                    net_para = classifier_list[client_id].cpu().state_dict()
                    for key in net_para:
                        if pos == 0:
                            classifier_global_para[cluster_id][key] = net_para[key] * fed_avg_freqs[pos]
                        else:
                            classifier_global_para[cluster_id][key] += net_para[key] * fed_avg_freqs[pos]
                uplink_bytes = len(selected) * small_model_bytes
                agg_time = time.perf_counter() - agg_start

                round_time = time.perf_counter() - round_start
                total_comm_bytes = downlink_bytes + uplink_bytes
                row = {
                    'cluster_id': cluster_id, 'round': r, 'num_selected': len(selected),
                    'selected_clients': '-'.join(str(x) for x in selected),
                    'downlink_bytes': downlink_bytes,
                    'downlink_megabytes': round(C.bytes_to_mb(downlink_bytes), 6),
                    'uplink_bytes': uplink_bytes,
                    'uplink_megabytes': round(C.bytes_to_mb(uplink_bytes), 6),
                    'total_comm_bytes': total_comm_bytes,
                    'total_comm_megabytes': round(C.bytes_to_mb(total_comm_bytes), 6),
                    'avg_comm_bytes_per_client': int(total_comm_bytes / len(selected)),
                    'avg_comm_megabytes_per_client': round(C.bytes_to_mb(total_comm_bytes / len(selected)), 6),
                    'local_train_time_sec': round(local_train_time, 6),
                    'aggregation_time_sec': round(agg_time, 6),
                    'round_time_sec': round(round_time, 6),
                }
                round_logger.write_row(row)
                round_rows.append(row)

                overall_comm += total_comm_bytes
                overall_time += round_time
                round_global_comm[r] += total_comm_bytes
                round_global_time[r] += round_time
                round_global_local[r] += local_train_time
                round_global_agg[r] += agg_time

            if args.eval_every and (r + 1) % args.eval_every == 0:
                probe_encoder = assemble_encoder_all(encoder_global_para, num_classes, device)
                accs, item = probe_round_accs(
                    probe_encoder, train_dls, clusters, num_clients, test_dls, train_domains,
                    args, device, fusion_dim, output_dim)
                item = {'round': r, **item}
                eval_rows.append(item)
                print(f'[eval] round {r}: ' + ', '.join(f'{d}={a}' for d, a in item['acc_per_domain'].items())
                      + f' | {mean_key}={item[mean_key]} {std_key}={item[std_key]}')
                probe_encoder.cpu()
                del probe_encoder

        encoder_all = assemble_encoder_all(encoder_global_para, num_classes, device)
        torch.save(encoder_all, os.path.join(final_dir, 'encoder_all.pth'))

        encoder_stage_summary = {
            'small_encoder_bytes': small_encoder_bytes,
            'small_encoder_megabytes': round(C.bytes_to_mb(small_encoder_bytes), 6),
            'small_classifier_bytes': small_classifier_bytes,
            'small_classifier_megabytes': round(C.bytes_to_mb(small_classifier_bytes), 6),
            'small_model_bytes': small_model_bytes,
            'small_model_megabytes': round(C.bytes_to_mb(small_model_bytes), 6),
            'total_encoder_stage_comm_bytes': int(overall_comm),
            'total_encoder_stage_comm_megabytes': round(C.bytes_to_mb(overall_comm), 6),
            'avg_encoder_stage_comm_per_global_round_bytes': int(np.mean(round_global_comm)),
            'avg_encoder_stage_comm_per_global_round_megabytes': round(C.bytes_to_mb(np.mean(round_global_comm)), 6),
            'total_encoder_stage_time_sec': round(overall_time, 6),
            'avg_encoder_stage_time_per_global_round_sec': round(float(np.mean(round_global_time)), 6),
            'avg_encoder_stage_local_train_time_per_global_round_sec': round(float(np.mean(round_global_local)), 6),
            'avg_encoder_stage_aggregation_time_per_global_round_sec': round(float(np.mean(round_global_agg)), 6),
            'encoder_round_eval': eval_rows,
        }
        # 逐轮曲线单独落盘（字段扁平化，与 Comparied 基线的 round_accs.json 可直接堆叠）
        acc_rows = []
        for item in eval_rows:
            row = {'round': item['round'], mean_key: item[mean_key], std_key: item[std_key]}
            row.update(item['acc_per_domain'])
            acc_rows.append(row)
        with open(os.path.join(args.outdir, 'encoder_round_accs.json'), 'w', encoding='utf-8') as f:
            json.dump(acc_rows, f, indent=2, ensure_ascii=False)
        if acc_rows:
            k = max(1, min(int(args.last_k_rounds), len(acc_rows)))
            tail = acc_rows[-k:]
            encoder_stage_summary['last_k_probes'] = k
            encoder_stage_summary['last_k_' + mean_key + '_avg'] = round(float(np.mean([x[mean_key] for x in tail])), 4)
            for d in train_domains:
                encoder_stage_summary[f'last_k_{d}_avg'] = round(float(np.mean([x[d] for x in tail])), 4)
            last_k_avg = encoder_stage_summary['last_k_' + mean_key + '_avg']
            print(f'[eval] last_k={k} {mean_key}_avg={last_k_avg}')
        eval_logger = C.CsvLogger(os.path.join(args.outdir, 'encoder_round_accs.csv'),
                                  ['round', mean_key, std_key] + list(train_domains))
        for row in acc_rows:
            eval_logger.write_row(row)
        with open(os.path.join(args.outdir, 'encoder_stage_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(encoder_stage_summary, f, indent=2, ensure_ascii=False)
        C.write_single_row_csv(os.path.join(args.outdir, 'encoder_stage_summary.csv'),
                               {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
                                for k, v in encoder_stage_summary.items()})
    else:
        if not args.encoder_ckpt or not os.path.exists(args.encoder_ckpt):
            raise SystemExit('--no_encoder_stage 需要提供一个已存在的 --encoder_ckpt（CombineAllModel 整体对象）')
        encoder_all = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
        encoder_all.to(device)
        actual = infer_encoder_out_dim(encoder_all, train_dls, device)
        if actual != fusion_dim:
            raise RuntimeError(f'加载的 encoder 输出维度 {actual} != hidden_dim*num_K={fusion_dim}')

    # ---------------- 阶段 4：DPCR 原型平移分类器重训 ----------------
    print('\n' + '=' * 70)
    print('Stage: DPCR prototype-translation classifier retraining')
    print('=' * 70)
    total_fusion_protos, cluster_domain_prototypes, proto_pool_by_class = build_proto_pool(
        encoder_all, train_dls, num_clients, clusters, device)
    client_to_cluster = build_client_to_cluster_map(clusters)

    classifier_bytes = state_dict_nbytes(
        SimpleClassifier(hidden_dim=fusion_dim, output_dim=output_dim).state_dict())

    # 虚拟特征采集（与逐轮 probe 同一个实现）
    pool = collect_virtual_pool(encoder_all, train_dls, total_fusion_protos, proto_pool_by_class,
                                num_clients, args.alpha_low, args.alpha_high, device)
    virtual_collect_time = pool['collect_time_sec']
    per_local_proto_bytes, per_virtual_bytes, per_num_virtual = (
        pool['per_local_proto_bytes'], pool['per_virtual_bytes'], pool['per_num_virtual'])
    per_proto_downlink = {cid: proto_dict_nbytes(cluster_domain_prototypes[client_to_cluster[cid]])
                          for cid in range(num_clients)}

    dpcr_logger = C.CsvLogger(
        os.path.join(args.outdir, 'dpcr_single_client_comm.csv'),
        ['client_id', 'domain', 'cluster_id', 'num_local_classes',
         'local_proto_upload_bytes', 'local_proto_upload_megabytes',
         'classifier_downlink_bytes', 'classifier_downlink_megabytes',
         'domain_proto_downlink_bytes', 'domain_proto_downlink_megabytes',
         'virtual_feature_upload_bytes', 'virtual_feature_upload_megabytes',
         'one_shot_total_bytes', 'one_shot_total_megabytes', 'num_virtual_samples'])
    for client_id in range(num_clients):
        local_proto_bytes = per_local_proto_bytes[client_id]
        cluster_proto_bytes = per_proto_downlink[client_id]
        virtual_upload_bytes = per_virtual_bytes[client_id]
        one_shot_total = local_proto_bytes + classifier_bytes + cluster_proto_bytes + virtual_upload_bytes
        dpcr_logger.write_row({
            'client_id': client_id, 'domain': str(selected_domain_list[client_id]),
            'cluster_id': client_to_cluster[client_id],
            'num_local_classes': len(total_fusion_protos[client_id]),
            'local_proto_upload_bytes': local_proto_bytes,
            'local_proto_upload_megabytes': round(C.bytes_to_mb(local_proto_bytes), 6),
            'classifier_downlink_bytes': classifier_bytes,
            'classifier_downlink_megabytes': round(C.bytes_to_mb(classifier_bytes), 6),
            'domain_proto_downlink_bytes': cluster_proto_bytes,
            'domain_proto_downlink_megabytes': round(C.bytes_to_mb(cluster_proto_bytes), 6),
            'virtual_feature_upload_bytes': virtual_upload_bytes,
            'virtual_feature_upload_megabytes': round(C.bytes_to_mb(virtual_upload_bytes), 6),
            'one_shot_total_bytes': one_shot_total,
            'one_shot_total_megabytes': round(C.bytes_to_mb(one_shot_total), 6),
            'num_virtual_samples': per_num_virtual[client_id]})

    X_server, y_server = pool['X'], pool['y']
    dpcr_epoch_logger = C.CsvLogger(os.path.join(args.outdir, 'dpcr_epoch_stats.csv'),
                                    ['epoch', 'num_batches', 'loss', 'epoch_time_sec'])
    dpcr_classifier, total_retrain_time = train_dpcr_classifier(
        X_server, y_server, fusion_dim, output_dim, epochs=args.retrain_epochs,
        lr=args.server_retrain_lr, batch_size=args.server_retrain_batch_size, device=device,
        epoch_logger=dpcr_epoch_logger, log_every=20)

    # ---------------- 阶段 5：协议 B 评估（仅 3 个训练域测试池） ----------------
    eval_start = time.perf_counter()
    final_model = CombineModel(encoder_all, dpcr_classifier)
    final_model.to(device)
    accs = global_evaluate(final_model, test_dls, device)
    evaluation_time = time.perf_counter() - eval_start

    final_stats = format_domain_accs(train_domains, accs)
    mean_acc, std_acc = final_stats['mean_acc'], final_stats['std_acc']
    torch.save(dpcr_classifier, os.path.join(final_dir, 'dpcr_classifier.pth'))
    torch.save(final_model, os.path.join(final_dir, 'final_model_full.pth'))

    total_local_proto = int(sum(per_local_proto_bytes.values()))
    total_classifier_down = int(num_clients * classifier_bytes)
    total_proto_down = int(sum(per_proto_downlink.values()))
    total_virtual_up = int(sum(per_virtual_bytes.values()))
    total_one_shot = total_local_proto + total_classifier_down + total_proto_down + total_virtual_up

    dpcr_summary = {
        'num_virtual_samples_total': int(len(y_server)),
        'avg_virtual_samples_per_client': round(float(np.mean(list(per_num_virtual.values()))), 3),
        'classifier_model_megabytes': round(C.bytes_to_mb(classifier_bytes), 6),
        'avg_local_proto_upload_megabytes_per_client': round(
            C.bytes_to_mb(float(np.mean(list(per_local_proto_bytes.values())))), 6),
        'avg_domain_proto_downlink_megabytes_per_client': round(
            C.bytes_to_mb(float(np.mean(list(per_proto_downlink.values())))), 6),
        'avg_virtual_feature_upload_megabytes_per_client': round(
            C.bytes_to_mb(float(np.mean(list(per_virtual_bytes.values())))), 6),
        'avg_one_shot_total_megabytes_per_client': round(C.bytes_to_mb(total_one_shot / num_clients), 6),
        'total_one_shot_comm_megabytes': round(C.bytes_to_mb(total_one_shot), 6),
        'virtual_feature_collect_time_sec': round(virtual_collect_time, 6),
        'retrain_epochs': args.retrain_epochs,
        'total_server_retrain_time_sec': round(total_retrain_time, 6),
        'avg_server_retrain_time_sec_per_epoch': round(total_retrain_time / args.retrain_epochs, 6),
        'evaluation_time_sec': round(evaluation_time, 6),
        'eval_protocol': 'B: training domains only',
        'final_acc_per_domain': {d: a for d, a in zip(train_domains, accs)},
        'final_mean_acc': round(mean_acc, 3),
        'final_std_acc': round(std_acc, 3),
        'final_worst_domain_acc': round(float(np.min(accs)), 3),
        'final_max_min_gap': round(float(np.max(accs) - np.min(accs)), 3),
        'num_probe_evals': len((encoder_stage_summary or {}).get('encoder_round_eval', [])),
    }
    # 与既有 Office/PACS 汇总脚本的字段习惯对齐：final_<domain>_acc 直接铺平
    dpcr_summary.update({f'final_{d}_acc': a for d, a in zip(train_domains, accs)})
    with open(os.path.join(args.outdir, 'dpcr_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(dpcr_summary, f, indent=2, ensure_ascii=False)

    table6_metrics = {
        'Number of clients': num_clients,
        'Number of clusters': num_K,
        'Total subspace upload (MB)': round(C.bytes_to_mb(sum(per_client_u_bytes.values())), 6),
        'Avg. upload per client (MB)': round(C.bytes_to_mb(float(np.mean(list(per_client_u_bytes.values())))), 6),
        'Total preprocessing time (s)': round(clustering_summary['one_shot_clustering_total_time_sec'], 6),
        'Server HC time (s)': round(clustering_summary['server_hc_time_sec'], 6),
        'Avg. per-client SVD time (s)': round(clustering_summary['avg_client_svd_time_sec'], 6),
        'Local prototype upload (MB)': dpcr_summary['avg_local_proto_upload_megabytes_per_client'],
        'Classifier downlink (MB)': dpcr_summary['classifier_model_megabytes'],
        'Domain prototype downlink (MB)': dpcr_summary['avg_domain_proto_downlink_megabytes_per_client'],
        'Virtual feature upload (MB)': dpcr_summary['avg_virtual_feature_upload_megabytes_per_client'],
        'One-shot total per client (MB)': dpcr_summary['avg_one_shot_total_megabytes_per_client'],
    }
    C.write_single_row_csv(os.path.join(args.outdir, 'table6_metrics.csv'), table6_metrics)

    table7_metrics = {
        'MAKP_Number_of_clusters': num_K,
        'MAKP_Encoder_size_MB': encoder_stage_summary['small_encoder_megabytes'] if encoder_stage_summary else 'N/A',
        'MAKP_Small_model_size_MB': encoder_stage_summary['small_model_megabytes'] if encoder_stage_summary else 'N/A',
        'MAKP_Total_communication_MB': encoder_stage_summary['total_encoder_stage_comm_megabytes'] if encoder_stage_summary else 'N/A',
        'MAKP_Avg_communication_per_global_round_MB': encoder_stage_summary['avg_encoder_stage_comm_per_global_round_megabytes'] if encoder_stage_summary else 'N/A',
        'MAKP_Total_runtime_s': encoder_stage_summary['total_encoder_stage_time_sec'] if encoder_stage_summary else 'N/A',
        'MAKP_Avg_runtime_per_global_round_s': encoder_stage_summary['avg_encoder_stage_time_per_global_round_sec'] if encoder_stage_summary else 'N/A',
        'DPCR_Classifier_size_MB': dpcr_summary['classifier_model_megabytes'],
        'DPCR_Total_one_shot_communication_MB': dpcr_summary['total_one_shot_comm_megabytes'],
        'DPCR_Avg_one_shot_communication_per_client_MB': dpcr_summary['avg_one_shot_total_megabytes_per_client'],
        'DPCR_Total_server_retraining_time_s': dpcr_summary['total_server_retrain_time_sec'],
        'DPCR_Evaluation_time_s': dpcr_summary['evaluation_time_sec'],
    }
    C.write_single_row_csv(os.path.join(args.outdir, 'table7_metrics.csv'), table7_metrics)

    overall_summary = {
        'dataset': 'DomainNet-subsetA',
        'eval_protocol': f'B: training domains only ({mean_key})',
        'train_domains': train_domains,
        'num_classes': num_classes,
        'num_clients': num_clients,
        'num_clusters': num_K,
        'client_counts': client_counts,
        'percent_per_client': percents,
        'cluster_source': clustering_summary['cluster_source'],
        'grouping': clustering_summary['grouping'],
        'group_seed': clustering_summary['group_seed'],
        'cluster_alpha': cluster_alpha,
        'beta_source': beta_source,
        'linkage': linkage,
        'clusters_used_for_training': clustering_summary['clusters_used_for_training'],
        'base_clusters_before_grouping': clustering_summary['base_clusters_before_grouping'],
        'clusters_used_domain_purity': clustering_summary['clusters_used_domain_purity'],
        'seed': args.seed,
        'data_seed': data_seed,
        'train_samples_per_domain': train_total,
        'test_samples_per_domain': test_sizes,
    }
    overall_summary.update({f'final_{d}_acc': a for d, a in zip(train_domains, accs)})
    overall_summary.update({
        'final_mean_acc': dpcr_summary['final_mean_acc'],
        'final_std_acc': dpcr_summary['final_std_acc'],
        'final_worst_domain_acc': dpcr_summary['final_worst_domain_acc'],
        'final_max_min_gap': dpcr_summary['final_max_min_gap'],
        'dpcr_total_one_shot_comm_megabytes': dpcr_summary['total_one_shot_comm_megabytes'],
        'dpcr_total_server_retrain_time_sec': dpcr_summary['total_server_retrain_time_sec'],
    })
    if encoder_stage_summary is not None:
        overall_summary.update({
            'total_encoder_stage_comm_megabytes': encoder_stage_summary['total_encoder_stage_comm_megabytes'],
            'total_encoder_stage_time_sec': encoder_stage_summary['total_encoder_stage_time_sec'],
            'avg_encoder_stage_time_per_global_round_sec':
                encoder_stage_summary['avg_encoder_stage_time_per_global_round_sec'],
        })
    with open(os.path.join(args.outdir, 'overall_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)
    # 归档位：既保留在 run 根目录（与既有 remote_ops 汇总脚本一致），也拷一份到 final/
    with open(os.path.join(final_dir, 'overall_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)
    C.write_single_row_csv(os.path.join(args.outdir, 'overall_summary.csv'),
                           {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                            for k, v in overall_summary.items()})

    final_payload = {
        'config': {k: (str(v) if isinstance(v, torch.device) else v) for k, v in vars(args).items()},
        'resolved_layout': {'train_domains': train_domains, 'client_counts': client_counts,
                            'percent_per_client': percents, 'num_clients': num_clients,
                            'source': layout_source},
        'resolved_beta': {'cluster_alpha': cluster_alpha, 'linkage': linkage, 'source': beta_source},
        'selected_domain_list': [str(d) for d in selected_domain_list],
        'round_stats': round_rows,
        'encoder_round_eval': (encoder_stage_summary or {}).get('encoder_round_eval', []),
        'final_eval': dpcr_summary,
    }
    with open(os.path.join(final_dir, 'final_report.json'), 'w', encoding='utf-8') as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 70)
    print(f'FINAL (protocol B, {len(train_domains)} training domains)')
    print(', '.join(f'{d}={a}' for d, a in zip(train_domains, accs)) +
          f' | {mean_key}={dpcr_summary["final_mean_acc"]} {std_key}={dpcr_summary["final_std_acc"]}')
    print('=' * 70)
    print('Saved files:')
    for name in ['svd_preprocess_stats.csv', 'adjacency_scores.csv', 'clustering_once_summary.json',
                 'encoder_stage_round_stats.csv', 'encoder_round_accs.json', 'encoder_stage_summary.json',
                 'dpcr_single_client_comm.csv', 'dpcr_summary.json',
                 'table6_metrics.csv', 'table7_metrics.csv', 'overall_summary.json']:
        print(f'  - {os.path.join(args.outdir, name)}')
    print(f'  - {os.path.join(final_dir, "final_report.json")} / overall_summary.json '
          f'/ encoder_all.pth / dpcr_classifier.pth / final_model_full.pth')


if __name__ == '__main__':
    main()
