import numpy as np
import torch
import copy
from collections import Counter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from hierarchical_clustering import *
from model import *
from tqdm import tqdm


RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
DOMAIN_LIST = ['caltech', 'amazon', 'webcam', 'dslr']

def record_net_data_stats(y_train, net_dataidx: list):
    unq, unq_cnt = np.unique(y_train[net_dataidx], return_counts=True)
    net_cls_counts = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    return net_cls_counts


class ImageFolder_Custom(datasets.DatasetFolder):
    def __init__(self, data_name, root, train, transform, subset_train_num=7, subset_capacity=10):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        if train:
            self.imagefolder_obj = datasets.ImageFolder(root=self.root + self.data_name, transform=self.transform)
        else:
            self.imagefolder_obj = datasets.ImageFolder(root=self.root + self.data_name, transform=self.transform)

        all_data = self.imagefolder_obj.samples
        self.train_index_list = []
        self.test_index_list = []
        for i in range(len(all_data)):
            if i % subset_capacity <= subset_train_num:
                self.train_index_list.append(i)
            else:
                self.test_index_list.append(i)

    def __len__(self):
        if self.train:
            return len(self.train_index_list)
        else:
            return len(self.test_index_list)

    def __getitem__(self, index):
        if self.train:
            used_index_list = self.train_index_list
        else:
            used_index_list = self.test_index_list

        path = self.imagefolder_obj.samples[used_index_list[index]][0]
        target = self.imagefolder_obj.samples[used_index_list[index]][1]
        target = int(target)
        img = self.imagefolder_obj.loader(path)
        img = self.transform(img)
        return img, target


def get_dataset_list(selected_domain_list):
    using_list = selected_domain_list
    Nor_TRANSFORM = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    train_dataset_list, test_dataset_list = [], []
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    for domain in using_list:
        train_dataset = ImageFolder_Custom(data_name=domain, root='./data/office_caltech_10/', train=True,
                                           transform=Nor_TRANSFORM)
        train_dataset_list.append(train_dataset)

    for domain in DOMAIN_LIST:
        test_dataset = ImageFolder_Custom(data_name=domain, root='./data/office_caltech_10/', train=False,
                                          transform=test_transform)
        test_dataset_list.append(test_dataset)

    return train_dataset_list, test_dataset_list


def partition_office_domain_skew_loaders(train_datasets: list, test_datasets: list, selected_domain_list: list,
                                         percent_dict=None):
    if percent_dict is None:
        percent_dict = {'caltech': 0.25, 'amazon': 0.25, 'webcam': 0.50, 'dslr': 1.0}
    init_len_dict = {}
    not_used_index_dict = {}
    net_dataidx_map = {}
    traindata_cls_counts = {}
    train_loaders, test_loaders = [], []

    for i in range(len(train_datasets)):
        domain_name = train_datasets[i].data_name
        if domain_name not in not_used_index_dict:
            all_train_index = np.array(train_datasets[i].train_index_list)
            not_used_index_dict[domain_name] = np.arange(len(all_train_index))
            init_len_dict[domain_name] = len(all_train_index)

    # 测试集的dataloader直接可以获得
    for i in range(len(test_datasets)):
        test_dataset = test_datasets[i]
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        test_loaders.append(test_loader)

    for i in range(len(train_datasets)):
        domain_name = train_datasets[i].data_name
        train_dataset = train_datasets[i]
        idxs = np.random.permutation(not_used_index_dict[domain_name])
        percent = percent_dict[domain_name]

        selected_idxs = idxs[0: int(percent * init_len_dict[domain_name])]
        not_used_index_dict[domain_name] = idxs[int(percent * init_len_dict[domain_name]):]

        train_sampler = SubsetRandomSampler(selected_idxs)
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
        train_loaders.append(train_loader)

        y_train = []
        for index in range(len(train_dataset.train_index_list)):
            y_train.append(int(train_dataset.imagefolder_obj.samples[train_dataset.train_index_list[index]][1]))
        traindata_cls_counts[i] = record_net_data_stats(np.array(y_train), selected_idxs)
        net_dataidx_map[i] = selected_idxs

    return traindata_cls_counts, net_dataidx_map, train_loaders, test_loaders


num_clients = 10
# selected_domain_dict = {'caltech': 3, 'amazon': 2, 'webcam': 1, 'dslr': 4}
# selected_domain_dict = {'caltech': 3, 'amazon': 2, 'webcam': 3, 'dslr': 2}
selected_domain_dict = {'caltech': 4, 'amazon': 3, 'webcam': 2, 'dslr': 1}
selected_domain_list = []

for i in selected_domain_dict:
    domain_num = selected_domain_dict[i]
    for j in range(domain_num):
        selected_domain_list.append(i)

selected_domain_list = np.random.permutation(selected_domain_list)
print(selected_domain_list)
result = Counter(selected_domain_list)
print(result)

train_dataset_list, test_dataset_list = get_dataset_list(selected_domain_list)
traindata_cls_counts, net_dataidx_map, train_dls, test_dls = partition_office_domain_skew_loaders(train_dataset_list,
                                                                                                  test_dataset_list,
                                                                                                  selected_domain_list)

traindata_cls_ratio = {}
budget = 10
for i in range(num_clients):
    total_sum = sum(list(traindata_cls_counts[i].values()))  # 该客户端的本地样本总数
    base = 1 / len(list(traindata_cls_counts[i].values()))  # 拥有的样本类别数分之1，可以认为是理想的比例

    temp_ratio = {}
    for k in traindata_cls_counts[i].keys():
        ss = traindata_cls_counts[i][k] / total_sum
        temp_ratio[k] = traindata_cls_counts[i][k] / total_sum
        if ss >= (base):  # 为什么这里加上0.05可能会凑不够20个主向量？
            temp_ratio[k] = traindata_cls_counts[i][k]
    sub_sum = sum(list(temp_ratio.values()))

    for k in temp_ratio.keys():
        temp_ratio[k] = (temp_ratio[k] / sub_sum) * budget

    round_ratio = round_to(list(temp_ratio.values()), budget)
    cnt = 0
    for k in temp_ratio.keys():
        temp_ratio[k] = round_ratio[cnt]
        cnt += 1

    traindata_cls_ratio[i] = temp_ratio

U_clients = []
n_basis = 5
cluster_alpha = 5.3  # the clustering threshold
linkage = 'average'  # Type of Linkage for HC

for idx in range(num_clients):
    dataidxs = net_dataidx_map[idx]
    sample_list = []
    label_list = []
    train_data = train_dataset_list[idx]

    for dataidx in dataidxs:
        sample_list.append(train_data[dataidx][0])
        label_list.append(int(train_data[dataidx][1]))

    train_data = torch.stack(sample_list, dim=0).numpy()
    labels = np.array(label_list)
    idxs_local = np.arange(len(sample_list))
    labels_local = np.array(label_list)
    # Sort Labels Train
    idxs_labels_local = np.vstack((idxs_local, labels_local))
    idxs_labels_local = idxs_labels_local[:, idxs_labels_local[1, :].argsort()]
    idxs_local = idxs_labels_local[0, :]
    labels_local = idxs_labels_local[1, :]

    uni_labels, cnt_labels = np.unique(labels_local, return_counts=True)
    print(f'Labels: {uni_labels}, Counts: {cnt_labels}')

    nlabels = len(uni_labels)
    cnt = 0
    U_temp = []

    for j in range(nlabels):
        local_ds1 = train_data[idxs_local[cnt:cnt + cnt_labels[j]]]
        local_ds1 = local_ds1.reshape(cnt_labels[j], -1)
        local_ds1 = local_ds1.T

        label1 = list(set(labels[idxs_local[cnt:cnt + cnt_labels[j]]]))
        if label1 in list(traindata_cls_ratio[idx].keys()):
            K = traindata_cls_ratio[idx][label1[0]]
        else:
            K = n_basis

        if K > 0:
            u1_temp, sh1_temp, vh1_temp = np.linalg.svd(local_ds1, full_matrices=False)
            u1_temp = u1_temp / np.linalg.norm(u1_temp, ord=2, axis=0)
            U_temp.append(u1_temp[:, 0:K])

        cnt += cnt_labels[j]

    U_clients.append(copy.deepcopy(np.hstack(U_temp)))

print(f'Client {idx}, Shape of U: {U_clients[-1].shape}, domain: {selected_domain_list[idx]}')

# ======================== 阶段3: 自动聚类得到簇 ========================
for r in range(1):
    print(f'Round {r}')
    clients_idxs = np.arange(num_clients)
    for idx in clients_idxs:
        print(f'Client {idx}, Labels: {traindata_cls_counts[idx]}')

    adj_mat = calculating_adjacency(clients_idxs, U_clients)
    clusters = hierarchical_clustering(copy.deepcopy(adj_mat), thresh=cluster_alpha, linkage=linkage)

    print('')
    print('Clusters: ')
    print(clusters)
    print('')
    print(f'Number of Clusters {len(clusters)}')
    print('')
    for jj in range(len(clusters)):
        print(f'Cluster {jj}: {len(clusters[jj])} Users')

clients_clust_id = {i: None for i in range(num_clients)}
for i in range(num_clients):
    for j in range(len(clusters)):
        if i in clusters[j]:
            clients_clust_id[i] = j
            break
print(f'Clients: Cluster_ID \n{clients_clust_id}')

encoder_list = []
classifier_list = []
num_ftrs = 512
for i in range(num_clients):
    encoder = resnet10Encoder(nclasses=10)
    encoder_list.append(encoder)
    classifier = SimpleClassifier(hidden_dim=num_ftrs, output_dim=10)
    classifier_list.append(classifier)

num_K = len(clusters)
encoder_global_para = [encoder_list[0].state_dict() for i in range(num_K)]
classifier_global_para = [classifier_list[0].state_dict() for i in range(num_K)]

encoder_round = 100
sample_ratio = 1.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_epochs = 10 # basic setting
# local_epochs = 20 # case study

local_fusion_protos = {}
# 正常长度的protos
local_normal_protos = {}
domain_normal_protos = {}


def agg_features(features):
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


def proto_aggregation(selected, local_normal_protos_dict):
    agg_protos_label = dict()
    for idx in selected:
        local_protos = local_normal_protos_dict[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label].cpu().numpy())
            else:
                agg_protos_label[label] = [local_protos[label].cpu().numpy()]
    for label, protos_list in agg_protos_label.items():
        agg_selected_proto = np.mean(protos_list, axis=0, keepdims=True)
        agg_protos_label[label] = torch.tensor(agg_selected_proto)
    return agg_protos_label


def train_net_encoder_classifier(net_id, net_encoder, net_classifier, train_dl, epochs, lr,
                                 optimizer_name, device):
    optimizer = torch.optim.SGD([{'params': net_encoder.parameters()}, {'params': net_classifier.parameters()}],
                                lr=lr, momentum=0.9, weight_decay=1e-5)
    classification_loss = nn.CrossEntropyLoss().to(device)

    local_class_features = {}

    if type(train_dl) == type([1]):
        pass
    else:
        train_dl = [train_dl]
    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dl:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                encoded = net_encoder(x)
                scores = net_classifier(encoded)
                loss = classification_loss(scores, target)
                loss.backward()
                optimizer.step()
                epoch_loss_collector.append(loss.item())
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        print(f'Epoch: {epoch}, loss: {epoch_loss}')
    protos = agg_features(local_class_features)
    local_normal_protos[net_id] = protos
    print(f'Network {net_id} training completed')


def local_train_net_encoder_classifier(encoder_list, classifier_list, selected, net_dataidx_map,
                                       domain_list, test_dl=None, device='cpu'):
    for net_id in range(len(encoder_list)):
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]
        print(f'Training network {net_id}, domain: {domain_list[net_id]}, number of training samples {len(dataidxs)}.')
        encoder_list[net_id].to(device)
        classifier_list[net_id].to(device)
        train_net_encoder_classifier(net_id, encoder_list[net_id], classifier_list[net_id],
                                     train_dls[net_id], local_epochs, lr=0.01, optimizer_name=None, device=device)
        encoder_list[net_id].to('cpu')
        classifier_list[net_id].to('cpu')

        # 计算新的domain prototypes
    individual_domain_protos = proto_aggregation(selected, local_normal_protos)
    return individual_domain_protos

# ======================== 阶段5: 簇级联邦学习 (Encoder 聚合) ========================
print('\n' + '='*60)
print('Starting Cluster-wise Federated Learning for Encoder')
print('='*60 + '\n')

for r in range(encoder_round):
    print(f'\n======FedAvg in clusters. In comm round {r}:======')
    top = int(num_clients * sample_ratio)
    participation = np.random.permutation(np.arange(num_clients))[:top]

    individual_domain_protos = {}

    # 每个cluster内部使用FedAvg
    for i in range(num_K):
        print(f'------FedAvg in cluster {i}. In comm round {r}:------')
        selected = []
        for j in clusters[i]:
            if j in participation:
                selected.append(j)
        if len(selected) == 0:
            print(f'No selected clients in cluster {i}, skip.')
            continue
        
        print(f'Selected clients in cluster {i}: {selected}')
        
        for idx in selected:  # 初始化本地模型
            encoder_list[idx].load_state_dict(encoder_global_para[i])
            classifier_list[idx].load_state_dict(classifier_global_para[i])

        individual_domain_protos[i] = local_train_net_encoder_classifier(encoder_list, classifier_list, selected,
                                                                         net_dataidx_map,
                                                                        selected_domain_list, device=device)

        total_data_samples = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_samples for r in selected]

        for idx in range(len(selected)):
            net_para = encoder_list[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    encoder_global_para[i][key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    encoder_global_para[i][key] += net_para[key] * fed_avg_freqs[idx]

        for idx in range(len(selected)):
            net_para = classifier_list[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    classifier_global_para[i][key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    classifier_global_para[i][key] += net_para[key] * fed_avg_freqs[idx]

    temp_domain_protos = {}
    for cluster_id, content in individual_domain_protos.items():
        for label, proto in content.items():
            if label in temp_domain_protos:
                temp_domain_protos[label].append(proto)
            else:
                temp_domain_protos[label] = [proto]

    domain_normal_protos = temp_domain_protos
    
    if r % 10 == 0:
        print(f'Completed encoder training round {r}')

print('\nEncoder training completed!')

encoder_selected = []
for i in range(num_K):
    encoder_selected.append(encoder_list[i])
    encoder_selected[i].load_state_dict(encoder_global_para[i])
    encoder_selected[i].to(device)
classifier_selected = []
for i in range(num_K):
    classifier_selected.append(classifier_list[i])
    classifier_selected[i].load_state_dict(classifier_global_para[i])
    classifier_selected[i].to(device)

# ======================== 定义encoder_all ========================
# 将所有簇的encoder特征拼接
class CombineAllModel(torch.nn.Module):
    def __init__(self, encoder_list):
        super(CombineAllModel, self).__init__()
        self.encoder_list = torch.nn.ModuleList(encoder_list)
    
    def forward(self, x):
        outputs = []
        for encoder in self.encoder_list:
            outputs.append(encoder(x))
        return torch.cat(outputs, dim=1)

encoder_all = CombineAllModel(encoder_selected)

# 如果想加载预训练的检查点而不是使用训练好的模型，取消注释以下行
# encoder_all = torch.load(
#     'encoder_10clients_officecaltech_4321_wo_proto_100r.pth',
#     map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# )
# torch.save(encoder_all, 'encoder_10clients_officecaltech_4321_wo_proto_100r.pth')

# ======================== 定义CombineModel ========================
class CombineModel(torch.nn.Module):
    def __init__(self, encoder, classifier):
        super(CombineModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
    
    def forward(self, x):
        encoded = self.encoder(x)
        output = self.classifier(encoded)
        return output
output_dim = 10
larger_classifier_global = SimpleClassifier(hidden_dim=num_ftrs * num_K, output_dim=output_dim)
unbiased_classifier_global = copy.deepcopy(larger_classifier_global)
start_from_classifier_weight = False

larger_classifier_global_para = larger_classifier_global.cpu().state_dict()
larger_classifier_list = []
for _ in range(num_clients):
    larger_classifier_list.append(SimpleClassifier(hidden_dim=num_ftrs * num_K, output_dim=output_dim))


def train_net_classifier(net_id, net_encoder, net_classifier, train_dl, test_dls, epochs, lr,
                         optimizer_name, device):
    optimizer = torch.optim.SGD([{'params': net_classifier.parameters()}], lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    if type(train_dl) == type([1]):
        pass
    else:
        train_dl = [train_dl]
    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dl:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                out = net_classifier(net_encoder(x))
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())
                # if batch_idx == 2:
                #     break  # only train 3 mini-batches
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        print(f'Epoch: {epoch}, loss: {epoch_loss}')
    print(f'Network {net_id} training completed')


def get_fusion_features(encoder, train_dl, device):
    local_class_fusion_features = {}
    status = encoder.training
    encoder.eval()
    for input, target in train_dl:
        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            fusion_features = encoder(input)
            for i in range(len(target)):
                if target[i].item() in local_class_fusion_features:
                    local_class_fusion_features[target[i].item()].append(fusion_features[i, :])
                else:
                    local_class_fusion_features[target[i].item()] = [fusion_features[i, :]]
    encoder.train(status)
    return local_class_fusion_features


def get_fusion_protos(encoder, train_dls, device):
    # 这里让所有客户端都在本地计算了自己数据集对应的prototypes
    total_fusion_protos = {}
    for net_id in range(num_clients):
        local_class_fusion_features = get_fusion_features(encoder, train_dls[net_id], device)
        local_fusion_protos = agg_features(local_class_fusion_features)
        total_fusion_protos[net_id] = local_fusion_protos
    return total_fusion_protos


def local_train_net_classifier(encoder, classifier_list, selected, net_dataidx_map, domain_list, test_dls=None, device='cpu'):
    for net_id in range(len(classifier_list)):
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]
        print(f'Training network {net_id}, domain: {domain_list[net_id]}, number of training samples {len(dataidxs)}.')
        encoder.to(device)
        classifier_list[net_id].to(device)
        n_epoch = 10
        # n_epoch = 1
        train_net_classifier(net_id, encoder, classifier_list[net_id],
                             train_dls[net_id], test_dls, n_epoch, lr=0.01, optimizer_name=None, device=device)

# 报告的准确率所用的超参数
uniform_left = 0.20
uniform_right = 0.50

# hyperparam study
# uniform_left = 0.4
# uniform_right = 0.9


class MixupDataset(Dataset):
    """Domain-balanced virtual feature set for DPCR (paper Eq. (8)).

    Each local feature is fused with every per-cluster (domain) prototype of
    the same class via random interpolation alpha ~ U(a, b).
    """
    def __init__(self, domain_protos, client_local_features, num_classes, device):
        self.data = []
        self.labels = []
        self.domain_protos = domain_protos  # dict[cls] -> list of prototype tensors (D,)
        self.num_classes = num_classes
        self.device = device
        self.all_features = client_local_features  # dict[net_id] -> dict[cls] -> list of features (D,)
        self.mixup_features_with_domain_protos()

    def mixup_features_with_domain_protos(self):
        l = uniform_left
        r_reg = uniform_right - l
        for cls in range(self.num_classes):
            num = 0
            protos = self.domain_protos.get(cls, [])
            for net_id in range(num_clients):
                for feature in self.all_features[net_id].get(cls, []):
                    for domain_proto in protos:
                        lam = float(np.round(l + r_reg * np.random.random(), 2))
                        domain_proto = domain_proto.to(self.device)
                        mixup_feature = (1 - lam) * feature + lam * domain_proto
                        self.data.append(mixup_feature)
                        num += 1
            self.labels += [cls] * num
        if len(self.data) == 0:
            raise RuntimeError('MixupDataset is empty: no virtual features were synthesized.')
        self.data = torch.stack(self.data).to(self.device)
        self.labels = torch.tensor(self.labels).long().to(self.device)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]


def training_global_classifier_with_mix_features(encoder, global_classifier, device):
    """DPCR (paper Stage II): retrain the global classifier on domain-balanced
    virtual features. Implements Eq. (5)-(9)."""
    print('=====DPCR: Training unbiased global classifier with domain-prototype mix-up features=====')

    # 1) Freeze the global spliced extractor E(Theta)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # 2) Extract local features (Eq. 7) and local prototypes (Eq. 5)
    client_local_class_fusion_features = {}
    client_local_fusion_protos = {}
    for net_id in range(num_clients):
        client_local_class_fusion_features[net_id] = get_fusion_features(encoder, train_dls[net_id], device)
        client_local_fusion_protos[net_id] = agg_features(client_local_class_fusion_features[net_id])

    # 3) Aggregate per-cluster domain prototypes (Eq. 6)
    domain_protos = {}
    for cluster in clusters:
        cluster_protos = proto_aggregation(cluster, client_local_fusion_protos)  # {cls: (1, D)}
        for cls, proto in cluster_protos.items():
            proto = proto.squeeze(0)  # (D,)
            if cls in domain_protos:
                domain_protos[cls].append(proto)
            else:
                domain_protos[cls] = [proto]

    # 4) Synthesize virtual features (Eq. 8) and retrain the classifier (Eq. 9)
    mixup_dataset = MixupDataset(domain_protos, client_local_class_fusion_features,
                                 num_classes=output_dim, device=device)
    mixup_dataloader = DataLoader(mixup_dataset, batch_size=64, shuffle=True)
    n_epoch = 200
    iterator = tqdm(range(n_epoch))
    optimizer = torch.optim.SGD(global_classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    global_classifier.to(device)

    for epoch in iterator:
        for features, labels in mixup_dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = global_classifier(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def global_evaluate(model, test_dls, device):
    status = model.training
    model.eval()
    accs = []
    for _, test_dl in enumerate(test_dls):
        total, top1, top5 = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(test_dl):
            with torch.no_grad():
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        top5acc = round(100 * top5 / total, 2)
        accs.append(top1acc)
    model.train(status)
    return accs


# 训练分类器，不考虑不cluster
classifier_round = 100
accs_dict = {}
mean_accs_list = []

# ======================== Stage II: DPCR 分类器校准（论文核心） ========================
print('\n' + '=' * 60)
print('Starting DPCR: Domain Prototype-based Classifier Retraining (Paper Stage II)')
print('=' * 60 + '\n')

# 用域原型融合的虚拟特征重训全局分类器（论文 Eq. 5-9）
training_global_classifier_with_mix_features(encoder_all, unbiased_classifier_global, device)

# 评估论文的全局模型：拼接提取器 + DPCR 重训的无偏分类器
unbiased_global_model = CombineModel(encoder_all, unbiased_classifier_global)
unbiased_global_model.to(device)
unbiased_global_model.eval()
accs = global_evaluate(unbiased_global_model, test_dls, device)
mean_acc = round(float(np.mean(accs)), 2)
print(f'[FKPL global model] mean test accuracy: {mean_acc}.')
print(f'[FKPL global model] individual test accuracy, '
      f'caltech: {accs[0]}, amazon: {accs[1]}, webcam: {accs[2]}, dslr: {accs[3]}.')



# for r in range(classifier_round):
#     print(f'In comm round: {r}')
#     arr = np.arange(num_clients)
#     np.random.shuffle(arr)
#     selected = arr[:int(num_clients * sample_ratio)]
#
#     for idx in selected:
#         larger_classifier_list[idx].load_state_dict(larger_classifier_global_para)
#     local_train_net_classifier(encoder_all, larger_classifier_list, selected, net_dataidx_map, device=device)
#     fed_avg_freqs = [1 / len(selected) for _ in selected]
#
#     # total_data_samples = sum([len(net_dataidx_map[r]) for r in selected])
#     # fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_samples for r in selected]
#
#     for idx in range(len(selected)):
#         net_para = larger_classifier_list[selected[idx]].cpu().state_dict()
#         if idx == 0:
#             for key in net_para:
#                 larger_classifier_global_para[key] = net_para[key] * fed_avg_freqs[idx]
#         else:
#             for key in net_para:
#                 larger_classifier_global_para[key] += net_para[key] * fed_avg_freqs[idx]
#
#     larger_classifier_global.load_state_dict(larger_classifier_global_para)
#     global_model = CombineModel(encoder_all, larger_classifier_global)
#     global_model.to(device)
#     accs = global_evaluate(global_model, test_dls, device)
#     mean_acc = round(np.mean(accs, axis=0), 3)
#     mean_accs_list.append(mean_acc)
#     for i in range(len(accs)):
#         if DOMAIN_LIST[i] in accs_dict:
#             accs_dict[DOMAIN_LIST[i]].append(accs[i])
#         else:
#             accs_dict[DOMAIN_LIST[i]] = [accs[i]]
#     print(f'global model mean test accuracy in round {r}: {mean_acc}.')
#     print(f'unbiased global model individual test accuracy, caltech: {accs[0]}, amazon: {accs[1]}, webcam: {accs[2]}, dslr: {accs[3]}.')

# 训练4个个性化分类头
result = {}
personalized_larger_classifier = [copy.deepcopy(larger_classifier_global) for i in range(num_K)]
for i in range(num_K):
    for r in range(classifier_round):
        print(f'=====Training personalized classifier in cluster {i}. In comm round: {r}=====')
        selected = clusters[i]
        personalized_larger_classifier_para = personalized_larger_classifier[i].state_dict()
        # 簇内本地模型更新为簇内全局模型
        for idx in selected:
            larger_classifier_list[idx].load_state_dict(personalized_larger_classifier_para)
        local_train_net_classifier(encoder_all, larger_classifier_list, selected, net_dataidx_map, selected_domain_list,
                                   test_dls=test_dls, device=device)

        total_data_samples = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_samples for r in selected]
        # 簇内分类器聚合
        for idx in range(len(selected)):
            net_para = larger_classifier_list[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    personalized_larger_classifier_para[key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    personalized_larger_classifier_para[key] += net_para[key] * fed_avg_freqs[idx]

        personalized_larger_classifier[i].load_state_dict(personalized_larger_classifier_para)
        personalized_global_model = CombineModel(encoder_all, personalized_larger_classifier[i])
        personalized_global_model.to(device)
        accs = global_evaluate(personalized_global_model, test_dls, device)


        print(f'unbiased global model individual test accuracy, caltech: {accs[0]}, amazon: {accs[1]}, webcam: {accs[2]}, dslr: {accs[3]}.')
        if r == classifier_round - 1:
            result[i] = accs

print(result)
