# FKPL 迷你版 - 完整工作流实现

## 项目说明

本项目是FKPL（Federated Knowledge Prototyping Learning）的迷你版本，支持**PACS**和**Office10**两个多域数据集。

代码实现了**论文 FKPL 的完整工作流**（对应论文 Fig. 3 / Algorithm 1 的两阶段）：

- **Stage I — 专家训练与聚合（MAKP）**：特征子空间计算 → 自动聚类 → 簇级联邦学习 → 特征拼接
- **Stage II — 分类器校准（DPCR）**：逐簇域原型聚合 → 虚拟特征合成 → 服务器重训无偏分类器
- **个性化（Personalization）**：簇内训练域专属分类器（论文 Fig. 7）

代码中的阶段划分：
1. ✅ 数据准备与分配
2. ✅ 特征子空间计算（截断 SVD）
3. ✅ 自动聚类（层次聚类）
4. ✅ 模型初始化
5. ✅ 簇级联邦学习（Encoder 聚合）- 100轮
6. ✅ 特征拼接（Feature Splicing）合成全局提取器
7. ✅ DPCR 分类器校准（域原型 + 虚拟特征重训）- 200 epoch
8. ✅ 分类器个性化训练 - 100轮

## 文件结构

```
MINI_FKPL/
├── fkpl_PACS.py                    # PACS数据集实验脚本
├── fkpl_officecaltech.py           # Office10数据集实验脚本
├── hierarchical_clustering.py      # 聚类算法实现
├── model.py                         # 神经网络模型定义
├── tSNE.py                          # t-SNE可视化工具
├── data/                            # 数据目录（需要下载数据集）
│   ├── Homework3-PACS-master/
│   │   └── PACS/                   # PACS数据集目录
│   └── office_caltech_10/          # Office10数据集目录
└── README_CN.md                     # 本文件
```

## 环境要求

- Python 3.7+
- PyTorch 1.10+
- NumPy, Scikit-learn
- torchvision

## 数据集准备

### PACS 数据集

1. 下载 PACS 数据集：https://github.com/thuml/PACS
2. 解压到 `data/Homework3-PACS-master/PACS/`
3. 目录结构应为：
   ```
   data/Homework3-PACS-master/PACS/
   ├── art_painting/
   ├── cartoon/
   ├── photo/
   └── sketch/
   ```

### Office10 数据集

1. 下载 Office-31 数据集：https://www.cs.bu.edu/~saenko/vis/office31.tar.gz
2. 下载 Caltech-256 数据集
3. 解压到 `data/office_caltech_10/`
4. 目录结构应为：
   ```
   data/office_caltech_10/
   ├── amazon/
   ├── caltech/
   ├── webcam/
   └── dslr/
   ```

## 运行实验

### 运行 PACS 实验
```bash
python fkpl_PACS.py
```

### 运行 Office10 实验
```bash
python fkpl_officecaltech.py
```

## 实验流程说明

### 阶段 1-2：数据准备（自动）
- 加载数据集
- 创建虚拟客户端分布
- 为每个客户端分配本地数据

### 阶段 3：自动聚类
- 计算客户端相似度矩阵（基于主角相似性）
- 使用层次聚类得到簇分组
- 打印聚类结果

**关键参数：**
- PACS: `cluster_alpha = 4.2`
- Office10: `cluster_alpha = 5.3`

### 阶段 5：簇级联邦学习（约 2-4 小时）
- 100 轮通信
- 每轮：
  - 簇内客户端本地训练（10 个 epoch）
  - FedAvg 聚合
- **输出：** 每个簇训练好的专家 encoder 和 classifier

### 阶段 6：特征拼接（Feature Splicing）
- 将 V 个簇的专家 encoder 拼接为全局提取器 `E(Θ) = [F(θ1), …, F(θV)]`
- 输出特征维度 = 512 × V
- 冻结参数，广播给所有客户端

### 阶段 7：DPCR 分类器校准（约 0.5-1 小时，论文 Stage II）
- 客户端用冻结的全局提取器提取本地特征，计算本地类原型（论文 Eq. 5）
- 服务器按簇聚合得到**域原型**（论文 Eq. 6）
- 客户端把本地特征与各域原型做随机插值，合成虚拟特征（论文 Eq. 8，α ~ U(a,b)）
- 服务器用虚拟特征重训全局分类器 200 epoch（论文 Eq. 9）
- **输出：** 无偏的全局分类器，评估并打印各域准确率（论文 Table I 主结果）

### 阶段 8：分类器个性化训练（约 1-2 小时，论文 Fig. 7）
- 为每个簇训练个性化分类头
- 100 轮循环
- 每轮测试四个域的准确率

## 修改参数

### 簇数量实验（如 2、3、5 簇）

修改聚类阈值：
```python
# fkpl_PACS.py
cluster_alpha = 4.2  # 改为不同值

# fkpl_officecaltech.py
cluster_alpha = 5.3  # 改为不同值
```

**阈值越大 → 簇数越少**
**阈值越小 → 簇数越多**

### 客户端数量

修改客户端分布字典：
```python
# PACS: 默认 20 个客户端
selected_domain_dict = {'art_painting': 4, 'cartoon': 5, 'photo': 5, 'sketch': 6}

# Office10: 默认 10 个客户端
selected_domain_dict = {'caltech': 4, 'amazon': 3, 'webcam': 2, 'dslr': 1}
```

### 本地训练轮数

```python
local_epochs = 10  # 改为其他值
```

### DPCR 虚拟特征融合比例（α ~ U(a,b)）

```python
# fkpl_PACS.py（论文 a=0.3, b=0.7）
uniform_left = 0.30
uniform_right = 0.70

# fkpl_officecaltech.py（论文 a=0.2, b=0.5）
uniform_left = 0.20
uniform_right = 0.50
```

### DPCR 分类器重训轮数

```python
# training_global_classifier_with_mix_features 中的 n_epoch
n_epoch = 200
```

## 输出结果

### 聚类结果
```
Clusters: 
[[0, 5, 2, 17], [1, 6, 13, 9, 11, 19], [3, 7, 18, 12, 16], [4, 8, 14, 10, 15]]
Number of Clusters 4
```

### 全局模型精度（DPCR 阶段输出，论文 Table I 主结果）
```
[FKPL global model] mean test accuracy: 67.00.
[FKPL global model] individual test accuracy, art_painting: ..., cartoon: ..., photo: ..., sketch: ...
```

### 最终精度（个性化阶段输出，论文 Fig. 7）
```python
result = {
    0: [精度_art_painting, 精度_cartoon, 精度_photo, 精度_sketch],
    1: [...],
    2: [...],
    3: [...]
}
```

## 故障排查

### 显存不足
- 减少 `batch_size`（在 `partition_office_domain_skew_loaders` 中）
- 从 64 改为 32

### 数据加载失败
- 检查数据集路径
- 确保数据集目录结构正确

### CUDA 错误
- 改为 CPU 模式：
```python
device = torch.device('cpu')
```

## 相关论文

- FKPL: Federated Knowledge Prototyping Learning
- Reference: [原论文链接]

## 许可证

MIT License

## 联系方式

如有问题，请提交 Issue 或联系项目维护者。
