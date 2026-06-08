# 气体传感器分类课程项目

这是一个面向课程设计与论文展示整理的子工程，围绕 UCI 数据集 **Gas Sensor Array Drift at Different Concentrations** 开展气体分类实验。仓库中保留了主线代码、关键结果、论文图表和 Overleaf 用的 `.tex` 文件，方便直接上传 GitHub 并继续扩展。

## 1. 项目说明

- 数据集：UCI `Gas Sensor Array Drift at Different Concentrations`
- 任务类型：6 类气体分类
- 特征维度：128
- 划分方式：固定 `train / val / test = 60% / 20% / 20%`
- 主评价指标：`Macro-F1`

## 2. 重要说明：数据文件不随仓库上传

由于 `data/` 目录中的原始数据和处理后数据文件体积较大，本仓库默认**不上传真实数据文件**。  
读者需要先从 UCI 官方链接自行下载原始数据，再按照本项目约定的文件名放入本地 `data/` 目录后运行。

截至 **2026-06-08**，可用的官方链接如下：

- UCI 官方数据集页面：  
  [https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations)
- 本项目推荐使用的单文件 CSV 直链：  
  [https://archive.ics.uci.edu/static/public/270/data.csv](https://archive.ics.uci.edu/static/public/270/data.csv)
- UCI 官方 ZIP 压缩包：  
  [https://archive.ics.uci.edu/static/public/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations.zip](https://archive.ics.uci.edu/static/public/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations.zip)

说明：

- 本项目脚本默认读取的是**单文件 CSV**，因此建议优先下载上面的 `data.csv`。
- 官方 ZIP 中通常是 `batch*.dat` 形式的原始批次文件，不是本项目直接使用的输入格式。
- 只要你下载的是 `data.csv`，就可以直接接入本项目的数据预处理脚本。

## 3. 数据放置方式

请先在项目根目录下准备如下路径结构：

```text
gas-sensor-classification-project/
├─ data/
│  └─ gas_sensor_array_drift_at_different_concentrations_uci_270.csv
```

也就是说，你下载完成后，需要把文件重命名为：

```text
gas_sensor_array_drift_at_different_concentrations_uci_270.csv
```

并放到：

```text
data/
```

如果你还想附带保存元数据，也可以额外保留：

```text
data/gas_sensor_array_drift_at_different_concentrations_uci_270.metadata.json
```

但要注意：**项目正常运行只强制依赖原始 CSV 文件，元数据不是必须的。**

## 4. 目录结构

```text
gas-sensor-classification-project/
├─ data/
│  ├─ README.md
│  └─ processed/
│     └─ gas_sensor_array_drift_at_different_concentrations_uci_270/
├─ figures/
├─ results/
│  ├─ gas_baselines/
│  ├─ deep_baselines/
│  ├─ final_model/
│  ├─ tsne_comparison_models/
│  ├─ overall_comparison.csv
│  └─ model_complexity.csv
├─ scripts/
│  ├─ prepare_gas_dataset.py
│  ├─ run_gas_baselines.py
│  ├─ run_deep_gas_models.py
│  ├─ run_eegnet_variant_suite.py
│  ├─ generate_publication_figures.py
│  ├─ generate_tsne_comparison_figure.py
│  └─ run_release_pipeline.py
├─ gas_paper_figures_overleaf.tex
├─ requirements.txt
└─ README.md
```

## 5. 环境安装

建议使用 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
```

如果你使用 GPU 环境，请根据本机 CUDA 版本安装合适的 `torch` 与 `torchvision`。

## 6. 从零开始运行

### 6.1 第一步：准备原始数据

下载 UCI 官方单文件 CSV：

[https://archive.ics.uci.edu/static/public/270/data.csv](https://archive.ics.uci.edu/static/public/270/data.csv)

下载后重命名为：

```text
gas_sensor_array_drift_at_different_concentrations_uci_270.csv
```

放入：

```text
data/
```

### 6.2 第二步：生成固定划分数据

```bash
python scripts/prepare_gas_dataset.py
```

运行完成后，会自动生成：

```text
data/processed/gas_sensor_array_drift_at_different_concentrations_uci_270/
├─ full_dataset_with_splits.csv
├─ train.csv
├─ val.csv
├─ test.csv
└─ split_summary.json
```

### 6.3 第三步：运行主线实验

一键运行全部主线流程：

```bash
python scripts/run_release_pipeline.py
```

如果你想分步骤执行，可以按下面顺序运行。

1. 传统机器学习基线

```bash
python scripts/run_gas_baselines.py
```

2. 深度学习基线

```bash
python scripts/run_deep_gas_models.py --output-dir results/deep_baselines
```

3. 最终模型对比

```bash
python scripts/run_eegnet_variant_suite.py --experiments baseline eca_wide_12_24 --output-dir results/final_model --seed 42
```

4. 生成论文图表

```bash
python scripts/generate_publication_figures.py
```

5. 生成二维 t-SNE 图

```bash
python scripts/generate_tsne_comparison_figure.py
```

## 7. 已整理好的结果

### 7.1 传统机器学习最佳结果

- 最佳传统模型：`Extra Trees`
- 测试集 `Macro-F1 = 0.6921`
- 测试集 `Accuracy = 0.6908`

### 7.2 深度学习基线最佳结果

- 最佳深度学习基线：`EEGNet-like`
- 测试集 `Macro-F1 = 0.7795`
- 测试集 `Accuracy = 0.7458`

### 7.3 最终模型结果

- 最终模型：`Refined EEGNet (12/24)`
- 测试集 `Macro-F1 = 0.7812`
- 测试集 `Accuracy = 0.7655`
- 测试集 `Balanced Accuracy = 0.8009`

## 8. 图表与论文文件

- 图表目录：`figures/`
- Overleaf 文件：`gas_paper_figures_overleaf.tex`
- 该 `.tex` 文件默认从 `figures/` 目录读取图片，适合直接上传到 Overleaf 工程

## 9. 仓库中保留了什么

本仓库保留的是：

- 实验代码
- 结果汇总文件
- 论文展示图表
- t-SNE 对比所需的小体积模型权重
- 可直接用于 Overleaf 的 `.tex` 文件

本仓库默认**不保留**：

- 原始数据文件
- 处理后的大体积划分文件

## 10. 推荐上传方式

建议将整个 `gas-sensor-classification-project` 文件夹作为独立仓库上传 GitHub。  
上传前只需确认不要把真实数据文件一并提交即可。
