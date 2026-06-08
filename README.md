# 气体传感器分类课程项目

这是一个可以直接上传 GitHub 的整理版子工程，围绕 UCI 数据集 **Gas Sensor Array Drift at Different Concentrations** 开展气体分类实验。项目已经整理出主线代码、数据集、关键结果、论文图表与 Overleaf 用的 `.tex` 文件，适合作为课程设计、开题展示或论文仓库基础版本。

## 1. 项目内容

- 数据集已内置：包含原始数据、元数据以及固定划分后的 `train / val / test`
- 传统机器学习代码：`Logistic Regression`、`kNN`、`SVM (RBF)`、`Random Forest`、`Extra Trees`
- 深度学习基线代码：`MLP`、`1D-CNN`、`LeNet1D`、`EEGNet-like`、`MobileNetV2`、`ShuffleNetV2`
- 最终模型代码：基于 `EEGNet` 的最终改进模型主线实验
- 图表生成代码：可直接生成论文展示用的雷达图、二维分面柱状图、三维精度-复杂度图和二维 t-SNE 图
- 论文素材：附带可在 Overleaf 中使用 `XeLaTeX` 编译的中文 `.tex` 文件

## 2. 目录结构

```text
gas-sensor-classification-project/
├─ data/
│  ├─ gas_sensor_array_drift_at_different_concentrations_uci_270.csv
│  ├─ gas_sensor_array_drift_at_different_concentrations_uci_270.metadata.json
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

## 3. 实验设置

- 数据集：`Gas Sensor Array Drift at Different Concentrations`
- 样本数：`13910`
- 特征维度：`128`
- 类别数：`6`
- 固定划分方式：
  - 每个类别内部保持原始样本顺序
  - 前 `60%` 作为训练集
  - 中间 `20%` 作为验证集
  - 最后 `20%` 作为测试集
- 主评价指标：`Macro-F1`

## 4. 环境安装

建议使用 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
```

如果你使用的是 GPU 环境，建议根据本机 CUDA 版本单独安装合适的 `torch` 与 `torchvision`。

## 5. 直接运行

### 5.1 运行全部主线流程

```bash
python scripts/run_release_pipeline.py
```

### 5.2 分步骤运行

1. 数据预处理

```bash
python scripts/prepare_gas_dataset.py
```

2. 传统机器学习基线

```bash
python scripts/run_gas_baselines.py
```

3. 深度学习基线

```bash
python scripts/run_deep_gas_models.py --output-dir results/deep_baselines
```

4. 最终模型对比

```bash
python scripts/run_eegnet_variant_suite.py --experiments baseline eca_wide_12_24 --output-dir results/final_model --seed 42
```

5. 生成论文图表

```bash
python scripts/generate_publication_figures.py
```

6. 生成二维 t-SNE 图

```bash
python scripts/generate_tsne_comparison_figure.py
```

## 6. 已整理好的结果

### 6.1 传统机器学习最佳结果

- 最佳传统模型：`Extra Trees`
- 测试集 `Macro-F1 = 0.6921`
- 测试集 `Accuracy = 0.6908`

### 6.2 深度学习基线最佳结果

- 最佳深度学习基线：`EEGNet-like`
- 测试集 `Macro-F1 = 0.7795`
- 测试集 `Accuracy = 0.7458`

### 6.3 最终模型结果

- 最终模型：`Refined EEGNet (12/24)`
- 测试集 `Macro-F1 = 0.7812`
- 测试集 `Accuracy = 0.7655`
- 测试集 `Balanced Accuracy = 0.8009`

## 7. 图表与论文文件

- 论文图表位于 `figures/`
- Overleaf 文件为根目录下的 `gas_paper_figures_overleaf.tex`
- 该 `.tex` 文件默认使用 `figures/` 目录中的图片，适合直接上传到 Overleaf 工程

## 8. 说明

- 仓库已经附带原始数据与固定划分后的数据，因此可以直接运行，不必再次下载
- `results/tsne_comparison_models/` 中附带了生成 t-SNE 图所需的对比模型权重，便于快速复现实验图
- 若使用 CPU 重新训练深度模型，耗时会明显更长，建议优先使用 GPU 环境

## 9. 推荐上传方式

建议将整个 `gas-sensor-classification-project` 文件夹作为独立仓库上传 GitHub。  
如果你后面还想继续补充摘要、参考文献、实验分析正文或者英文版 README，也可以在这个子工程基础上继续扩展。
