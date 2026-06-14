# SurvivalNet

SurvivalNet 是一个面向 TCGA 生存分析的 Python 工具包，主要比较三类模型：

- Cox
- LASSO-Cox
- DeepSurv

项目采用 `src/` 布局，可通过 `pip install -e .` 安装。

## 功能

- 读取 TCGA 临床表和表达矩阵
- 按病人级别合并数据
- 切分训练集和测试集
- 训练 Cox、LASSO-Cox、DeepSurv
- repeated CV 和 one-SE 规则选参
- 输出 summary、hazard ratio、风险分层图

## 安装

```bash
conda env create -f environment.yml
conda activate survivalnet
pip install -e .
```

## 运行

```bash
make all
```

也可以分步运行：

```bash
make prepare
make cox
make lasso
make deepsurv
make summary
make plot
```

## 数据来源

默认示例使用 STAD 队列：

- `examples/GDC TCGA Stomach Cancer (STAD)/TCGA-STAD.clinical.tsv`
- `examples/GDC TCGA Stomach Cancer (STAD)/TCGA-STAD.star_tpm.tsv`

其他示例队列也在 `examples/` 目录下。

## 输出结果

输出保存在 `output/<数据集名>/`：

- `process/merged.csv`
- `process/train.csv`
- `process/test.csv`
- `final/*_summary.csv`
- `final/*_hazard_ratios.csv`
- `final/*_test_scored.csv`
- `final/model_summary.csv`
- `final/model_comparison.png`

## Notebook

见 `examples/demo.ipynb`，里面有完整演示流程。

## 测试

```bash
pytest
```

## 环境文件

请使用 `environment.yml` 保证可复现性。

## 项目结构

```text
project_name/
├── pyproject.toml
├── README.md
├── src/
├── tests/
├── examples/
└── environment.yml
```

## 备注

- 如果临床表中包含临床变量，主流程会自动合并到分析表中。
- DeepSurv 在当前仓库中使用了更小、更稳的搜索空间。

