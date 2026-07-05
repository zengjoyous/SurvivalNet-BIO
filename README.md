# SurvivalNet

SurvivalNet 是一个面向 TCGA 生存分析的 Python 工具包，主要比较 Cox、LASSO-Cox 和 DeepSurv 三类模型。

SurvivalNet is a TCGA-oriented survival analysis package for comparing Cox, LASSO-Cox, and DeepSurv models on expression-based survival data.

## 中文

### 功能

- 读取 TCGA 临床表和表达矩阵
- 按病人级别合并数据
- 切分训练集和测试集
- 训练 Cox、LASSO-Cox、DeepSurv
- repeated CV 和 one-SE 规则选参
- 输出 summary、hazard ratio、风险分层图

### 安装

```bash
conda env create -f environment.yml
conda activate survivalnet
pip install -e .
```

### 运行

```bash
snakemake --cores 1
```

也可以分步运行：

```bash
snakemake --cores 1 prepare
snakemake --cores 1 cox
snakemake --cores 1 lasso
snakemake --cores 1 deepsurv
snakemake --cores 1 summary
snakemake --cores 1 plot
```

### 数据来源

默认示例使用 STAD 队列：

- `examples/GDC TCGA Stomach Cancer (STAD)/TCGA-STAD.clinical.tsv`
- `examples/GDC TCGA Stomach Cancer (STAD)/TCGA-STAD.star_tpm.tsv`

其他示例队列也在 `examples/` 目录下。

### 输出结果

输出保存在 `output/<数据集名>/`：

- `process/merged.csv`
- `process/train.csv`
- `process/test.csv`
- `final/*_summary.csv`
- `final/*_hazard_ratios.csv`
- `final/*_test_scored.csv`
- `final/model_summary.csv`
- `final/model_comparison.png`

### Notebook

见 `examples/demo.ipynb`，里面有完整演示流程。

### 测试

```bash
pytest
```

### 项目结构

```text
project_name/
├── pyproject.toml
├── README.md
├── Snakefile
├── Dockerfile
├── mkdocs.yml
├── docs/
├── src/
├── scripts/
├── tests/
├── examples/
└── environment.yml
```

### 文档站

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

### 备注

- 如果临床表中包含临床变量，主流程会自动合并到分析表中。
- DeepSurv 在当前仓库中使用了更小、更稳的搜索空间。
- 现在推荐使用 Snakemake 作为主工作流入口。
- 额外提供了 `Dockerfile` 和 `mkdocs.yml`，便于容器化和文档站搭建。

## English

### Features

- TCGA clinical/expression table loading
- patient-level merging and train/test splitting
- Cox, LASSO-Cox, and DeepSurv training
- repeated CV and one-SE selection
- summary tables, hazard-ratio tables, and risk-stratification plots

### Installation

```bash
conda env create -f environment.yml
conda activate survivalnet
pip install -e .
```

### Run

```bash
snakemake --cores 1
```

You can also run individual steps:

```bash
snakemake --cores 1 prepare
snakemake --cores 1 cox
snakemake --cores 1 lasso
snakemake --cores 1 deepsurv
snakemake --cores 1 summary
snakemake --cores 1 plot
```

### Data

The default example uses the STAD cohort:

- `examples/GDC TCGA Stomach Cancer (STAD)/TCGA-STAD.clinical.tsv`
- `examples/GDC TCGA Stomach Cancer (STAD)/TCGA-STAD.star_tpm.tsv`

Other example cohorts are available in `examples/`.

### Output

Outputs are written to `output/<dataset name>/`:

- `process/merged.csv`
- `process/train.csv`
- `process/test.csv`
- `final/*_summary.csv`
- `final/*_hazard_ratios.csv`
- `final/*_test_scored.csv`
- `final/model_summary.csv`
- `final/model_comparison.png`

### Notebook

See `examples/demo.ipynb` for an end-to-end demonstration.

### Testing

```bash
pytest
```

### Project Structure

```text
project_name/
├── pyproject.toml
├── README.md
├── Snakefile
├── Dockerfile
├── mkdocs.yml
├── docs/
├── src/
├── scripts/
├── tests/
├── examples/
└── environment.yml
```

### Documentation Site

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

### Notes

- Clinical columns can be merged into the analysis table when present.
- DeepSurv uses a smaller, more stable search space in this repository.
- Snakemake is the recommended workflow entry point.
- `Dockerfile` and `mkdocs.yml` are included for containerization and documentation.

## License

See [LICENSE](LICENSE).
