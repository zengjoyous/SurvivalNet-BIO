# SurvivalNet

SurvivalNet is a TCGA-oriented survival analysis package for comparing Cox, LASSO-Cox, and DeepSurv models on expression-based survival data.

## Documentation

- [English README](README_EN.md)
- [中文 README](README_CN.md)

## Quick Start

```bash
conda env create -f environment.yml
conda activate survivalnet
pip install -e .
make all
```

## Project Layout

- `src/` package source code
- `scripts/` command-line entry points
- `examples/` example TCGA data and demo notebook
- `tests/` pytest test suite
- `environment.yml` reproducible environment

## License

See [LICENSE](LICENSE).
