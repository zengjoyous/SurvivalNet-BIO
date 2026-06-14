PYTHON ?= python
EXAMPLES_DIR ?= examples/GDC TCGA Stomach Cancer (STAD)
OUTPUT_DIR ?= output/GDC TCGA Stomach Cancer (STAD)

CLINICAL ?= $(EXAMPLES_DIR)/TCGA-STAD.survival.tsv
EXPRESSION ?= $(EXAMPLES_DIR)/TCGA-STAD.star_tpm.tsv

.PHONY: prepare cox lasso deepsurv summary plot all clean

prepare:
	$(PYTHON) scripts/prepare_split.py --clinical "$(CLINICAL)" --expression "$(EXPRESSION)" --output "$(OUTPUT_DIR)"

cox:
	$(PYTHON) scripts/train_cox.py --input "$(OUTPUT_DIR)/process" --output "$(OUTPUT_DIR)"

lasso:
	$(PYTHON) scripts/train_lasso.py --input "$(OUTPUT_DIR)/process" --output "$(OUTPUT_DIR)"

deepsurv:
	$(PYTHON) scripts/train_deepsurv.py --input "$(OUTPUT_DIR)/process" --output "$(OUTPUT_DIR)"

summary:
	$(PYTHON) scripts/summary.py --output "$(OUTPUT_DIR)"

plot:
	$(PYTHON) scripts/plot_models.py --output-dir "$(OUTPUT_DIR)"

all: prepare cox lasso deepsurv summary plot

clean:
	rm -rf "$(OUTPUT_DIR)"
