configfile: "config.yaml"

from pathlib import Path

project_root = Path(config.get("project_root", ".")).resolve()
dataset = config["datasets"][0]
output_dir = Path(dataset["output"])
clinical_path = Path(dataset["clinical"])
expression_path = Path(dataset["expression"])
output_dir_str = str(output_dir)


rule all:
    input:
        f"{output_dir}/final/model_summary.csv",
        f"{output_dir}/final/model_comparison.png",


rule prepare:
    input:
        clinical=clinical_path,
        expression=expression_path,
    output:
        merged=f"{output_dir}/process/merged.csv",
        train=f"{output_dir}/process/train.csv",
        test=f"{output_dir}/process/test.csv",
    params:
        output_dir=output_dir_str,
        test_size=config.get("prepare", {}).get("test_size", 0.30),
        max_features=config.get("prepare", {}).get("max_features", 200),
    shell:
        """
        python scripts/prepare_split.py \
            --clinical "{input.clinical}" \
            --expression "{input.expression}" \
            --output "{params.output_dir}" \
            --test-size {params.test_size} \
            --max-features {params.max_features}
        """


rule cox:
    input:
        train=f"{output_dir}/process/train.csv",
        test=f"{output_dir}/process/test.csv",
    output:
        summary=f"{output_dir}/final/cox_summary.csv",
        hazard_ratios=f"{output_dir}/final/cox_hazard_ratios.csv",
        search=f"{output_dir}/final/cox_penalizer_search.csv",
        search_folds=f"{output_dir}/final/cox_penalizer_search_folds.csv",
        feature_frequency=f"{output_dir}/final/cox_feature_frequency.csv",
        test_scored=f"{output_dir}/final/cox_test_scored.csv",
    params:
        output_dir=output_dir_str,
        penalizers=config.get("cox", {}).get("penalizers", "0.01,0.05,0.1,0.5,1.0"),
    shell:
        """
        python scripts/train_cox.py \
            --input "{params.output_dir}/process" \
            --output "{params.output_dir}" \
            --penalizers "{params.penalizers}"
        """


rule lasso:
    input:
        train=f"{output_dir}/process/train.csv",
        test=f"{output_dir}/process/test.csv",
    output:
        summary=f"{output_dir}/final/lasso_summary.csv",
        hazard_ratios=f"{output_dir}/final/lasso_hazard_ratios.csv",
        selected_features=f"{output_dir}/final/lasso_selected_features.csv",
        search=f"{output_dir}/final/lasso_penalizer_search.csv",
        search_folds=f"{output_dir}/final/lasso_penalizer_search_folds.csv",
        feature_frequency=f"{output_dir}/final/lasso_feature_frequency.csv",
        test_scored=f"{output_dir}/final/lasso_test_scored.csv",
    params:
        output_dir=output_dir_str,
        penalizers=config.get("lasso", {}).get("penalizers", "0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1"),
        folds=config.get("lasso", {}).get("folds", 5),
        repeats=config.get("lasso", {}).get("repeats", 3),
        one_se=config.get("lasso", {}).get("one_se", True),
        random_state=config.get("lasso", {}).get("random_state", 42),
    shell:
        """
        python scripts/train_lasso.py \
            --input "{params.output_dir}/process" \
            --output "{params.output_dir}" \
            --penalizers "{params.penalizers}" \
            --folds {params.folds} \
            --repeats {params.repeats} \
            {('--no-one-se' if not params.one_se else '')} \
            --random-state {params.random_state}
        """


rule deepsurv:
    input:
        train=f"{output_dir}/process/train.csv",
        test=f"{output_dir}/process/test.csv",
    output:
        summary=f"{output_dir}/final/deepsurv_summary.csv",
        search=f"{output_dir}/final/deepsurv_search.csv",
        search_readable=f"{output_dir}/final/deepsurv_search_readable.csv",
        search_folds=f"{output_dir}/final/deepsurv_search_folds.csv",
        test_scored=f"{output_dir}/final/deepsurv_test_scored.csv",
    params:
        output_dir=output_dir_str,
        hidden_dim1_grid=config.get("deepsurv", {}).get("hidden_dim1_grid", "16,32,64"),
        hidden_dim2_grid=config.get("deepsurv", {}).get("hidden_dim2_grid", "8,16,32"),
        dropout_grid=config.get("deepsurv", {}).get("dropout_grid", "0.0,0.1,0.2"),
        learning_rate_grid=config.get("deepsurv", {}).get("learning_rate_grid", "0.0001,0.0003,0.001"),
        batch_size_grid=config.get("deepsurv", {}).get("batch_size_grid", "16,32"),
        epochs=config.get("deepsurv", {}).get("epochs", 80),
        patience=config.get("deepsurv", {}).get("patience", 8),
        validation_split=config.get("deepsurv", {}).get("validation_split", 0.1),
        random_state=config.get("deepsurv", {}).get("random_state", 42),
        batch_norm=config.get("deepsurv", {}).get("batch_norm", True),
        log_transform=config.get("deepsurv", {}).get("log_transform", True),
        max_features=config.get("deepsurv", {}).get("max_features", 120),
        min_expression_rate=config.get("deepsurv", {}).get("min_expression_rate", 0.05),
    shell:
        """
        python scripts/train_deepsurv.py \
            --input "{params.output_dir}/process" \
            --output "{params.output_dir}" \
            --hidden-dim1-grid "{params.hidden_dim1_grid}" \
            --hidden-dim2-grid "{params.hidden_dim2_grid}" \
            --dropout-grid "{params.dropout_grid}" \
            --learning-rate-grid "{params.learning_rate_grid}" \
            --batch-size-grid "{params.batch_size_grid}" \
            --epochs {params.epochs} \
            --patience {params.patience} \
            --validation-split {params.validation_split} \
            --random-state {params.random_state} \
            {('--no-batch-norm' if not params.batch_norm else '')} \
            {('--no-log-transform' if not params.log_transform else '')} \
            --max-features {params.max_features} \
            --min-expression-rate {params.min_expression_rate}
        """


rule summary:
    input:
        f"{output_dir}/final/cox_summary.csv",
        f"{output_dir}/final/lasso_summary.csv",
        f"{output_dir}/final/deepsurv_summary.csv",
    output:
        f"{output_dir}/final/model_summary.csv",
    params:
        output_dir=output_dir_str,
    shell:
        """
        python scripts/summary.py --output "{params.output_dir}"
        """


rule plot:
    input:
        f"{output_dir}/final/cox_test_scored.csv",
        f"{output_dir}/final/lasso_test_scored.csv",
        f"{output_dir}/final/deepsurv_test_scored.csv",
    output:
        f"{output_dir}/final/model_comparison.png",
    params:
        output_dir=output_dir_str,
    shell:
        """
        python scripts/plot_models.py --output-dir "{params.output_dir}"
        """
