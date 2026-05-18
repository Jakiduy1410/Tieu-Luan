import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from core.processor import apply_preprocessing


RANDOM_STATE = 42
TEXT_COLUMN = "Comment"
TARGET_COLUMN = "Label"
DEFAULT_PREPROCESSING = ["none", "basic", "advanced"]
DEFAULT_MODELS = ["naive_bayes", "linear_svm", "random_forest", "logistic_regression"]


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _serializable_params(params: dict) -> dict:
    serializable = {}
    for key, value in params.items():
        if key in {"tfidf", "classifier"}:
            continue
        try:
            json.dumps(value)
        except TypeError:
            continue
        else:
            serializable[key] = value
    return serializable


def _parse_csv_arg(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {TEXT_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")
    return df


def _prepare_xy(df: pd.DataFrame, preprocessing: str) -> tuple[pd.Series, pd.Series]:
    processed = apply_preprocessing(df, method=preprocessing, text_column=TEXT_COLUMN)
    x = processed[TEXT_COLUMN].fillna("").astype(str)
    y = processed[TARGET_COLUMN].astype(int)
    return x, y


def _base_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )


def _tfidf_grid(quick: bool) -> dict:
    if quick:
        return {
            "tfidf__max_features": [1000],
            "tfidf__ngram_range": [(1, 1)],
            "tfidf__min_df": [1],
            "tfidf__sublinear_tf": [True],
        }

    return {
        "tfidf__max_features": [3000, 5000, 10000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__sublinear_tf": [True],
    }


def _model_configs(quick: bool) -> dict:
    if quick:
        return {
            "naive_bayes": {
                "estimator": MultinomialNB(),
                "params": {"classifier__alpha": [1.0]},
            },
            "linear_svm": {
                "estimator": LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
                "params": {"classifier__C": [1.0]},
            },
            "random_forest": {
                "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                "params": {
                    "classifier__n_estimators": [50],
                    "classifier__max_depth": [50],
                    "classifier__min_samples_split": [2],
                    "classifier__class_weight": ["balanced"],
                },
            },
            "logistic_regression": {
                "estimator": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
                "params": {
                    "classifier__C": [1.0],
                    "classifier__class_weight": ["balanced"],
                    "classifier__solver": ["liblinear"],
                },
            },
        }

    return {
        "naive_bayes": {
            "estimator": MultinomialNB(),
            "params": {"classifier__alpha": [0.1, 0.5, 1.0, 2.0]},
        },
        "linear_svm": {
            "estimator": LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
            "params": {"classifier__C": [0.1, 1.0, 10.0]},
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 50],
                "classifier__min_samples_split": [2, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "logistic_regression": {
            "estimator": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
            "params": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
                "classifier__solver": ["liblinear"],
            },
        },
    }


def _build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", _base_tfidf()),
            ("classifier", clone(estimator)),
        ]
    )


def _evaluate_predictions(y_true: pd.Series, y_pred) -> dict:
    precision, recall, f1_spam, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        pos_label=1,
        average="binary",
        zero_division=0,
    )
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision_spam": precision,
        "test_recall_spam": recall,
        "test_f1_spam": f1_spam,
        "test_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "test_f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def _baseline_row(
    preprocessing: str,
    model_name: str,
    estimator,
    x_train: pd.Series,
    y_train: pd.Series,
    x_test: pd.Series,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> tuple[dict, Pipeline]:
    pipeline = _build_pipeline(estimator)
    cv_scores = cross_validate(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring={"f1_macro": "f1_macro"},
        n_jobs=-1,
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    metrics = _evaluate_predictions(y_test, y_pred)

    row = {
        "experiment_type": "baseline",
        "preprocessing": preprocessing,
        "model": model_name,
        "best_params": _json(_serializable_params(pipeline.get_params(deep=True))),
        "cv_f1_macro_mean": cv_scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": cv_scores["test_f1_macro"].std(),
        **metrics,
    }
    return row, pipeline


def _tuned_row(
    preprocessing: str,
    model_name: str,
    estimator,
    params: dict,
    x_train: pd.Series,
    y_train: pd.Series,
    x_test: pd.Series,
    y_test: pd.Series,
    cv: StratifiedKFold,
    quick: bool,
) -> tuple[dict, GridSearchCV]:
    pipeline = _build_pipeline(estimator)
    param_grid = {**_tfidf_grid(quick), **params}
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={"f1_macro": "f1_macro", "f1_spam": "f1"},
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        verbose=1,
    )
    search.fit(x_train, y_train)
    y_pred = search.best_estimator_.predict(x_test)
    metrics = _evaluate_predictions(y_test, y_pred)

    row = {
        "experiment_type": "tuned",
        "preprocessing": preprocessing,
        "model": model_name,
        "best_params": _json(search.best_params_),
        "cv_f1_macro_mean": search.cv_results_["mean_test_f1_macro"][search.best_index_],
        "cv_f1_macro_std": search.cv_results_["std_test_f1_macro"][search.best_index_],
        **metrics,
    }
    return row, search


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = [
        "preprocessing",
        "model",
        "best_params",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
        "test_accuracy",
        "test_precision_spam",
        "test_recall_spam",
        "test_f1_spam",
        "test_f1_macro",
        "test_f1_weighted",
    ]
    display = df[columns].copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}")

    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in display.to_numpy()
    ]
    path.write_text("\n".join([header, separator, *rows]) + "\n", encoding="utf-8")


def _plot_f1_macro(tuned_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=tuned_df,
        x="preprocessing",
        y="test_f1_macro",
        hue="model",
    )
    plt.ylim(0, 1)
    plt.title("Test F1 Macro by preprocessing and model")
    plt.xlabel("Preprocessing")
    plt.ylabel("Test F1 Macro")
    plt.legend(title="Model", loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _write_best_artifacts(
    output_dir: Path,
    tuned_df: pd.DataFrame,
    best_pipeline: Pipeline,
    x_test: pd.Series,
    y_test: pd.Series,
) -> None:
    best_row = tuned_df.sort_values("test_f1_macro", ascending=False).iloc[0]
    y_pred = best_pipeline.predict(x_test)

    confusion = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=[0, 1]),
        index=["actual_ham_0", "actual_spam_1"],
        columns=["predicted_ham_0", "predicted_spam_1"],
    )
    confusion.to_csv(output_dir / "best_confusion_matrix.csv", encoding="utf-8-sig")

    report = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T
    report.to_csv(output_dir / "best_classification_report.csv", encoding="utf-8-sig")

    joblib.dump(best_pipeline, output_dir / "best_pipeline.joblib")

    summary = [
        "# Best Tuned Model",
        "",
        f"- Preprocessing: {best_row['preprocessing']}",
        f"- Model: {best_row['model']}",
        f"- CV F1 Macro: {best_row['cv_f1_macro_mean']:.4f} +/- {best_row['cv_f1_macro_std']:.4f}",
        f"- Test F1 Macro: {best_row['test_f1_macro']:.4f}",
        f"- Test F1 Spam: {best_row['test_f1_spam']:.4f}",
        f"- Best Params: `{best_row['best_params']}`",
        "",
        "The independent test set is used only after hyperparameter selection.",
    ]
    (output_dir / "best_model_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def run_experiments(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    preprocessing_methods: Iterable[str],
    model_names: Iterable[str],
    cv_folds: int,
    quick: bool,
    sample_size: int | None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = _load_dataset(train_path)
    test_df = _load_dataset(test_path)

    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=RANDOM_STATE)
        test_df = test_df.sample(n=min(max(100, sample_size // 4), len(test_df)), random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    configs = _model_configs(quick)

    rows = []
    tuned_estimators: dict[tuple[str, str], Pipeline] = {}
    prepared_data: dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}

    for preprocessing in preprocessing_methods:
        print(f"\n=== Preprocessing: {preprocessing} ===")
        x_train, y_train = _prepare_xy(train_df, preprocessing)
        x_test, y_test = _prepare_xy(test_df, preprocessing)
        prepared_data[preprocessing] = (x_train, y_train, x_test, y_test)

        for model_name in model_names:
            if model_name not in configs:
                raise ValueError(f"Unsupported model '{model_name}'. Choose from: {sorted(configs)}")

            config = configs[model_name]
            print(f"\n--- Baseline: {model_name} ---")
            baseline, _ = _baseline_row(
                preprocessing,
                model_name,
                config["estimator"],
                x_train,
                y_train,
                x_test,
                y_test,
                cv,
            )
            rows.append(baseline)

            print(f"\n--- GridSearchCV: {model_name} ---")
            tuned, search = _tuned_row(
                preprocessing,
                model_name,
                config["estimator"],
                config["params"],
                x_train,
                y_train,
                x_test,
                y_test,
                cv,
                quick,
            )
            rows.append(tuned)
            tuned_estimators[(preprocessing, model_name)] = search.best_estimator_

    results = pd.DataFrame(rows)
    all_results_path = output_dir / "all_results.csv"
    baseline_path = output_dir / "baseline_results.csv"
    tuned_path = output_dir / "tuned_results.csv"

    results.to_csv(all_results_path, index=False, encoding="utf-8-sig")
    results[results["experiment_type"] == "baseline"].to_csv(
        baseline_path, index=False, encoding="utf-8-sig"
    )
    tuned_df = results[results["experiment_type"] == "tuned"].copy()
    tuned_df.to_csv(tuned_path, index=False, encoding="utf-8-sig")
    _write_markdown_table(tuned_df, output_dir / "tuned_results.md")
    _plot_f1_macro(tuned_df, output_dir / "f1_macro_comparison.png")

    best_row = tuned_df.sort_values("test_f1_macro", ascending=False).iloc[0]
    _, _, best_x_test, best_y_test = prepared_data[best_row["preprocessing"]]
    best_pipeline = tuned_estimators[(best_row["preprocessing"], best_row["model"])]
    _write_best_artifacts(output_dir, tuned_df, best_pipeline, best_x_test, best_y_test)

    print(f"\nSaved experiment artifacts to: {output_dir.resolve()}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible TF-IDF spam/ham experiments with baseline and GridSearchCV."
    )
    parser.add_argument("--train-path", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / datetime.now().strftime("research_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--preprocessing",
        default=",".join(DEFAULT_PREPROCESSING),
        help="Comma-separated values from: none,basic,advanced",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated values from: naive_bayes,linear_svm,random_forest,logistic_regression",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a tiny grid for smoke tests before running the full research grid.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional train sample size for local smoke tests. Omit for full data.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    preprocessing_methods = _parse_csv_arg(args.preprocessing, DEFAULT_PREPROCESSING)
    model_names = _parse_csv_arg(args.models, DEFAULT_MODELS)

    run_experiments(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        preprocessing_methods=preprocessing_methods,
        model_names=model_names,
        cv_folds=args.cv_folds,
        quick=args.quick,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
