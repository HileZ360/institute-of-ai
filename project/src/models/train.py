"""Обучение и сравнение моделей для задачи прогнозирования оттока."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data.loader import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    dataset_quality_summary,
    display_path,
    find_dataset,
    load_churn_data,
    split_features_target,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"sklearn\.(linear_model\._linear_loss|utils\.extmath)",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"

SCORING = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def make_preprocessor() -> ColumnTransformer:
    """Создаёт preprocessing для числовых и категориальных признаков."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_models() -> dict[str, object]:
    """Возвращает фиксированный набор sklearn-моделей для сравнения."""
    return {
        "DummyClassifier": DummyClassifier(
            strategy="most_frequent",
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression": LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=15),
    }


def make_pipeline(model: object) -> object:
    """Оборачивает модель в pipeline с preprocessing, кроме baseline."""
    if isinstance(model, DummyClassifier):
        return model
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", model),
        ]
    )


def _positive_scores(model: object, x_values: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_values)[:, 1]
    scores = model.decision_function(x_values)
    return (scores - scores.min()) / (scores.max() - scores.min())


def evaluate_model(model: object, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Считает метрики и элементы матрицы ошибок на test set."""
    predictions = model.predict(x_test)
    scores = _positive_scores(model, x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Возвращает имена признаков после one-hot кодирования."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return [str(name) for name in names]


def _save_eda_figures(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=TARGET_COLUMN, hue=TARGET_COLUMN, palette="Set2")
    ax.set_title("Распределение целевой переменной Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Количество клиентов")
    ax.legend_.remove()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
    plt.title("Корреляционная матрица признаков")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=160)
    plt.close()

    numeric_for_plot = [
        "call_failures",
        "subscription_length",
        "seconds_of_use",
        "frequency_of_use",
        "frequency_of_sms",
        "distinct_called_numbers",
        "customer_value",
    ]
    axes = df[numeric_for_plot].hist(figsize=(12, 9), bins=30)
    for axis in axes.ravel():
        axis.set_xlabel("")
    plt.suptitle("Распределения ключевых числовых признаков")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_distributions.png", dpi=160)
    plt.close()

    compare_features = [
        "complaints",
        "status",
        "frequency_of_use",
        "seconds_of_use",
        "customer_value",
        "distinct_called_numbers",
    ]
    melted = df.melt(
        id_vars=TARGET_COLUMN,
        value_vars=compare_features,
        var_name="feature",
        value_name="value",
    )
    grid = sns.catplot(
        data=melted,
        x=TARGET_COLUMN,
        y="value",
        col="feature",
        kind="box",
        col_wrap=3,
        sharey=False,
        height=3.2,
        aspect=1.15,
    )
    grid.fig.suptitle("Сравнение признаков по классам Churn", y=1.03)
    grid.set_axis_labels("Churn", "Значение")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "features_by_churn.png", dpi=160)
    plt.close()


def _save_model_figures(
    test_results: pd.DataFrame,
    best_name: str,
    best_model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_importances: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_df = test_results.sort_values("f1", ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["model"], plot_df["f1"], label="F1")
    plt.scatter(plot_df["roc_auc"], plot_df["model"], color="#d62728", label="ROC-AUC")
    plt.xlabel("Метрика")
    plt.title("Сравнение моделей на test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_f1_roc_auc.png", dpi=160)
    plt.close()

    predictions = best_model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["pred 0", "pred 1"],
        yticklabels=["true 0", "true 1"],
    )
    plt.title(f"Confusion matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_best_model.png", dpi=160)
    plt.close()

    if not feature_importances.empty:
        top = feature_importances[
            feature_importances["model"] == "GradientBoostingClassifier"
        ].head(10)
        if top.empty:
            top = feature_importances.head(10)
        top = top.sort_values("importance", ascending=True)
        plt.figure(figsize=(8, 5))
        plt.barh(top["feature"], top["importance"], color="#4c78a8")
        plt.xlabel("Важность")
        plt.title("Топ-10 важных признаков")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance_top10.png", dpi=160)
        plt.close()


def _extract_tree_importances(fitted_models: dict[str, object]) -> pd.DataFrame:
    rows = []
    for model_name in ["RandomForestClassifier", "GradientBoostingClassifier"]:
        pipeline = fitted_models[model_name]
        feature_names = get_feature_names(pipeline)
        importances = pipeline.named_steps["model"].feature_importances_
        for rank, index in enumerate(np.argsort(importances)[::-1][:10], start=1):
            rows.append(
                {
                    "model": model_name,
                    "rank": rank,
                    "feature": feature_names[index],
                    "importance": importances[index],
                }
            )
    return pd.DataFrame(rows)


def _extract_logistic_coefficients(fitted_models: dict[str, object]) -> pd.DataFrame:
    pipeline = fitted_models["LogisticRegression"]
    feature_names = get_feature_names(pipeline)
    coefficients = pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    ).assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
    return coef_df.sort_values("abs_coefficient", ascending=False).head(20)


def run_experiment(save_artifacts: bool = True) -> dict:
    """Запускает полный эксперимент для ноутбуков и CLI."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = find_dataset()
    df = load_churn_data(dataset_path, remove_duplicates=True)
    quality = dataset_quality_summary(dataset_path)
    x_values, y_values = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=0.20,
        stratify=y_values,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_rows = []
    test_rows = []
    confusion_matrices = {}
    fitted_models: dict[str, object] = {}

    for model_name, estimator in make_models().items():
        LOGGER.info("Обучаю %s", model_name)
        model = make_pipeline(estimator)
        cv_result = cross_validate(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            return_train_score=True,
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        fitted_models[model_name] = model

        cv_row = {"model": model_name}
        for metric_name in SCORING:
            cv_row[f"cv_{metric_name}_mean"] = float(
                np.mean(cv_result[f"test_{metric_name}"])
            )
            cv_row[f"cv_{metric_name}_std"] = float(
                np.std(cv_result[f"test_{metric_name}"])
            )
            cv_row[f"train_{metric_name}_mean"] = float(
                np.mean(cv_result[f"train_{metric_name}"])
            )
        cv_rows.append(cv_row)

        test_metrics = evaluate_model(model, x_test, y_test)
        test_rows.append({"model": model_name, **test_metrics})
        confusion_matrices[model_name] = [
            [test_metrics["tn"], test_metrics["fp"]],
            [test_metrics["fn"], test_metrics["tp"]],
        ]

    cv_results = pd.DataFrame(cv_rows).sort_values("cv_f1_mean", ascending=False)
    test_results = pd.DataFrame(test_rows).sort_values(
        ["f1", "roc_auc", "recall"], ascending=False
    )

    best_name = str(test_results.iloc[0]["model"])
    best_model = fitted_models[best_name]
    feature_importances = _extract_tree_importances(fitted_models)
    logistic_coefficients = _extract_logistic_coefficients(fitted_models)

    metadata = {
        "dataset": quality,
        "target": "Churn",
        "code_target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "feature_leakage_note": (
            "Признак status оставлен в учебном эксперименте, но для production "
            "нужно проверить момент его фиксации относительно факта оттока."
        ),
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "selection_rule": "лучшая test F1, затем ROC-AUC и Recall",
        "best_model": best_name,
        "best_model_path": display_path(BEST_MODEL_PATH),
        "best_test_metrics": {
            key: float(test_results.iloc[0][key])
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        },
    }

    if save_artifacts:
        cv_results.to_csv(ARTIFACTS_DIR / "model_comparison_cv.csv", index=False)
        test_results.to_csv(ARTIFACTS_DIR / "model_comparison_test.csv", index=False)
        feature_importances.to_csv(
            ARTIFACTS_DIR / "feature_importance_top10.csv",
            index=False,
        )
        logistic_coefficients.to_csv(
            ARTIFACTS_DIR / "logistic_coefficients_top.csv",
            index=False,
        )
        with (ARTIFACTS_DIR / "confusion_matrices.json").open("w", encoding="utf-8") as file:
            json.dump(confusion_matrices, file, ensure_ascii=False, indent=2)
        with (ARTIFACTS_DIR / "model_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        joblib.dump(best_model, BEST_MODEL_PATH)
        _save_eda_figures(df)
        _save_model_figures(
            test_results,
            best_name,
            best_model,
            x_test,
            y_test,
            feature_importances,
        )

    LOGGER.info("Лучшая модель: %s", best_name)
    LOGGER.info("Метрики лучшей модели на test: %s", metadata["best_test_metrics"])
    return {
        "cv_results": cv_results,
        "test_results": test_results,
        "best_model": best_name,
        "metadata": metadata,
    }


def main() -> None:
    results = run_experiment(save_artifacts=True)
    print("\n=== Метрики на test ===")
    print(
        results["test_results"][
            ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]
        ].round(4).to_string(index=False)
    )
    print(f"\nЛучшая модель: {results['best_model']}")
    print(f"Модель сохранена: {display_path(BEST_MODEL_PATH)}")


if __name__ == "__main__":
    main()
