"""
Random Forest classification — two prediction targets:
  Case 1 (7-class): winner + victory_status  →  'Target'
  Case 2 (3-class): winner only              →  'white' / 'black' / 'draw'
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as RFC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "rated", "turns", "white_rating", "black_rating",
    "opening_ply", "fix_time", "variable_time", "opening_eco_",
]

PARAM_GRID_DEPTHS = [5, 15]
PARAM_GRID_BINS = [365]
PARAM_GRID_TREES = [5, 15, 30, 60]


def _build_evaluators(label_col: str):
    kwargs = dict(predictionCol="prediction", labelCol=label_col)
    return {
        "accuracy":  MulticlassClassificationEvaluator(metricName="accuracy",        **kwargs),
        "precision": MulticlassClassificationEvaluator(metricName="precisionByLabel", **kwargs),
        "recall":    MulticlassClassificationEvaluator(metricName="recallByLabel",    **kwargs),
        "f1":        MulticlassClassificationEvaluator(metricName="f1",               **kwargs),
    }


def _cross_validate(rf, param_grid, evaluator, train):
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5,
    )
    return cv.fit(train)


def _print_metrics(evals: dict, predictions):
    print(f"Accuracy : {evals['accuracy'].evaluate(predictions):.4f}")
    print(f"Precision: {evals['precision'].evaluate(predictions):.4f}")
    print(f"Recall   : {evals['recall'].evaluate(predictions):.4f}")
    print(f"F1       : {evals['f1'].evaluate(predictions):.4f}")


def _plot_confusion_matrix(predictions, label_col: str, class_labels: list,
                           save_path: str = None):
    preds_and_labels = (
        predictions.select(["prediction", label_col])
        .withColumn("label", F.col(label_col).cast(FloatType()))
        .orderBy("prediction")
        .select(["prediction", "label"])
    )
    metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
    cm = metrics.confusionMatrix().toArray()
    row_sums = np.sum(cm, axis=1)

    ax = sns.heatmap(cm / row_sums[:, None], annot=True, fmt=".2%")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(class_labels, rotation=45)
    ax.set_yticklabels(class_labels, rotation=0)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return cm


# ---------------------------------------------------------------------------
# Case 1 — 7-class (winner + victory_status)
# ---------------------------------------------------------------------------

CASE1_DROP = ("id", "white_id", "black_id", "opening_name", "winner")
CASE1_LABELS = [
    "white_resign", "black_resign", "white_mate",
    "black_mate", "draw-draw", "black_outoftime", "white_outoftime",
]


def run_case1(df, save_cm_path: str = "outputs/case1_confusion_matrix.png"):
    df1 = df.drop(*CASE1_DROP)

    indexer = StringIndexer(
        inputCols=["Target", "opening_eco"],
        outputCols=["Target_", "opening_eco_"],
        handleInvalid="skip",
    )
    df1 = indexer.fit(df1).transform(df1)

    assembler = VectorAssembler(inputCols=NUMERIC_FEATURES, outputCol="features")
    data = assembler.transform(df1).select("features", "Target_")

    fractions = {float(i): 0.7 for i in range(7)}
    train = data.sampleBy("Target_", fractions=fractions)
    test = data.subtract(train)

    rf = RFC(featuresCol="features", labelCol="Target_", impurity="gini")
    evals = _build_evaluators("Target_")

    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.maxDepth, PARAM_GRID_DEPTHS)
        .addGrid(rf.maxBins,  PARAM_GRID_BINS)
        .addGrid(rf.numTrees, PARAM_GRID_TREES)
        .build()
    )

    cv_model = _cross_validate(rf, param_grid, evals["accuracy"], train)
    predictions = cv_model.transform(test)

    print("\n=== Case 1: 7-class (winner + victory_status) ===")
    _print_metrics(evals, predictions)
    _plot_confusion_matrix(predictions, "Target_", CASE1_LABELS, save_cm_path)
    return cv_model, predictions


# ---------------------------------------------------------------------------
# Case 2 — 3-class (winner only)
# ---------------------------------------------------------------------------

CASE2_DROP = ("id", "white_id", "black_id", "opening_name", "Target")
CASE2_LABELS = ["white", "black", "draw"]


def run_case2(df, save_cm_path: str = "outputs/case2_confusion_matrix.png"):
    df2 = df.drop(*CASE2_DROP)

    indexer = StringIndexer(
        inputCols=["winner", "opening_eco"],
        outputCols=["winner_", "opening_eco_"],
        handleInvalid="skip",
    )
    df2 = indexer.fit(df2).transform(df2).drop("opening_eco", "winner")

    assembler = VectorAssembler(inputCols=NUMERIC_FEATURES, outputCol="features")
    data = assembler.transform(df2).select("features", "winner_")

    train = data.sampleBy("winner_", fractions={0.0: 0.7, 1.0: 0.7, 2.0: 0.7})
    test = data.subtract(train)

    rf = RFC(featuresCol="features", labelCol="winner_", impurity="gini")
    evals = _build_evaluators("winner_")

    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.maxDepth, PARAM_GRID_DEPTHS)
        .addGrid(rf.maxBins,  PARAM_GRID_BINS)
        .addGrid(rf.numTrees, PARAM_GRID_TREES)
        .build()
    )

    cv_model = _cross_validate(rf, param_grid, evals["accuracy"], train)
    predictions = cv_model.transform(test)

    print("\n=== Case 2: 3-class (winner only) ===")
    _print_metrics(evals, predictions)
    _plot_confusion_matrix(predictions, "winner_", CASE2_LABELS, save_cm_path)
    return cv_model, predictions
