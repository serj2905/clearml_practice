from clearml import Task
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


PROJECT_NAME = "MLOps 4 Tracking Digits"
DIGIT_LABELS = [str(label) for label in range(10)]


def create_task(task_name: str, params: dict) -> Task:
    task = Task.init(project_name=PROJECT_NAME, task_name=task_name)
    task.connect(params)
    return task


def evaluate_model(model, x_test, y_test) -> dict:
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)
        metrics["auc_roc"] = roc_auc_score(
            y_test,
            y_proba,
            average="weighted",
            multi_class="ovr",
        )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=range(10)),
    }


def log_evaluation(task: Task, evaluation: dict, iteration: int = 0) -> None:
    logger = task.get_logger()

    for metric_name, metric_value in evaluation["metrics"].items():
        value = float(metric_value)
        logger.report_scalar(
            title="test_metrics",
            series=metric_name,
            value=value,
            iteration=iteration,
        )
        print(f"{metric_name}: {value:.4f}")

    logger.report_confusion_matrix(
        title="confusion_matrix",
        series="test",
        matrix=evaluation["confusion_matrix"],
        iteration=iteration,
        xaxis="Predicted label",
        yaxis="True label",
        xlabels=DIGIT_LABELS,
        ylabels=DIGIT_LABELS,
        yaxis_reversed=True,
    )
