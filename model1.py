from sklearn.linear_model import LogisticRegression

from config import config
from data import get_data
from tracking import create_task, evaluate_model, log_evaluation


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(task, model, x_test, y_test) -> None:
    evaluation = evaluate_model(model, x_test, y_test)
    log_evaluation(task, evaluation)


def log_model_parameters(task, model) -> None:
    task.upload_artifact(
        name="regression_coefficients",
        artifact_object=model.coef_.tolist(),
    )
    task.upload_artifact(
        name="regression_intercept",
        artifact_object=model.intercept_.tolist(),
    )


if __name__ == "__main__":
    params = config["logistic_regression"]
    task = create_task(
        task_name="Logistic Regression",
        params={
            "model": "LogisticRegression",
            "random_state": config["random_state"],
            "test_size": config["data"]["test_size"],
            **params,
        },
    )

    logistic_regression_model = LogisticRegression(
        random_state=config["random_state"],
        max_iter=params["max_iter"],
        penalty=params["penalty"],
        C=params["C"],
        solver=params["solver"],
    )

    data = get_data()
    train(logistic_regression_model, data["x_train"], data["y_train"])
    log_model_parameters(task, logistic_regression_model)
    test(task, logistic_regression_model, data["x_test"], data["y_test"])
    task.close()
