from sklearn.tree import DecisionTreeClassifier

from config import config
from data import get_data
from tracking import create_task, evaluate_model, log_evaluation


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(task, model, x_test, y_test) -> None:
    evaluation = evaluate_model(model, x_test, y_test)
    log_evaluation(task, evaluation)


def log_model_parameters(task, model) -> None:
    logger = task.get_logger()
    logger.report_scalar(
        title="model_parameters",
        series="tree_depth",
        value=model.get_depth(),
        iteration=0,
    )
    logger.report_scalar(
        title="model_parameters",
        series="leaf_count",
        value=model.get_n_leaves(),
        iteration=0,
    )


if __name__ == "__main__":
    params = config["decision_tree"]
    task = create_task(
        task_name="Decision Tree",
        params={
            "model": "DecisionTreeClassifier",
            "random_state": config["random_state"],
            "test_size": config["data"]["test_size"],
            **params,
        },
    )

    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=params["max_depth"],
        criterion=params["criterion"],
    )

    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    log_model_parameters(task, decision_tree_model)
    test(task, decision_tree_model, data["x_test"], data["y_test"])
    task.close()
