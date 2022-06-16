import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, mean_squared_error,
                             mean_absolute_error, r2_score, matthews_corrcoef,
                             ConfusionMatrixDisplay)
matplotlib.rcParams['figure.figsize'] = [10, 7]


def log_metrics(task: str, name: str, eval_type: str,
                current_epoch: int, y_pred, y_true):
    """External function for compute metrics

    Args:
        task (str): Current task
        name (str): Experiment name
        eval_type (str): Evaluation type
        current_epoch (int): Current epoch
        y_pred (_type_): Y pred
        y_true (_type_): Y true
    """
    if task == "Classification":
        _log_classification(name, eval_type, current_epoch,  y_pred, y_true)

    elif task == "Regression":
        _log_regression(name, eval_type, current_epoch,  y_pred, y_true)


def _log_classification(name, run: str, num: int, y_pred, y_true):
    """Calculate and log to mlflow classification metrics

    Args:
        name (str): Experiment name
        run (str): Evaluation type
        num (int): Current epoch
        y_pred (_type_): Y pred
        y_true (_type_): Y true
    """
    # * Calculate metrics
    acc = accuracy_score(y_pred, y_true)
    f1 = f1_score(y_pred, y_true)
    prec = precision_score(y_pred, y_true)
    rec = recall_score(y_pred, y_true)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        roc = roc_auc_score(y_pred, y_true)
    except Exception:
        roc = 0

    # * Plot confusion matrix for test run
    if run == "test":
        cm_raw = ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                                         colorbar=False,
                                                         cmap="Blues",
                                                         normalize=None)
        cm_norm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                                          colorbar=False,
                                                          cmap="Blues",
                                                          normalize="true")

        with TemporaryDirectory(dir="./") as temp_dir:
            cm_raw.figure_.savefig(f"{temp_dir}/cm_{num}_raw.png")
            cm_norm.figure_.savefig(f"{temp_dir}/cm_{num}_norm.png")
            plt.close("all")


def _log_regression(name, run: str, num: int, y_pred, y_true):
    """Calculate and log to mlflow regression metrics

    Args:
        name (str): Experiment name
        run (str): Evaluation type
        num (int): Current epoch
        y_pred (_type_): Y pred
        y_true (_type_): Y true
    """
    # * Calculate metrics
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # * Plot regression for test run
    if run == "test":
        data = {"true": y_true, "pred": y_pred}
        df_reg = pd.DataFrame.from_dict(data)

        fig = plt.figure(figsize=(10, 7))
        reg_plot = sns.regplot(data=df_reg, x="true",
                               y="pred", fit_reg=True)
        ax = fig.axes[0]
        anchored_text = AnchoredText(f"R2 = {round(r2, 4)}", loc=2)
        ax.add_artist(anchored_text)

        with TemporaryDirectory(dir="./output/") as temp_dir:
            reg_plot.figure.savefig(f"{temp_dir}/reg_{num}.png")
            plt.close("all")
