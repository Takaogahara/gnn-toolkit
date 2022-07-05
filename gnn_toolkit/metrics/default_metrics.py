import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import mlflow.pytorch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, mean_squared_error,
                             mean_absolute_error, r2_score, matthews_corrcoef,
                             ConfusionMatrixDisplay, balanced_accuracy_score)
from deepchem.metrics import bedroc_score

from default_custom import (concordance_correlation, q2_3_function,
                            tropsha_roy_criteria)
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
    y_pred = y_pred.astype(int)
    # * Calculate metrics
    acc = accuracy_score(y_pred, y_true)
    acc_bal = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_pred, y_true)
    prec = precision_score(y_pred, y_true)
    rec = recall_score(y_pred, y_true)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        roc = roc_auc_score(y_pred, y_true)
    except Exception:
        roc = 0

    bed_pred = np.zeros((y_pred.size, y_pred.max()+1))
    bed_pred[np.arange(y_pred.size), y_pred] = 1
    bedroc = bedroc_score(y_true, bed_pred)

    # * Log on mlflow
    mlflow.log_metric(key=f"F1 Score-{run}", value=float(f1), step=num)
    mlflow.log_metric(key=f"Accuracy-{run}", value=float(acc), step=num)
    mlflow.log_metric(key=f"Accuracy balanced-{run}", value=float(acc_bal),
                      step=num)
    mlflow.log_metric(key=f"Precision-{run}", value=float(prec), step=num)
    mlflow.log_metric(key=f"Recall-{run}", value=float(rec), step=num)
    mlflow.log_metric(key=f"ROC-AUC-{run}", value=float(roc), step=num)
    mlflow.log_metric(key=f"MCC-{run}", value=float(mcc), step=num)
    mlflow.log_metric(key=f"Bedroc-{run}", value=float(bedroc), step=num)

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

        mlflow.log_figure(cm_raw.figure_, f"cm_{num}_raw.png")
        mlflow.log_figure(cm_norm.figure_, f"cm_{num}_norm.png")
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
    q2_f1 = r2_score(y_true, y_pred)

    # * Log on mlflow
    mlflow.log_metric(key=f"RMSE-{run}", value=float(rmse), step=num)
    mlflow.log_metric(key=f"MAE-{run}", value=float(mae), step=num)
    mlflow.log_metric(key=f"Q2 f1-{run}", value=float(q2_f1), step=num)

    # * Plot regression for test run
    if run == "test":
        q2_f2 = r2_score(y_true, y_pred)
        q2_f3 = q2_3_function(y_true, y_pred)
        ccc = concordance_correlation(y_true, y_pred)

        tropsha, roy = tropsha_roy_criteria(y_true, y_pred)
        r2_0, r2_01 = tropsha
        r2_m, delta_r2_m = roy

        mlflow.log_metric(key=f"Q2 f2-{run}", value=float(q2_f2), step=num)
        mlflow.log_metric(key=f"Q2 f3-{run}", value=float(q2_f3), step=num)
        mlflow.log_metric(key=f"CCC-{run}", value=float(ccc), step=num)
        mlflow.log_metric(key=f"Tropsha 0-{run}", value=float(r2_0), step=num)
        mlflow.log_metric(key=f"Tropsha 1-{run}", value=float(r2_01), step=num)
        mlflow.log_metric(key=f"Roy R2m-{run}", value=float(r2_m), step=num)
        mlflow.log_metric(key=f"Roy Delta R2m-{run}", value=float(delta_r2_m),
                          step=num)

        data = {"true": y_true, "pred": y_pred}
        df_reg = pd.DataFrame.from_dict(data)

        fig = plt.figure(figsize=(10, 7))
        reg_plot = sns.regplot(data=df_reg, x="true",
                               y="pred", fit_reg=True)
        ax = fig.axes[0]
        anchored_text = AnchoredText(f"Q2 f2 = {round(q2_f2, 4)}", loc=2)
        ax.add_artist(anchored_text)

        mlflow.log_figure(reg_plot.figure, f"reg_{num}.png")
        plt.close("all")
