import numpy as np
import math


def numeric_scorer(y_true, y_pred):
    """_summary_

    res_SS = Residual Sum of Squares (squared sum)
    tot_SS = Total Sum of Squares (squared sum)
    res_sum = Residual Sum (sum)
    res_abs_sum = Sum of the absolute values

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    res_SS = ((y_true - y_pred) ** 2).sum(dtype=np.float64)
    tot_SS = ((y_true - np.average(y_true)) ** 2).sum(dtype=np.float64)
    res_sum = (y_true - y_pred).sum(dtype=np.float64)
    res_abs_sum = (abs(y_true - y_pred)).sum(dtype=np.float64)

    mape_sum = (abs(y_true - y_pred) / abs(y_true)).sum(dtype=np.float64)

    r2 = 1 - (res_SS / tot_SS)  # Coefficient of determination (R2)
    mae = res_abs_sum / len(y_true)  # Mean absolute error (MAE)
    mse = res_SS / len(y_true)  # Mean squared error (MSE)
    rmse = mse**(0.5)  # Root-mean-square error (RMSE ou RMSD)
    msd = res_sum / len(y_true)  # Mean signed difference (MSD)
    mape = mape_sum / len(y_true)  # Mean absolute percentage error (MAPE)

    metrics = {'r2': [r2], 'MAE': [mae], 'MSE': [mse],
               'RMSE': [rmse], 'MSD': [msd], 'MAPE': [mape]}

    return metrics


def concordance_correlation(y_true, y_pred):
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    numerator = 2 * np.sum((y_true - mean_true) *
                           (y_pred - mean_pred))

    denominator_1 = np.sum((y_true - mean_true) ** 2)

    denominator_2 = np.sum((y_pred - mean_pred) ** 2)

    denominator_3 = len(y_true) * ((mean_true - mean_pred) ** 2)

    ccc = numerator / (denominator_1 + denominator_2 + denominator_3)

    return ccc


def q2_3_function(y_true, y_pred):
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    res_SS = ((y_true - y_pred) ** 2).sum(dtype=np.float64)
    tot_SS = ((y_true - np.average(y_true)) ** 2).sum(dtype=np.float64)

    q2f3 = 1 - ((res_SS / len(y_true)) / (tot_SS / len(y_true)))

    return q2f3


def tropsha_roy_criteria(y_true, y_pred):
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    # * Tropsha criteria
    kappa_1 = ((y_true - y_pred).sum(
        dtype=np.float64)) / ((y_pred) ** 2).sum(dtype=np.float64)
    y_kappa_1 = y_pred * kappa_1
    numerator_1 = ((y_pred - y_kappa_1) ** 2).sum(dtype=np.float64)
    denominator_1 = ((y_pred - np.average(y_pred)) ** 2).sum(dtype=np.float64)
    r0_1 = 1 - (numerator_1 / denominator_1)

    kappa_2 = ((y_true - y_pred).sum(
        dtype=np.float64)) / ((y_true) ** 2).sum(dtype=np.float64)
    y_kappa_2 = y_true * kappa_2
    numerator_2 = ((y_true - y_kappa_2) ** 2).sum(dtype=np.float64)
    denominator_2 = ((y_true - np.average(y_true)) ** 2).sum(dtype=np.float64)
    r0_2 = 1 - (numerator_2 / denominator_2)

    # * Roy criteria
    res_SS = ((y_true - y_pred) ** 2).sum(dtype=np.float64)
    tot_SS = ((y_true - np.average(y_true)) ** 2).sum(dtype=np.float64)
    r2 = 1 - (res_SS / tot_SS)

    r2_m = r2 * (1 - math.sqrt(r2 - r0_1))
    delta_r2_m = abs(r2 - r0_1)

    return [r0_1, r0_2], [r2_m, delta_r2_m]
