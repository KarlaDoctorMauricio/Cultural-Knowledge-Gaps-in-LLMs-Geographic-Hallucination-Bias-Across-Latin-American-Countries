"""Group calibration post-processing for regression predictions."""

from __future__ import annotations

import numpy as np


def group_calibration_regression(preds, sensitive):
    calibrated = preds.copy()
    global_mean = preds.mean()

    for group in np.unique(sensitive):
        mask = sensitive == group
        group_mean = preds[mask].mean()
        calibrated[mask] = preds[mask] - (group_mean - global_mean)

    return calibrated


def apply_postprocessing(preds_dict, sensitive, target_models=None):
    """
    Apply group calibration to selected models in ``preds_dict``.

    Esta funcion reescala scores post-hoc para igualar medias entre grupos por
    construccion algebraica; no modifica las respuestas del modelo ni mide mejora
    real. Se mantiene como metodo demostrativo de mitigacion aplicable si los
    scores se usan como umbral de decision en produccion, no como resultado de
    investigacion.
    """
    new_dict = preds_dict.copy()

    if target_models is None:
        target_models = ["lagrangian"]
    elif target_models == "all":
        target_models = list(preds_dict.keys())

    for model_name in target_models:
        if model_name not in preds_dict:
            continue
        new_name = f"{model_name}_post"
        new_dict[new_name] = group_calibration_regression(
            preds_dict[model_name],
            sensitive,
        )

    return new_dict
