import torch
from torch import Tensor
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import classification_report
import pandas as pd
from typing import Optional, Union

from classifiers.bagger.voting_bagger import VotingBagger

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import sys; sys.path.append(".")
from utils import log


LogisticRegressionArgs = dict(
    C=1.2,
    max_iter=5000,
)

LinearSVMArgs = dict(
    class_weight='balanced',
    C=0.8,
    penalty='l2',
    loss='squared_hinge',
    dual=True,
    max_iter=5000,
    random_state=42,
    verbose=1,
)

RBFSVMArgs = dict(
    kernel='rbf',
    C=5,
    gamma='scale',
    class_weight='balanced',
    max_iter=5000,
    verbose=True,
)

DecisionTreeArgs = dict(
    criterion='gini',
    class_weight='balanced',
    max_depth=30,
    min_samples_leaf=2,
    ccp_alpha=0.0,
    random_state=42,
)

RandomForestArgs = dict(
    n_estimators=100,
    class_weight='balanced',
    n_jobs=-1,
    max_depth=20,
    random_state=42,
)

XGBoostArgs = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=None,  # provided when building
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42,
)

def get_bagger_model(
        enable_logistic_regression: bool = True,
        enable_linear_svm: bool = True,
        enable_nonlinear_svm: bool = True,
        enable_decision_tree: bool = True,
        enable_random_forest: bool = True,
        enable_xgb: bool = True,
        *,
        logistic_regression_args: Optional[dict] = None,
        linear_svm_args: Optional[dict] = None,
        nonlinear_svm_args: Optional[dict] = None,
        decision_tree_args: Optional[dict] = None,
        random_forest_args: Optional[dict] = None,
        xgb_args: Optional[dict] = None,
        xgb_ref_y_train: Optional[NDArray] = None,
        **kwargs,
):
    models = []

    if enable_logistic_regression:
        if logistic_regression_args is None:
            logistic_regression_args = LogisticRegressionArgs
        models.append(LogisticRegression(**logistic_regression_args))
    
    if enable_linear_svm:
        if linear_svm_args is None:
            linear_svm_args = LinearSVMArgs
        models.append(LinearSVC(**linear_svm_args))
    
    if enable_nonlinear_svm:
        if nonlinear_svm_args is None:
            nonlinear_svm_args = RBFSVMArgs;
        models.append(SVC(**nonlinear_svm_args))
    
    if enable_decision_tree:
        if decision_tree_args is None:
            decision_tree_args = DecisionTreeArgs
        models.append(DecisionTreeClassifier(**decision_tree_args))
    
    if enable_random_forest:
        if random_forest_args is None:
            random_forest_args = RandomForestArgs
        models.append(RandomForestClassifier(**random_forest_args))
    
    if enable_xgb:
        if xgb_args is None:
            xgb_args = XGBoostArgs
        assert xgb_ref_y_train is not None, "Provide `y_train` labels when XGBoost is enabled"
        scale_pos_weight = float(sum(xgb_ref_y_train == 0)) / sum(xgb_ref_y_train == 1)
        xgb_args['scale_pos_weight'] = scale_pos_weight
        models.append(XGBClassifier(**xgb_args))
    
    assert len(models) > 0, "Enable at least one model to build boost model"
    
    return VotingBagger(*models)


def train_bagger_model(model: VotingBagger, x_train: Union[NDArray, Tensor], y_train: Union[NDArray, Tensor]):
    """
    Args:
        x_train: shape `[sample_count, feat_count]`
        y_train: shape `[sample_count]`
    """
    if isinstance(x_train, Tensor):
        x_train = x_train.cpu().numpy()
    if isinstance(y_train, Tensor):
        y_train = y_train.cpu().numpy()
    
    if len(x_train.shape) == 1:
        x_train = x_train.reshape(1, x_train.shape[0])
    
    model.train(x_train, y_train)

    return model


def predict_bagger_model(model: VotingBagger, x_test: Union[NDArray, Tensor], y_test: Optional[Union[NDArray, Tensor]] = None, visualize: bool = False) -> NDArray:
    """
    Args:
        x_test: shape `[sample_count, feat_count]`
        y_test: shape `[sample_count]` (optional)
    Returns:
        y_pred (NDArray): shape `[sample_count]`
    """
    if isinstance(x_test, Tensor):
        x_test = x_test.cpu().numpy()
    if isinstance(y_test, Tensor):
        y_test = y_test.cpu().numpy()
    
    if len(x_test.shape) == 1:
        x_test = x_test.reshape(1, x_test.shape[0])
    
    if visualize:
        y_pred = model.predict_and_visalize(x_test)
    y_pred = model.predict(x_test)

    if y_test is not None:
        log.info(classification_report(y_test, y_pred))