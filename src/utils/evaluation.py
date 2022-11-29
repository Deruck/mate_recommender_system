from typing import List
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from json import dumps

from .logger_manager import LoggerManager

logger = LoggerManager.get_logger()

def evaluate_model(predict_probs: List[float], true_label: List[int]) -> None:
    probs = np.array(predict_probs)
    predict = (probs > 0.5).astype(int)
    true = np.array(true_label)
    acc = accuracy_score(true, predict)
    f1 = f1_score(true, predict)
    auc = roc_auc_score(true, probs)
    tn, fp, fn, tp = confusion_matrix(true, predict).ravel()
    report = f"""
    
    ===============================
           Model Evaluation
    -------------------------------
    - accuracy: {acc:.4f}
    - f1-score: {f1:.4f}
    - auc-score: {auc:.4f}
    - confusion matrix: 
        tp: {tp:5}    fp: {fp:5}
        fn: {fn:5}    tn: {tn:5}
    ===============================
    
    """
    logger.info(report)