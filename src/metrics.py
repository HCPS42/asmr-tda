import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target
import time

from config import (
    EXPERIMENTS_RESULTS_FILE,
    TIME_DELAY, DIMENSION, STRIDE,
    CHANNELS, TARGET, N_INTERVALS_PER_PERSON_PER_CLASS,
    TRAIN_VAL_SAME_PEOPLE, N_PEOPLE, CLASSIFIER_NAME,
    EXPERIMENT_GOAL
)


def compute_metrics(step, y_true, y_pred, y_probs):
    average_type = 'binary' if type_of_target(y_true) == 'binary' else 'macro'

    metrics = {
        f'{step} Samples': y_true.shape[0],
        f'{step} Accuracy': accuracy_score(y_true, y_pred),
        f'{step} Precision': precision_score(y_true, y_pred, average=average_type),
        f'{step} Recall': recall_score(y_true, y_pred, average=average_type),
        f'{step} F1 Score': f1_score(y_true, y_pred, average=average_type)
    }
    
    if type_of_target(y_true) == 'binary':
        metrics[f'{step} ROC AUC'] = roc_auc_score(y_true, y_probs)
    else:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        metrics[f'{step} ROC AUC'] = roc_auc_score(y_true_bin, y_probs, multi_class='ovr', average=average_type)
    
    return metrics

def print_metrics(metrics):
    for key, value in metrics.items():
        if isinstance(value, int):
            print(f'{key}: \t{value}')
        elif isinstance(value, float):
            print(f'{key}: \t{value:.2f}')

def save_result_to_file(result):
    filename = EXPERIMENTS_RESULTS_FILE
    if os.path.exists(filename):
        results_df = pd.read_pickle(filename)
    else:
        results_df = pd.DataFrame()
    result_df = pd.DataFrame([result])
    results_df = pd.concat([results_df, result_df], ignore_index=True)
    results_df.to_pickle(filename)

def evaluate_pipeline(pipeline, X_train, X_val, y_train, y_val, execution_time, save_result=False):
    y_pred_train = pipeline.predict(X_train)
    if type_of_target(y_train) == 'binary':
        y_probs_train = pipeline.predict_proba(X_train)[:, 1]
    else:
        y_probs_train = pipeline.predict_proba(X_train)
    train_metrics = compute_metrics('Training', y_train, y_pred_train, y_probs_train)

    y_pred_val = pipeline.predict(X_val)
    if type_of_target(y_val) == 'binary':
        y_probs_val = pipeline.predict_proba(X_val)[:, 1]
    else:
        y_probs_val = pipeline.predict_proba(X_val)
    val_metrics = compute_metrics('Validation', y_val, y_pred_val, y_probs_val)

    metrics = {**train_metrics, **val_metrics}

    print_metrics(metrics)

    print(f'\nExecution Time: {execution_time:.2f} seconds')

    if save_result:
        params = {
            'Time Delay': TIME_DELAY,
            'Dimension': DIMENSION,
            'Stride': STRIDE,
            'Channels': CHANNELS,
            'Target': TARGET,
            'Intervals per Person per Class': N_INTERVALS_PER_PERSON_PER_CLASS,
            'Train Validation Same People': TRAIN_VAL_SAME_PEOPLE,
            'Number of People': N_PEOPLE,
            'Classifier': CLASSIFIER_NAME,
            'Experiment Goal': EXPERIMENT_GOAL,
            'Execution Time': execution_time
        }
        
        result = {**params, **metrics}

        save_result_to_file(result)
