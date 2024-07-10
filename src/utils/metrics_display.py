import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def display_metrics(y_true, y_pred_prob, prefix: str, class_labels: List[str]) -> Dict:
    """
    Get performance plot.
    
    --------------------
    Parameters
    ----------
    - `y_true` : True labels.
    - `y_pred_prob` : Probability of predict labels
    - `prefix` : Prefix for the plot names.
    
    Returns
    -------
    - Performance plots. 
    --------------------
    """
    
    roc_figure = _roc_display(y_true, y_pred_prob, class_labels)
    cm_figure = _confusion_matrix_display(y_true, y_pred_prob, class_labels)
    cr_figure = _classification_report_display(y_true, y_pred_prob, class_labels)
    return  {
        f"{prefix}_roc_curve": roc_figure,
        f"{prefix}_confusion_matrix": cm_figure,
        f"{prefix}_classification_report": cr_figure
    }
    

def _roc_display(y_true, y_pred_prob, class_labels: List[str]) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    --------------------
    Parameters
    ----------
    - `y_true` : array-like of shape (n_samples,)
        True labels.
    - `y_pred_prob` : array-like of shape (n_samples, n_classes)
        Probability predictions for each class.
    - `class_labels` : List of str
        Labels for each class.
    
    Returns
    -------
    - `fig` : matplotlib.figure.Figure
        Figure object with the ROC curves plotted.
    --------------------
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    num_class = y_pred_prob.shape[1]
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_pred_prob[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Plot ROC curve for each class
    fig, ax = plt.subplots(figsize=(5, 5))
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(num_class), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Class {class_labels[i]} (AUC = {roc_auc[i]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (One-vs-All)')
    ax.legend(loc="lower right")
    
    return fig


from typing import List
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import pandas as pd
import seaborn as sns

def _roc_display(y_true, y_pred_prob, class_labels: List[str]) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    --------------------
    Parameters
    ----------
    - `y_true` : array-like of shape (n_samples,)
        True labels.
    - `y_pred_prob` : array-like of shape (n_samples, n_classes)
        Probability predictions for each class.
    - `class_labels` : List of str
        Labels for each class.
    
    Returns
    -------
    - `fig` : matplotlib.figure.Figure
        Figure object with the ROC curves plotted.
    --------------------
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    num_class = y_pred_prob.shape[1]
    
    # Calculate ROC curve and ROC area for each class
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_pred_prob[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Plot ROC curve for each class
    fig, ax = plt.subplots(figsize=(5, 5))
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(num_class), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Class {class_labels[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Plot chance line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (One-vs-All)')
    
    # Add legend
    ax.legend(loc="lower right")
    
    return fig

def _confusion_matrix_display(y_true, y_pred_prob, class_labels: List[str]) -> plt.Figure:
    """
    Plot confusion matrix for multi-class classification.
    
    --------------------
    Parameters
    ----------
    - `y_true` : array-like of shape (n_samples,)
        True labels.
    - `y_pred_prob` : array-like of shape (n_samples, n_classes)
        Probability predictions for each class.
    - `class_labels` : List of str
        Labels for each class.
    
    Returns
    -------
    - `fig` : matplotlib.figure.Figure
        Figure object with the confusion matrix plotted.
    --------------------
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, np.argmax(y_pred_prob, axis=1))
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    display.plot(cmap="Greens", values_format='d', ax=ax)
    
    ax.set_title("Confusion Matrix")
    
    return fig

def _classification_report_display(y_true, y_pred_prob, class_labels: List[str]) -> plt.Figure:
    """
    Plot classification report as a heatmap for multi-class classification.
    
    --------------------
    Parameters
    ----------
    - `y_true` : array-like of shape (n_samples,)
        True labels.
    - `y_pred_prob` : array-like of shape (n_samples, n_classes)
        Probability predictions for each class.
    - `class_labels` : List of str
        Labels for each class.
    
    Returns
    -------
    - `fig` : matplotlib.figure.Figure
        Figure object with the classification report plotted.
    --------------------
    """
    # Generate classification report
    report = classification_report(y_true, np.argmax(y_pred_prob, axis=1), output_dict=True, target_names=class_labels)
    df_report = pd.DataFrame(report).transpose()
    
    # Plot classification report as heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap='Greens', ax=ax)
    ax.set_title('Classification Report')
    
    return fig
