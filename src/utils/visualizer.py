from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    y_true: np.array,
    y_pred: np.array
) -> sns.heatmap:
    
    classes = [0, 1]
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    
    return sns.heatmap(df_cm, annot=True).get_figure()