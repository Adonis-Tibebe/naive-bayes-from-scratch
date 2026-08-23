import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_confusion_matrix(y_true, y_pred, classes=None):
    """
    Computes an N x N confusion matrix where rows represent actual classes 
    and columns represent predicted classes.
    
    Args:
        y_true (array-like): Ground truth target labels.
        y_pred (array-like): Predicted target labels from the model.
        classes (array-like, optional): Explicit list of class labels.
        
    Returns:
        tuple: (confusion_matrix (np.ndarray), classes (np.ndarray))
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    else:
        classes = np.array(classes)
        
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Map class labels to matrix indices
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for actual, predicted in zip(y_true, y_pred):
        if actual in class_to_idx and predicted in class_to_idx:
            i = class_to_idx[actual]
            j = class_to_idx[predicted]
            cm[i, j] += 1
            
    return cm, classes


def compute_accuracy(y_true, y_pred):
    """
    Computes overall classification accuracy.
    Formula: Accuracy = (Correct Predictions) / (Total Predictions)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return np.mean(y_true == y_pred)


def compute_classification_metrics(y_true, y_pred, classes=None):
    """
    Computes per-class Precision, Recall, and F1-Score alongside 
    Macro-averaged summary metrics using the confusion matrix.
    
    Formulas per class (c):
        True Positives (TP)  = CM[c, c]
        False Positives (FP) = sum(CM[:, c]) - TP
        False Negatives (FN) = sum(CM[c, :]) - TP
        
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
        
    Returns:
        dict: Structured dictionary containing per-class and macro metrics.
    """
    cm, classes = compute_confusion_matrix(y_true, y_pred, classes=classes)
    
    per_class_metrics = {}
    precisions = []
    recalls = []
    f1_scores = []
    
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        # Guard against division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": int(cm[i, :].sum())
        }
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
    # Compute Macro-Averages (unweighted mean across all classes)
    macro_metrics = {
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores))
    }
    
    return {
        "confusion_matrix": cm,
        "classes": classes,
        "per_class": per_class_metrics,
        "macro": macro_metrics,
        "accuracy": compute_accuracy(y_true, y_pred)
    }

    

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap='Blues', save_path=None):
    """
    Visualizes the computed confusion matrix as a color-coded heatmap.
    
    Args:
        cm (np.ndarray): The raw confusion matrix array.
        classes (list/array): The ordered list of class labels.
        title (str): The title of the plot.
        cmap (str): The matplotlib colormap to use (default 'Blues').
        save_path (str, optional): If provided, saves the figure to disk.
    """
    plt.figure(figsize=(8, 6))
    
    # Create the heatmap using seaborn for clean styling
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14})
    
    # Add descriptive labels
    plt.title(title, fontsize=16, pad=15)
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Visualized confusion matrix saved to {save_path}")
        
    plt.show()