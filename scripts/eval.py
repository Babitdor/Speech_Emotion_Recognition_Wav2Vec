import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(p: EvalPrediction):
    # If predictions are a tuple (logits, ...), take the first element
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # For classification, take the argmax to get predicted class indices
    preds = np.argmax(preds, axis=1)
    labels = p.label_ids
    # Compute accuracy
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
