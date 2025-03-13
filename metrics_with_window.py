import numpy as np
import matplotlib.pyplot as plt
import bisect
from sklearn.metrics import auc

def gather_act(activation, conf):
    x = 10

    right = 0
    gathering = False
    max_num = 0
    max_idx = 0

    act_pos = []

    for i in range(len(activation)):
        if gathering and i > right:
            act_pos.append(max_idx)
            gathering = False

        if activation[i] > conf:
            if not gathering:
                if i + x < len(activation):
                    right = i + x
                else:
                    right = len(activation) - 1
                gathering = True
                max_num = activation[i]
                max_idx = i
            elif activation[i] > max_num:
                max_num = activation[i]
                max_idx = i
    if gathering:
        act_pos.append(max_idx)

    return act_pos

def precision_recall_f1_with_window(y_true, y_pred, window_size=10, threshold=0.5):
    true_indices = np.where(y_true == 1)[0]
    pred_indices=gather_act(y_pred,threshold)
    #pred_indices = np.where(y_pred >= threshold)[0]
    
    matched_pred_indices = set()
    true_positive = 0

    for true_idx in true_indices:
        lower_bound = true_idx - window_size
        upper_bound = true_idx + window_size
        
        left = bisect.bisect_left(pred_indices, lower_bound)
        right = bisect.bisect_right(pred_indices, upper_bound)
        
        candidates = pred_indices[left:right]
        
        if len(candidates)>0:
            best_candidate = max(candidates, key=lambda x: y_pred[x])
            
            if best_candidate not in matched_pred_indices:
                true_positive += 1
                matched_pred_indices.add(best_candidate)
    
    predicted_positive = len(pred_indices)
    actual_positive = len(true_indices)
    
    precision = true_positive / predicted_positive if predicted_positive > 0 else 0
    recall = true_positive / actual_positive if actual_positive > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def precision_recall_curve_with_window(y_true, y_pred, window_size=10):
    thresholds = np.linspace(0, 1, 101)
    precisions, recalls, f1_scores = [], [], []
    
    for threshold in thresholds:
        precision, recall, f1 = precision_recall_f1_with_window(y_true, y_pred, window_size=window_size, threshold=threshold)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    return np.array(precisions), np.array(recalls), np.array(f1_scores)

def evaluate_motion_rhythm(y_true, y_pred, window_size=10, threshold=0.6, save_path='pr_curve.png'):
    B, T = y_true.shape
    
    total_precision, total_recall, total_f1 = 0, 0, 0
    
    for b in range(B):
        precision, recall, f1 = precision_recall_f1_with_window(y_true[b], y_pred[b], window_size=window_size, threshold=threshold)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    avg_precision = total_precision / B
    avg_recall = total_recall / B
    avg_f1 = total_f1 / B
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    precision_curve, recall_curve, f1_scores = precision_recall_curve_with_window(y_true_flat, y_pred_flat, window_size=window_size)
    pr_auc = auc(recall_curve, precision_curve)
    
    plt.figure()
    plt.plot(recall_curve, precision_curve, marker='.', label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Window Matching)')
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
    
    return avg_precision, avg_recall, avg_f1, pr_auc
