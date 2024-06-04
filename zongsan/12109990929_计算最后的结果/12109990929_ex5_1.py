import numpy as np

# 给定的混淆矩阵数据 (移除背景类)
confusion_matrix = np.array([
    [0.90, 0.10],  # mask (实际)
    [0.95, 0.05]   # nomask (实际)
])

# 计算 TP, FP, FN, TN
TP_mask = confusion_matrix[0, 0]
FP_mask = confusion_matrix[1, 0]
FN_mask = confusion_matrix[0, 1]
TN_mask = confusion_matrix[1, 1]

TP_nomask = confusion_matrix[1, 1]
FP_nomask = confusion_matrix[0, 1]
FN_nomask = confusion_matrix[1, 0]
TN_nomask = confusion_matrix[0, 0]

# 计算 Precision, Recall, F1 Score
def calculate_metrics(TP, FP, FN, TN):
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1_score

# mask 类别
precision_mask, recall_mask, f1_score_mask = calculate_metrics(TP_mask, FP_mask, FN_mask, TN_mask)
print(f"mask 类别 - TP: {TP_mask}, FP: {FP_mask}, FN: {FN_mask}, TN: {TN_mask}")
print(f"mask 类别 - Precision: {precision_mask:.3f}, Recall: {recall_mask:.3f}, F1 Score: {f1_score_mask:.3f}")

# nomask 类别
precision_nomask, recall_nomask, f1_score_nomask = calculate_metrics(TP_nomask, FP_nomask, FN_nomask, TN_nomask)
print(f"nomask 类别 - TP: {TP_nomask}, FP: {FP_nomask}, FN: {FN_nomask}, TN: {TN_nomask}")
print(f"nomask 类别 - Precision: {precision_nomask:.3f}, Recall: {recall_nomask:.3f}, F1 Score: {f1_score_nomask:.3f}")
