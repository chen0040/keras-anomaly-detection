from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score


def report_evaluation_metrics(y_true, y_pred):
    average_precision = average_precision_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    recall = recall_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    f1 = f1_score(y_true, y_pred, labels=[0, 1], pos_label=1)

    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Precision: {0:0.2f}'.format(precision))
    print('Recall: {0:0.2f}'.format(recall))
    print('F1: {0:0.2f}'.format(f1))
