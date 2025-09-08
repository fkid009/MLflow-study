from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(y_te, y_pred, average):
    acc = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, average=average)
    recall = recall_score(y_te, y_pred, average=average)
    f1 = f1_score(y_te, y_pred, average=average)
    return acc, precision, recall, f1