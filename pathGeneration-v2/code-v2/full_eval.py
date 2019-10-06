import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, \
    roc_auc_score, precision_score, recall_score


thres = 0.5

def f1_score(preds, labels, thres, average='micro'):
    '''Returns (precision, recall, F1 score) from a batch of predictions (thresholded probabilities)
       given a batch of labels (for macro-averaging across batches)'''
    #preds = (probs >= thres).astype(np.int32)
    # print('probs:',probs)
    # print('labels:',labels)
    # print('preds:',preds)
    #preds=probs
    # print(preds)
    # print(labels)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=average,
                                                                 warn_for=())
    return p, r, f

def auc_pr(probs, labels, average='micro'):
    '''Precision integrated over all thresholds (area under the precision-recall curve)'''
    if average == 'macro' or average is None:
        sums = labels.sum(0)
        nz_indices = np.logical_and(sums != labels.shape[0], sums != 0)
        probs = probs[:, nz_indices]
        labels = labels[:, nz_indices]
    return average_precision_score(labels, probs, average=average)


def auc_roc(probs, labels, average='micro'):
    '''Area under the ROC curve'''
    if average == 'macro' or average is None:
        sums = labels.sum(0)
        nz_indices = np.logical_and(sums != labels.shape[0], sums != 0)
        probs = probs[:, nz_indices]
        labels = labels[:, nz_indices]
    # print('labels:',labels)
    # print('probs:',probs)
    return roc_auc_score(labels, probs, average=average)


def precision_at_k(probs, labels, k, average='micro'):
    indices = np.argpartition(-probs, k-1, axis=1)[:, :k]
    preds = np.zeros(probs.shape, dtype=np.int)
    preds[np.arange(preds.shape[0])[:, np.newaxis], indices] = 1
    return precision_score(labels, preds, average=average)


def recall_at_k(probs, labels, k, average='micro'):
    indices = np.argpartition(-probs, k-1, axis=1)[:, :k]
    preds = np.zeros(probs.shape, dtype=np.int)
    preds[np.arange(preds.shape[0])[:, np.newaxis], indices] = 1
    return recall_score(labels, preds, average=average)


def full_evaluate(pred,probs, gold, thres=0.5):
    # pred = np.array(pred)
    # gold = np.array(gold)
    #print(pred)
    micro_p, micro_r, micro_f1 = f1_score(pred, gold, thres, average='micro')
    macro_p,macro_r,macro_f1= f1_score(pred, gold, thres, average='macro')

    # micro_auc_pr= auc_pr(pred, gold, average='micro')
    # macro_auc_pr= auc_pr(pred, gold, average='macro')

    micro_auc_roc= auc_roc(pred, gold, average='micro')
    macro_auc_roc= auc_roc(pred, gold, average='macro')

    precision_8= precision_at_k(probs, gold, 8, average='micro')
    precision_40= precision_at_k(probs, gold, 40, average='micro')

    recall_8= recall_at_k(probs, gold, 8, average='micro')
    recall_40=recall_at_k(probs, gold, 40, average='micro')
    
    return micro_p,macro_p,micro_r,macro_r,micro_f1,macro_f1,micro_auc_roc,macro_auc_roc,precision_8,precision_40,recall_8,recall_40


def jaccrad(predList, referList):  # terms_reference为源句子，terms_model为候选句子
    grams_reference = set(predList)  # 去重；如果不需要就改为list
    grams_model = set(referList)
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp  # 并集
    jaccard_coefficient = temp*1.0 / fenmu  # 交集
    return jaccard_coefficient