import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np

recall_level_default = 0.80

def print_measures(type, auroc, aupr, fpr, recall_level=recall_level_default):
    print('\t\t\t\t' + type)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def get_and_print_results(type, in_score, out_score):
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(out_score, in_score)
    aurocs.append(measures[0])
    auprs.append(measures[1])
    fprs.append(measures[2])

    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)

    print_measures(type, auroc, aupr, fpr)

    return measures



def evaluate_metrics(type, ood, id):
    get_and_print_results(type, ood, id)


def evalIDvsOOD(result_folder):
    if not (os.path.isdir(result_folder)):
        try:
            os.mkdir(result_folder)
        except OSError:
            print("Creation of the directory %s failed" % result_folder)

    ids_file_name = result_folder + '/ids.npz'
    oods_file_name = result_folder + '/oods.npz'

    if os.path.isfile(ids_file_name):
        saved_ids = np.load(ids_file_name)
        disagreement_id = saved_ids['disagreements']
        exp_lls_id = saved_ids['exp_lls']
        entropies_id = saved_ids['entropies']
        waics_id = saved_ids['waics']
        variations_in_lls_id = saved_ids['stds_lls']
        typicality_ids = saved_ids['typicalities']
    else:
        print('no IDs file is found')
        return -1

    if os.path.isfile(oods_file_name):
        saved_oods = np.load(oods_file_name)
        disagreement_ood = saved_oods['disagreements']
        exp_lls_ood = saved_oods['exp_lls']
        entropies_ood = saved_oods['entropies']
        waics_ood = saved_oods['waics']
        variations_in_lls_ood = saved_oods['stds_lls']
        typicality_oods = saved_oods['typicalities']
    else:
        print('no OoDs file is found')
        return -1


    evaluate_metrics("Disagreement Score", disagreement_ood, disagreement_id)
    evaluate_metrics("Expected LL", exp_lls_ood, exp_lls_id)
    evaluate_metrics("Entropies", entropies_ood, entropies_id)
    evaluate_metrics("WAICs", waics_ood, waics_id)
    evaluate_metrics("Stdandard deviations of LLs", variations_in_lls_id, variations_in_lls_ood)
    evaluate_metrics("Typicality", typicality_ids, typicality_oods)

