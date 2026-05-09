from .simple import RandomSelector, AllSelector
from .mepsi import MEPSISelector
from .kappa_pruning import KappaPruning


def all_pruning(rfc, ensemble_size, Xtest, Ytest):
    pruning_algo = AllSelector(ensemble_size)
    all_list = pruning_algo.pruning()
    score_test = rfc.score(Xtest, Ytest)

    return all_list, score_test


def random_pruning(rfc, ensemble_size, subset_size, Xtest, Ytest):
    pruning_algo = RandomSelector(ensemble_size)
    sample_list = pruning_algo.pruning(subset_size)

    rfc.reset_estimator_masker(sample_list)
    score_test = rfc.score(Xtest, Ytest)
    rfc.reset_estimator_masker(None)

    return sample_list, score_test


def mepsi_pruning(rfc, ensemble_size, subset_size, Xtrain, Xtest, Ytrain, Ytest, predictions, node_counts, edit_dists):
    pruning_algo = MEPSISelector(ensemble_size, Xtrain, Ytrain, predictions, node_counts, edit_dists)
    pruning_list = pruning_algo.pruning(size=subset_size)
    rfc.reset_estimator_masker(pruning_list)
    score_test = rfc.score(Xtest, Ytest)
    rfc.reset_estimator_masker(None)

    return pruning_list, score_test


def kappa_pruning(rfc, kappa_statistics, ensemble_size, subset_size, Xtest, Ytest):
    pruning_algo = KappaPruning(ensemble_size)
    pruning_list = pruning_algo.pruning(kappa_statistics, subset_size)

    rfc.reset_estimator_masker(pruning_list)
    score_test = rfc.score(Xtest, Ytest)
    rfc.reset_estimator_masker(None)

    return pruning_list, score_test
