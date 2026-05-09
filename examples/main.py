import os
import numpy as np

from tqdm import tqdm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from mepsi.forest import RandomForestClassifier
from mepsi.metric import TreeEditMetric, KappaStatisticsMetric
from mepsi.pruning.functions import (
    kappa_pruning,
    all_pruning,
    random_pruning,
    mepsi_pruning,
)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def train_forest(ensemble_size, Xtrain, Ytrain):
    rfc = RandomForestClassifier(
        n_estimators=ensemble_size,
        criterion="entropy",
        bootstrap=True,
        max_features="auto",
        ccp_alpha=0.2,
        random_state=129
    )
    rfc = rfc.fit(Xtrain, Ytrain)
    return rfc


if __name__ == "__main__":
    ensemble_size = 200
    subset_size = 20
    dataset = load_digits()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=129)

    records_ccpalpha = []
    rfc = train_forest(ensemble_size, Xtrain, Ytrain)
    pruning_list, score_test = all_pruning(rfc, ensemble_size, Xtest, Ytest)
    print("All score:", score_test)

    pruning_list, score_test = random_pruning(
        rfc, ensemble_size, subset_size, Xtest, Ytest
    )
    print("Random Score:", score_test)

    tree_edit = TreeEditMetric(rfc.estimators_)
    edit_dists = tree_edit.cal_pairwirse_measures(return_array=True)

    pruning_list, score_test = mepsi_pruning(
        rfc,
        ensemble_size,
        subset_size,
        Xtrain,
        Xtest,
        Ytrain,
        Ytest,
        predictions=rfc.predict_estimators(Xtrain),
        node_counts=np.array([float(rfc.estimators_[i].tree_.node_count) for i in range(ensemble_size)]),
        edit_dists=edit_dists,
    )

    print("MEPSI score:", score_test)

    kappa_metric = KappaStatisticsMetric(labels=Ytrain, inds_predict=np.array(rfc.predict_estimators(Xtrain)))
    kappa_statistics = kappa_metric.cal_pairwirse_measures(return_array=True)

    pruning_list, score_test = kappa_pruning(
        rfc,
        kappa_statistics,
        ensemble_size,
        subset_size,
        Xtest,
        Ytest,
    )

    print("Kappa Pruning Score:", score_test)
