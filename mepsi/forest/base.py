import numbers
import numpy as np

from typing import List, Set, Union
from abc import ABCMeta, abstractmethod

from sklearn.utils import check_random_state
from sklearn.base import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin


def _set_random_states(estimator, random_state=None):
    """Set fixed random_state parameters for an estimator.

    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.

    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.

    random_state : int or RandomState, default=None
        Pseudo-random number generator to control the generation of the random
        integers. Pass an int for reproducible output across multiple function
        calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:

        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)


class EnsembleSelector:
    def __init__(self):
        self.maskers_ = [True] * self.n_estimators

    def reset_estimator_masker(self, enable_estimators: Union[Set[BaseEstimator], List[BaseEstimator]] = None):
        """_summary_

        Parameters
        ----------
        enable_estimators : Union[Set[BaseEstimator], List[BaseEstimator]], optional
            _description_, by default None
        """
        if enable_estimators is None:
            self.maskers_ = [True] * self.n_estimators
        else:
            self.maskers_ = [False] * self.n_estimators
            for _estimator_idx in enable_estimators:
                self.maskers_[_estimator_idx] = True

    def _get_masked_estimators(self):
        return [tree for _idx, tree in enumerate(self.estimators_) if self.maskers_[_idx]]


class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, EnsembleSelector, metaclass=ABCMeta):
    """Base class for all ensemble classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    base_estimator : object
        The base estimator from which the ensemble is built.

    n_estimators : int, default=10
        The number of estimators in the ensemble.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    # overwrite _required_parameters from MetaEstimatorMixin
    _required_parameters: List[str] = []

    @abstractmethod
    def __init__(self, base_estimator, *, n_estimators=10, estimator_params=tuple()):
        # Set parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        super(BaseEnsemble, self).__init__()
        # Don't instantiate estimators now! Parameters of base_estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # self.estimators_ needs to be filled by the derived classes in fit.

    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute.

        Sets the base_estimator_` attributes.
        """
        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError("n_estimators must be an integer, " "got {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, " "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)

        estimator_params = {p: getattr(self, p) for p in self.estimator_params}

        if "ccp_alpha" in estimator_params and hasattr(estimator_params["ccp_alpha"], "__call__"):
            estimator_params["ccp_alpha"] = estimator_params["ccp_alpha"]()

        estimator.set_params(**estimator_params)

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Return the index'th estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        """Return iterator over estimators in the ensemble."""
        return iter(self.estimators_)
