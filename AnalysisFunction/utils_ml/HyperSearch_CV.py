"""
2020/11/10 by Owen Yang


此文件包含三个寻参器的类：    
    1. 一个基类 ParamSearcherCV；
    2. 两个继承类 GridSearcherCV 和 RandSearcherCV
这三个类都是基于交叉验证的寻参器，基本同 sklearn.model_selection.BaseSearchCV 保持一致。


"""

import time
import numpy as np
import pandas as pd

from collections import defaultdict
from functools import partial
from joblib import Parallel, delayed, logger
from itertools import product

from numpy.ma import MaskedArray
from scipy.stats import rankdata


from sklearn.base import clone
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.model_selection._validation import _fit_and_score, _aggregate_score_dicts

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.model_selection._split import check_cv
from sklearn.utils.validation import indexable, check_is_fitted, _check_fit_params
from sklearn.metrics._scorer import _check_multimetric_scoring

from sklearn.model_selection import ParameterGrid, ParameterSampler


from .HyperSearch_base import ParamSearcher, GetDefaultRange
from .ML_assistant import dic2str



class ParamSearcherCV(ParamSearcher):
	"""
	Base class for cross-validation base searching.

	Important members are fit, predict.
	
	Implements a "fit" and a "score" method.
	It also implements "predict", "predict_proba", "decision_function",
	"transform" and "inverse_transform" if they are implemented in the
	estimator used.

	The parameters of the estimator used to apply these methods are optimized
	by cross-validated search over a param searching domain.


	Attributes
	----------

	cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

	"""
	@_deprecate_positional_args
	def __init__(
			self, 
			task_type = None,
			estimator = None,
			srch_domain = None,
			scoring = None,
			n_jobs = None,
			refit = True,
			verbose = 1,
			pre_dispatch = '2*n_jobs',
			error_score = np.nan,
			cv = 5,
			return_train_score = False,
		):
		super(ParamSearcherCV, self).__init__(
			task_type,
			estimator,
			srch_domain,
			scoring,
			n_jobs,
			pre_dispatch,
			refit,
			verbose,
			error_score,
		)
		self.cv = 5 
		self.return_train_score = return_train_score

		self.SetUp['cv'] = self.cv
		self.SetUp['return_train_score'] = self.return_train_score

		self.SearcherCV = None


	def __call__(self, X, Y=None):
		"""
			fit dataset and return best estimator
		"""
		self.fit(X, Y)
		return super(ParamSearcherCV, self).__call__()

	def score(self, X):
		"""
			Returns the score on the given data, if the estimator has been refit.
        	This uses the score defined by ``scoring`` where provided, and the
        	``best_estimator_.score`` method otherwise.

        	Parameters
        	----------
        	X : array-like of shape (n_samples, n_features)
            	Input data, where n_samples is the number of samples and
            	n_features is the number of features.
        	Returns
        	-------
        	score : float
        	"""
		self._check_is_fitted('score')
		if self.scorer_ is None:
			raise ValueError("No score function explicitly defined, "
							"and the estimator doesn't provide one %s"
							% self.best_estimator_)
		score = self.scorer_
		return score(self.best_estimator_, X)


	@_deprecate_positional_args
	def fit(self, X, y=None, *, groups=None, **fit_params):
		"""Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator
		"""
		estimator = self.estimator
		cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

		scorers, self.multimetric_ = _check_multimetric_scoring(self.estimator, scoring=self.scoring)

		if self.multimetric_:
			if self.refit is not False and (
				not isinstance(self.refit, str) or
				# This will work for both dict / list (tuple)
				self.refit not in scorers
			) and not callable(self.refit):
				raise ValueError("For multi-metric scoring, the parameter "
								"refit must be set to a scorer key or a "
								"callable to refit an estimator with the "
								"best parameter setting on the whole "
								"data and make the best_* attributes "
								"available for that metric. If this is "
								"not needed, refit should be set to "
								"False explicitly. %r was passed."
								% self.refit)
			else:
				refit_metric = self.refit
		else:
			refit_metric = 'score'

		X, y, groups = indexable(X, y, groups)
		fit_params = _check_fit_params(X, fit_params)

		n_splits = cv.get_n_splits(X, y, groups)

		base_estimator = clone(self.estimator)

		parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                            pre_dispatch=self.pre_dispatch)

		fit_and_score_kwargs = dict(scorer=scorers,
									fit_params=fit_params,
									return_train_score=self.return_train_score,
									return_n_test_samples=True,
									return_times=True,
									return_parameters=False,
									error_score=self.error_score,
									verbose=self.verbose)
		results = {}
		with parallel:
			all_candidate_params = []
			all_out = []

			def evaluate_candidates(candidate_params):
				candidate_params = list(candidate_params)
				n_candidates = len(candidate_params)

				if self.verbose > 0:
					print("Fitting {0} folds for each of {1} candidates,"
						" totalling {2} fits".format(n_splits, n_candidates, n_candidates * n_splits))

				out = parallel(delayed(_fit_and_score)(
									clone(base_estimator),
									X, y,
									train=train, test=test,
									parameters=parameters,
									**fit_and_score_kwargs
								) for parameters, (train, test) in 
								product(candidate_params, cv.split(X, y, groups))
							)
				print(len(out))

				if len(out) < 1:
					raise ValueError('No fits were performed. '
									'Was the CV iterator empty? '
									'Were there no candidates?')
				elif len(out) != n_candidates * n_splits:
					raise ValueError('cv.split and cv.get_n_splits returned '
									'inconsistent results. Expected {} '
									'splits, got {}'
									.format(n_splits, len(out) // n_candidates)
					)

				all_candidate_params.extend(candidate_params)
				all_out.extend(out)

				nonlocal results
				results = self._format_results(all_candidate_params, scorers, n_splits, all_out)
				return results

			self._run_search(evaluate_candidates)

		# For multi-metric evaluation, store the best_index_, best_params_ and
		# best_score_ iff refit is one of the scorer names
		# In single metric evaluation, refit_metric is "score"
		if self.refit or not self.multimetric_:
			# If callable, refit is expected to return the index of the best
			# parameter set.
			if callable(self.refit):
				self.best_index_ = self.refit(results)
				if not isinstance(self.best_index_, numbers.Integral):
					raise TypeError('best_index_ returned is not an integer')
				if (self.best_index_ < 0 or self.best_index_ >= len(results["params"])):
					raise IndexError('best_index_ index out of range')
			else:
				self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
				self.best_score_ = results["mean_test_%s" % refit_metric][self.best_index_]
			self.best_params_ = results["params"][self.best_index_]

		if self.refit:
			# we clone again after setting params in case some
			# of the params are estimators as well.
			self.best_estimator_ = clone(clone(base_estimator).set_params(**self.best_params_))
			refit_start_time = time.time()
			if y is not None:
				self.best_estimator_.fit(X, y, **fit_params)
			else:
				self.best_estimator_.fit(X, **fit_params)
			refit_end_time = time.time()
			self.refit_time_ = refit_end_time - refit_start_time

		# Store the only scorer not as a dict for single metric evaluation
		self.scorer_ = scorers if self.multimetric_ else scorers['score']

		self.results_ = results
		self.n_splits_ = n_splits

		return self

	def _format_results(self, candidate_params, scorers, n_splits, out):
		n_candidates = len(candidate_params)

		# if one choose to see train score, "out" will contain train score info
		if self.return_train_score:
			(train_score_dicts, test_score_dicts, test_sample_counts, fit_time, score_time) = zip(*out)
		else:
			(test_score_dicts, test_sample_counts, fit_time, score_time) = zip(*out)

		# test_score_dicts and train_score dicts are lists of dictionaries and
		# we make them into dict of lists
		test_scores = _aggregate_score_dicts(test_score_dicts)
		if self.return_train_score:
			train_scores = _aggregate_score_dicts(train_score_dicts)

		results = {}
		def _store(key_name, array, weights=None, splits=False, rank=False):
			"""A small helper to store the scores/times to the cv_results_"""
			# When iterated first by splits, then by parameters
			# We want `array` to have `n_candidates` rows and `n_splits` cols.
			array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
			if splits:
				for split_i in range(n_splits):
					# Uses closure to alter the results
					results["split%d_%s" % (split_i, key_name)] = array[:, split_i]

			array_means = np.average(array, axis=1, weights=weights)
			results['mean_%s' % key_name] = array_means
			# Weighted std is not directly available in numpy
			array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights))
			results['std_%s' % key_name] = array_stds

			if rank:
				results["rank_%s" % key_name] = np.asarray(rankdata(-array_means, method='min'), dtype=np.int32)

		_store('fit_time', fit_time)
		_store('score_time', score_time)
		# Use one MaskedArray and mask all the places where the param is not
		# applicable for that candidate. Use defaultdict as each candidate may
		# not contain all the params
		param_results = defaultdict(partial(MaskedArray,
											np.empty(n_candidates,),
											mask=True,
											dtype=object))
		for cand_i, params in enumerate(candidate_params):
			for name, value in params.items():
				# An all masked empty array gets created for the key
				# `"param_%s" % name` at the first occurrence of `name`.
				# Setting the value at an index also unmasks that index
				param_results["param_%s" % name][cand_i] = value

		results.update(param_results)
		# Store a list of param dicts at the key 'params'
		results['params'] = candidate_params

		# NOTE test_sample counts (weights) remain the same for all candidates
		test_sample_counts = np.array(test_sample_counts[:n_splits], dtype=np.int)

		for scorer_name in scorers.keys():
			# Computed the (weighted) mean and std for test scores alone
			_store('test_%s' % scorer_name, test_scores[scorer_name], splits=True, rank=True, weights=None)
			if self.return_train_score:
				_store('train_%s' % scorer_name, train_scores[scorer_name], splits=True)

		return results




class GridSearcherCV(ParamSearcherCV):
	"""
	The parameters of the estimator used to apply these methods are optimized
	by cross-validated grid-search over a parameter grid.

	"""
	def __init__(
			self,
			task_type,
			estimator,
			srch_domain = None,
			scoring = None,
			n_jobs = None,
			refit = True, 
			verbose = 1,
			pre_dispatch = '2*n_jobs', 
			error_score = np.nan,
			cv = None,
			return_train_score = False,
		):
		super(GridSearcherCV, self).__init__(
			task_type,
			estimator,
			srch_domain,
			scoring,
			n_jobs,
			pre_dispatch,
			refit,
			verbose,
			error_score,
			cv,
			return_train_score,
		)

		if self.srch_domain is None:
			self.srch_domain = GetDefaultRange(self.model_name, 'GridSearch')

	def __call__(self, X, Y):
		print('使用遍历化网格搜索对{}模型进行自动参数搜寻...'.format(self.model_name))
		return super(GridSearcherCV, self).__call__(X, Y)

	def _run_search(self, evaluate_candidates):
		evaluate_candidates(ParameterGrid(self.srch_domain))



class RandSearcherCV(ParamSearcherCV):
	"""
	The parameters of the estimator used to apply these methods are optimized
	by cross-validated randomized-search over a parameter domain.

	Attributes
	----------

	n_iter: integer, default=10
		Number of parameter settings that are sampled. n_iter trades
		off runtime vs quality of the solution.


	random_state: int or RandomState instance, default=None
		Pseudo random number generator state used for random uniform sampling
		from lists of possible values instead of scipy.stats distributions.

		Pass an int for reproducible output across multiple function calls.

	"""
	def __init__(
			self,
			task_type,
			estimator,
			srch_domain = None,
			scoring = None,
			n_jobs = None,
			refit = True, 
			verbose = 1,
			pre_dispatch = '2*n_jobs', 
			error_score = np.nan,
			cv = None,
			return_train_score = False,
			n_iter = 10,
			random_state = None,
		):
		super(RandSearcherCV, self).__init__(
			task_type,
			estimator,
			srch_domain,
			scoring,
			n_jobs,
			pre_dispatch,
			refit,
			verbose,
			error_score,
			cv,
			return_train_score,
		)
		self.n_iter = n_iter if n_iter is not None else 10
		self.random_state = random_state

		if self.srch_domain is None:
			self.srch_domain = GetDefaultRange(self.model_name, 'RandSearch')

	def __call__(self, X, Y):
		print('使用随机化搜索对{}模型进行自动参数搜寻...'.format(self.model_name))
		return super(RandSearcherCV, self).__call__(X, Y)

	def _run_search(self, evaluate_candidates):
		evaluate_candidates(ParameterSampler(
				self.srch_domain, 
				self.n_iter,
				random_state=self.random_state
			)
		)
