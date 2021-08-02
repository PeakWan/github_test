"""
2020/11/10 by Owen Yang

这一文件包含两部分:
	1. 所有模型的默认寻参范围
	2. 广义寻参器的基类


默认寻参范围：
	利用两个 dict 进行存储:
		1. GridDefaultRange: 用来进行网格遍历式搜索的参数空间
		2. RandDefaultRange: 用来进行随机式搜索的参数空间
	给定模型名称，其寻参范围将由函数 func::GetDefaultRange 返还



广义寻参器的基类：
	class::ParamSearcher
		广义搜索器的基类，由 sklearn.model_selection.BaseSearchCV 改写得到。
        额外增加了3个重要的类属性：
        	1. string::task_type: 任务类型，目前支持classification，regression，和clustering
        	2. string::model_name: 被寻参模型的模型名称
        	3. dict/list-dict::srch_domain: 寻参范围

"""


import pandas as pd

from abc import ABCMeta

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.utils.validation import check_is_fitted


from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.metaestimators import if_delegate_has_method

from .params import GridDefaultRange, RandDefaultRange
from .ML_assistant import *



"""
			Implemented with CV searcher:
                        'LogisticRegression':LogisticRegression(**kwargs),
                        'SVC':SVC(**kwargs),
                        'KNeighborsClassifier':KNeighborsClassifier(**kwargs),

                        'LinearRegression':LinearRegression(**kwargs),
                        'LinearSVR':LinearSVR(**kwargs),
                        'KNeighborsRegressor':KNeighborsRegressor(**kwargs),

			Implemented with Self searcher:
                        'KMeans':KMeans(**kwargs),
                        'Birch':Birch(**kwargs),
                        'SpectralClustering':SpectralClustering(**kwargs),
                        'AgglomerativeClustering':AgglomerativeClustering(**kwargs),
                        'GMM':GaussianMixture(**kwargs)


			Exist better solution:
						'Lasso' --> LassoCV
						'Ridge' --> RidgeCV
                        'LogisticRegression':LogisticRegression --> LogisticRegressionCV

                        'XGBRegressor':XGBRegressor(**kwargs) --> 
                        'XGBClassifier':XGBClassifier(**kwargs) --> 
                        'RandomForestRegressor':RandomForestRegressor(**kwargs) --> base_estimator_
                        'RandomForestClassifier':RandomForestClassifier(**kwargs) --> base_estimator_
"""



def GetDefaultRange(
		model_name, 
		search_name = 'RandSearch',
	):
	if search_name == 'GridSearch':
		return GridDefaultRange[model_name]

	if search_name == 'RandSearch':
		return RandDefaultRange[model_name]



class ParamSearcher(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
	"""
	2020/11/10 by Owen Yang

	A base class for hyperparam searcher.
	Rewritten from the sklearn.model_selection.BaseSearchCV class.


	Attributes
	----------
	task_type: string
		modeling task type
		Now includes ['classification', 'egression', 'clustering']
		(Maybe expand in the future.)

	estimator: estimator object.
		This is assumed to implement the scikit-learn estimator interface.
		Either estimator needs to provide a ``score`` function,
		or ``scoring`` must be passed.

	srch_domain: dict or list of dictionaries
		Dictionary with parameters names (`str`) as keys and lists of
		parameter settings to try as values, or a list of such
		dictionaries, in which case the grids spanned by each dictionary
		in the list are explored. This enables searching over any sequence
		of parameter settings.

		if None then would take default values from GetDefaultRange()

	model_name: string
		derived from self.estimator object.

	scoring: str, callable, list/tuple or dict, default=None
		A single str (see :ref:`scoring_parameter`) or a callable
		(see :ref:`scoring`) to evaluate the predictions on the test set.

		For evaluating multiple metrics, either give a list of (unique) strings
		or a dict with names as keys and callables as values.

		NOTE that when using custom scorers, each scorer should return a single
		value. Metric functions returning a list/array of values can be wrapped
		into multiple scorers that return one value each.

		See :ref:`multimetric_grid_search` for an example.

		If None, the estimator's score method is used.

	n_jobs : int, default=None
		Number of jobs to run in parallel.
		``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
		``-1`` means using all processors. See :term:`Glossary <n_jobs>` 
		for more details. 
		(``-1`` should be prevented from users, only for developer/administrator)

	pre_dispatch : int, or str, default=None
		Controls the number of jobs that get dispatched during parallel
		execution. Reducing this number can be useful to avoid an
		explosion of memory consumption when more jobs get dispatched
		than CPUs can process. This parameter can be:
			- None, in which case all the jobs are immediately
			  created and spawned. Use this for lightweight and
			  fast-running jobs, to avoid delays due to on-demand
			  spawning of the jobs
			- An int, giving the exact number of total jobs that are
			  spawned
			- A str, giving an expression as a function of n_jobs, as in '2*n_jobs'
		``None`` means n_jobs

	refit : bool, str, or callable, default=True
		Refit an estimator using the best found parameters on the whole dataset.
        
		For multiple metric evaluation, this needs to be a `str` denoting the
		scorer that would be used to find the best parameters for refitting
		the estimator at the end.
        
		Where there are considerations other than maximum score in
		choosing a best estimator, ``refit`` can be set to a function which
		returns the selected ``best_index_`` given ``cv_results_``. In that
		case, the ``best_estimator_`` and ``best_params_`` will be set
		according to the returned ``best_index_`` while the ``best_score_``
		attribute will not be available.
        
		The refitted estimator is made available at the ``best_estimator_``
		attribute and permits using ``predict`` directly on this
		``GridSearchCV`` instance.
        
		Also for multiple metric evaluation, the attributes ``best_index_``,
		``best_score_`` and ``best_params_`` will only be available if
		``refit`` is set and all of them will be determined w.r.t this specific
		scorer.
        
		See ``scoring`` parameter to know more about multiple metric evaluation.

	verbose : integer, default=None
		Controls the verbosity: the higher, the more messages.
		``None`` means 0

	error_score : 'raise' or numeric, default=None
		Value to assign to the score if an error occurs in estimator fitting.
		If set to 'raise', the error is raised. If a numeric value is given,
		FitFailedWarning is raised. This parameter does not affect the refit
		step, which will always raise the error.
		``None`` means np.nan.

	"""
	@_deprecate_positional_args
	def __init__(
			self, 
			task_type = None,
			estimator = None,
			srch_domain = None,
			scoring = None,
			n_jobs = None,
			pre_dispatch = None,
			refit = True,
			verbose = None,
			error_score = None,
		):
		assert task_type is not None
		self.task_type = task_type

		assert estimator is not None
		self.estimator = estimator
		self.model_name = estimator.__class__.__name__

		self.srch_domain = srch_domain
		self.scoring = scoring

		self.n_jobs = 1 if n_jobs is None else n_jobs
		self.pre_dispatch = self.n_jobs if pre_dispatch is None else pre_dispatch
		self.refit = True if refit is None else refit
		self.verbose = 0 if verbose is None else verbose
		self.error_score = np.nan if error_score is None else error_score
		
		self.SetUp = {
			'scoring' : self.scoring,
			'n_jobs': self.n_jobs,
			'pre_dispatch': self.pre_dispatch,
			'refit': self.refit,
			'verbose': self.verbose,
			'error_score': self.error_score,
		}

		self.best_estimator_ = None
		self.searched = False

	def __call__(self):
		print('{}模型参数自动搜索完成并已用最佳参数拟合模型，拟合过程耗时: {}秒。'.format(self.model_name, self.refit_time_))
		print('自动搜索得到的最佳{}参数集为:\n'.format(self.model_name) + dic2str(self.best_params_, self.model_name))
		return self.best_estimator_

	def report(self):
		print('{}模型参数自动搜索过程总结如下:'.format(self.model_name))
		print(pd.DataFrame(self.results_))

	def test_predict(self, X):
		self._check_is_fitted('test_predict')

		if self.task_type == 'Classfication':
			return self.predict_proba(X)
		elif self.task_type == 'Regression' or self.task_type == 'Clustering':
			return self.predict(X)
		else:
			raise NotImplementedError

	@property
	def _estimator_type(self):
		return self.estimator._estimator_type

	@property
	def _pairwise(self):
		# allows cross-validation to see 'precomputed' metrics
		return getattr(self.estimator, '_pairwise', False)

	def _check_is_fitted(self, method_name):
		if not self.refit:
			raise NotFittedError('This %s instance was initialized '
								'with refit=False. %s is '
								'available only after refitting on the best '
								'parameters. You can refit an estimator '
								'manually using the ``best_params_`` '
								'attribute'
								% (type(self).__name__, method_name))
		else:
			check_is_fitted(self)

	@if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
	def predict(self, X):
		"""Call predict on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
		self._check_is_fitted('predict')
		return self.best_estimator_.predict(X)

	@if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
	def predict_proba(self, X):
		"""Call predict_proba on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
		self._check_is_fitted('predict_proba')
		return self.best_estimator_.predict_proba(X)

	@if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
	def predict_log_proba(self, X):
		"""Call predict_log_proba on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
		self._check_is_fitted('predict_log_proba')
		return self.best_estimator_.predict_log_proba(X)

	@if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
	def decision_function(self, X):
		"""Call decision_function on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
		self._check_is_fitted('decision_function')
		return self.best_estimator_.decision_function(X)

	@if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
	def transform(self, X):
		"""Call transform on the estimator with the best found parameters.
        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
		self._check_is_fitted('transform')
		return self.best_estimator_.transform(X)

	@if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
	def inverse_transform(self, Xt):
		"""Call inverse_transform on the estimator with the best found params.
        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.
        Parameters
        ----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
		"""
		self._check_is_fitted('inverse_transform')
		return self.best_estimator_.inverse_transform(Xt)

	@property
	def n_features_in_(self):
		# For consistency with other estimators we raise a AttributeError so
		# that hasattr() fails if the search estimator isn't fitted.
		try:
			check_is_fitted(self)
		except NotFittedError as nfe:
			raise AttributeError(
				"{} object has no n_features_in_ attribute."
				.format(self.__class__.__name__)
			) from nfe

		return self.best_estimator_.n_features_in_

	@property
	def classes_(self):
		self._check_is_fitted("classes_")
		return self.best_estimator_.classes_


	def _run_search(self, evaluate_candidates):
		"""Repeatedly calls `evaluate_candidates` to conduct a search.
        This method, implemented in sub-classes, makes it possible to
        customize the the scheduling of evaluations: GridSearchCV and
        RandomizedSearchCV schedule evaluations for their whole parameter
        search space at once but other more sequential approaches are also
        possible: for instance is possible to iteratively schedule evaluations
        for new regions of the parameter search space based on previously
        collected evaluation results. This makes it possible to implement
        Bayesian optimization or more generally sequential model-based
        optimization by deriving from the BaseSearchCV abstract base class.
        Parameters
        ----------
        evaluate_candidates : callable
            This callback accepts a list of candidates, where each candidate is
            a dict of parameter settings. It returns a dict of all results so
            far, formatted like ``cv_results_``.
        Examples
        --------
        ::
            def _run_search(self, evaluate_candidates):
                'Try C=0.1 only if C=1 is better than C=10'
                all_results = evaluate_candidates([{'C': 1}, {'C': 10}])
                score = all_results['mean_test_score']
                if score[0] < score[1]:
                    evaluate_candidates([{'C': 0.1}])
        """
		raise NotImplementedError("_run_search not implemented.")

	@_deprecate_positional_args
	def fit(self, X):
		"""
    		Given data X, fit model with all possible param setup in self.srch_domain
    		then update/define self.best_estimator_
		"""
		raise NotImplementedError("fit not implemented.")


	def _format_results(self, candidate_params, scorer_name, out):
		raise NotImplementedError("_format_results not implemented.")





