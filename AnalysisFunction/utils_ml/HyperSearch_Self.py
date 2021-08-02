"""
2020/11/19 by Owen Yang


此文件包含三个寻参器的类：    
    1. 一个基类 ParamSearcherSelf；
    2. 两个继承类 GridSearcherSelf 和 RandSearcherSelf
不同于基于交叉验证的模型选择，这一系列寻参器根据单独给定的评分标准进行模型选择。
这一寻参器类型专门为非监督式学习设计（比如聚类）。
当前支持的评分标准有(待扩展)：
    1. 轮廓系数(Silhouette score): 添加时间2020/11/07


具体寻参器类信息如下：

    class::ParamSearcherSelf 
        基于给定选择标准的基类搜索器，由 sklearn.model_selection.BaseSearchCV 改写得到。
        寻参过程利用 joblib 库进行并行计算。

    class::GridSearcherSelf
        网格遍历式搜索器
    
    class::RandSearcherSelf
        随机式搜索器


辅助函数：
    func::_fit_and_score
        从 sklearn.model_selection._validation._fit_and_score 改写得到。
        其中 scorer 被替换为 self_scorer，具体信息见下方。

"""


import time
import numpy as np
import pandas as pd

from collections import defaultdict
from functools import partial
from joblib import Parallel, delayed, logger

from numpy.ma import MaskedArray
from scipy.stats import rankdata

from sklearn.base import clone
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.model_selection import ParameterGrid, ParameterSampler

from sklearn.metrics import silhouette_score


from .HyperSearch_base import ParamSearcher, GetDefaultRange
from .ML_assistant import dic2str



def _fit_and_score(
        estimator, X, 
        self_scorer, 
        parameters, 
        verbose, 
        error_score=np.nan,
        return_times=False, 
    ):

    """
    Rewritten from sklearn.model_selection._validation._fit_and_score

    Fit estimator and compute scores for a given dataset especially for clustering.
    No cross-validation implemented, therefore self_scorer should be different from
    the score used in model training.


    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to clustering.

    self_scorer : A single callable (or a dict mapping scorer name to the callable)
        Currently only single callable implemented.
        If it is a single callable, the return value for ``score`` is a single float.

        (Later consider dict of scores.
         For a dict, it should be one mapping the scorer name to the scorer
         callable object / function.
        )

        The callable object / fn should have signature ``self_scorer(x, y)``.

    verbose : int
        The verbosity level.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.

    parameters : dict or None
        Parameters to be set on the estimator.

    return_times : bool, default=False
        Whether to return the fit/score times.


    Returns
    -------
    train_scores : dict of scorer name -> float
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.
    """

    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s'%(k, v) for k, v in parameters.items()))
        print("[Fit&Score] %s %s" % (msg, (64 - len(msg)) * '.'))

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)
        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    if (estimator.__class__.__name__ in ['SpectralClustering', 'AgglomerativeClustering']):
        labels = estimator.fit_predict(X)
    else:
        estimator.fit(X)
        labels = estimator.predict(X)

    fit_time = time.time() - start_time
    score = self_scorer(X, labels)
    score_time = time.time() - start_time - fit_time

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[Fit&Score] END "
        result_msg = ""
        if verbose > 2: 
            result_msg += f"score={score:.3f})"
        result_msg += f" total time={logger.short_format_time(total_time)}"

    ret = [score]
    if return_times:
        ret.extend([fit_time, score_time])
    return ret



class ParamSearcherSelf(ParamSearcher):
    """
    Base class for hyper parameter search in unsupervised learning.
    Currently using silouette score as selection criteria.


    Attributes
    -----------
    scoring: str, callable, list/tuple or dict, default=None
        If None, using silouette score.
        (Note: this attribute is overwritten from class::ParamSearcher)

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
        ):
        super(ParamSearcherSelf, self).__init__(
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
        self.scoring = silhouette_score if scoring is None else scoring

    def __call__(self, X):
        self.fit(X)
        return super(ParamSearcherSelf, self).__call__()

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
        compared = self.best_estimator_.predict(X)
        return score(X, compared)


    @_deprecate_positional_args
    def fit(self, X):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        future:
           **fit_params : dict of str -> object
               Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, 
                            verbose=self.verbose,
                            pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(self_scorer=self.scoring,
                                    return_times=True,
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
                    print("Fitting for each of {} candidates".format(n_candidates))

                out = parallel(delayed(_fit_and_score)(
                        clone(base_estimator), X,
                        parameters=parameters,
                        **fit_and_score_kwargs,
                    ) for parameters in candidate_params
                )

                if len(out) < 1:
                    raise ValueError('No fits were performed. Were there no candidates?')
                elif len(out) != n_candidates :
                    raise ValueError('inconsistent results. Expected {} '
                                     'runs, got {}'.format(n_candidates, len(out)))

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                nonlocal results
                results = self._format_results(all_candidate_params, 'silhouette_score', all_out)
                return results

            self._run_search(evaluate_candidates)

        self.best_index_ = results["rank_silhouette_score"].argmin()
        self.best_score_ = results['silhouette_score'][self.best_index_]
        self.best_params_= results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some of the params are estimators as well.
            self.best_estimator_ = clone(clone(base_estimator).set_params(**self.best_params_))

            refit_start_time = time.time()
            self.best_estimator_.fit(X)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        self.scorer_ = self.scoring
        self.results_ = results
        return self


    def _format_results(self, candidate_params, scorer_name, out):
        n_candidates = len(candidate_params)
        (score, fit_time, score_time) = zip(*out)

        results = {}

        def _store(key_name, array, rank=False):
            array = np.array(array, dtype=np.float64)
            tot = len(array) + 1
            results[key_name] = array
            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    tot - rankdata(array, method='min'), dtype=np.int32,
                )

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

        _store(scorer_name, score, rank=True)
        return results




class GridSearcherSelf(ParamSearcherSelf):
    """
    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid using silhouette_score.

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
        ):
        super(GridSearcherSelf, self).__init__(
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
        if self.srch_domain is None:
            self.srch_domain = GetDefaultRange(self.model_name, 'GridSearch')

    def __call__(self, X):
        print('使用遍历化网格搜索对{}模型进行自动参数搜寻...'.format(self.model_name))
        return super(GridSearcherSelf, self).__call__(X)

    def _run_search(self, evaluate_candidates):
        evaluate_candidates(ParameterGrid(self.srch_domain))



class RandSearcherSelf(ParamSearcherSelf):
    """
    The parameters of the estimator used to apply these methods are optimized
    by randomized-search over a parameter domain using silhouette_score.


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
            srch_domain=None,
            scoring = None, 
            n_jobs = None,
            refit = True, 
            verbose = 1,
            pre_dispatch = '2*n_jobs', 
            error_score = np.nan,
            n_iter = 10,
            random_state = None,
        ):
        super(RandSearcherSelf, self).__init__(
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
        self.n_iter = n_iter if n_iter is not None else 10
        self.random_state = random_state

        if self.srch_domain is None:
            self.srch_domain = GetDefaultRange(self.model_name, 'RandSearch')

    def __call__(self, X):
        print('使用随机化搜索对{}模型进行自动参数搜寻...'.format(self.model_name))
        return super(RandSearcherSelf, self).__call__(X)

    def _run_search(self, evaluate_candidates):
        evaluate_candidates(ParameterSampler(
                self.srch_domain, 
                self.n_iter,
                random_state=self.random_state,
            )
        )
