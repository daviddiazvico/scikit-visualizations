"""
Scikit-learn-compatible visualizations for scores and hypothesis testing.
Friedman and Holm tests based on https://github.com/citiususc/stac.

@author: David Diaz Vico
@license: MIT
"""

import itertools as it
import numpy as np
import pandas as pd
from scipy.stats import f, mannwhitneyu, norm, rankdata, wilcoxon


def scores_table(datasets, estimators, scores, stds=None,
                 greater_is_better=True, method='average'):
    """ Scores table.

        Prints a table where each row represents a dataset and each column
        represents an estimator.

        Parameters
        ----------
        datasets: array-like
                  List of dataset names.
        estimators: array-like
                    List of estimator names.
        scores: array-like
                Matrix of scores where each column represents a model.
        stds: array_like, default=None
              Matrix of standard deviations where each column represents a
              model.
        greater_is_better: boolean, default=True
                           Whether a greater score is better (score) or worse
                           (loss).
        method: {'average', 'min', 'max', 'dense', 'ordinal'}, default='average'
                Method used to solve ties.

        Returns
        -------
        table: array-like
               Table of mean and standard deviation of each estimator-dataset
               pair. A ranking of estimators is also generated.
    """
    ranks = np.asarray([rankdata(-m, method=method) if greater_is_better else rankdata(m, method=method) for m in scores])
    table = pd.DataFrame(data=scores, index=datasets, columns=estimators)
    for i, d in enumerate(datasets):
        for j, e in enumerate(estimators):
            table.loc[d, e] = '{0:.2f}'.format(scores[i, j])
            if stds is not None:
                table.loc[d, e] += ' ±{0:.2f}'.format(stds[i, j])
            table.loc[d, e] += ' ({0:.1f})'.format(ranks[i, j])
    table.loc['rank mean'] = np.around(np.mean(ranks, axis=0), decimals=4)
    return table


def friedman_test(samples, greater_is_better=True, method='average'):
    """ Friedman ranking test.

        Tests the hypothesis thtat in a set of dependent sample models, at least
        two of the models represent populations with different median values.

        Parameters
        ----------
        samples: array-like
                 Matrix of samples where each column represents a model.
        greater_is_better: boolean, default=True
                           Whether a greater score is better (score) or worse
                           (loss).
        method: {'average', 'min', 'max', 'dense', 'ordinal'}, default='average'
                Method used to solve ties.

        Returns
        -------
        fvalue: float
                F-value of the test.
        pvalue: float
                p-value from the F-distribution.
        ranks: array-like
               Ranking for each model.
        pivots: array-like
                Pivotal quantities for each model.

        References
        ----------
        M. Friedman, The use of ranks to avoid the assumption of normality
                     implicit in the analysis of variance, Journal of the
                     American Statistical Association 32 (1937) 674-701.
        D.J. Sheskin, Handbook of parametric and nonparametric statistical
                      procedures. crc Press, 2003, Test 25: The Friedman Two-Way
                      Analysis of Variance by Ranks
    """
    ranks = np.asarray([rankdata(-s, method=method) if greater_is_better else rankdata(s, method=method) for s in samples])
    ranks = np.nanmean(ranks, axis=0)
    (n, k) = samples.shape
    pivots = [r / np.sqrt(k * (k + 1) / (6. * n)) for r in ranks]
    chi2 = ((12 * n) / float((k * (k + 1)))) * ((np.sum(r ** 2 for r in ranks)) - ((k * (k + 1) ** 2) / float(4)))
    fvalue = ((n - 1) * chi2) / float((n * (k - 1) - chi2))
    pvalue = 1 - f.cdf(fvalue, k - 1, (k - 1) * (n - 1))
    return fvalue, pvalue, ranks, pivots


def holm_multitest(models, pivots):
    """ Holm post-hoc test using the pivotal quantities obtained by a ranking
        test.

        Tests the hypothesis that the ranking of each pair of models are
        different.

        Parameters
        ----------
        models: array-like
                Model names.
        pivots: array-like
                Model pivotal quantities.

        Returns
        -------
        comparisons: array-like
                     Strings identifier of each comparison with format 'model_i
                     vs model_j'.
        pvalues: array-like
                 The associated adjusted p-values which can be compared with a
                 significance level.

        References
        ----------
        O.J. S. Holm, A simple sequentially rejective multiple test procedure,
                      Scandinavian Journal of Statistics 6 (1979) 65-70.
    """
    ranks = dict(zip(models, pivots))
    k = len(models)
    keys = list(ranks.keys())
    versus = list(it.combinations(range(k), 2))
    comparisons = [keys[vs[0]] + " vs " + keys[vs[1]] for vs in versus]
    zvalues = [abs(ranks[keys[vs[0]]] - ranks[keys[vs[1]]]) for vs in versus]
    pvalues = [2 * (1 - norm.cdf(abs(z))) for z in zvalues]
    pvalues, zvalues, comparisons = map(list, zip(*sorted(zip(pvalues, zvalues,
                                                              comparisons),
                                                          key=lambda t: t[0])))
    m = int(k * (k - 1) / 2.)
    adjpvalues = [min(max((m - j) * pvalues[j] for j in range(i + 1)),
                      1) for i in range(m)]
    return comparisons, adjpvalues


def wilcoxon_multitest(models, samples, zero_method='wilcox', correction=False):
    """ Post-hoc test using the pivotal quantities obtained by a Wilcoxon
        signed rank test.

        Tests the hypothesis that the scores of each pair of models come from
        different distributions.

        Parameters
        ----------
        models: array-like
                Model names.
        samples: array-like
                 Matrix of samples where each column represents a model.
        zero_method: {'pratt', 'wilcox', 'zsplit'}, default: 'wilcox'
                     'pratt': Pratt treatment: includes zero-differences in the
                                               ranking process (more
                                               conservative).
                     'wilcox': Wilcox treatment: discards all zero-differences.
                     'zsplit': Zero rank split: just like Pratt, but spliting
                               the zero rank between positive and negative ones.
        correction: bool, default=False
                    If True, apply continuity correction by adjusting the
                    Wilcoxon rank statistic by 0.5 towards the mean value when
                    computing the z-statistic.

        Returns
        -------
        comparisons: array-like
                     Strings identifier of each comparison with format 'model_i
                     vs model_j'.
        pvalues: array-like
                 The associated p-values which can be compared with a
                 significance level.
    """
    versus = list(it.combinations(range(len(models)), 2))
    comparisons = [models[vs[0]] + " vs " + models[vs[1]] for vs in versus]
    pvalues = [wilcoxon(samples[:, vs[0]], samples[:, vs[1]],
                        zero_method=zero_method, correction=correction)[1] for vs in versus]
    return comparisons, pvalues


def mannwhitneyu_multitest(models, samples, use_continuity=True,
                           alternative=None):
    """ Post-hoc test using the pivotal quantities obtained by a Mann-Whitney
        rank test.

        Tests the hypothesis that the scores of each pair of models come from
        different distributions.

        Parameters
        ----------
        models: array-like
                Model names.
        samples: array-like
                 Matrix of samples where each column represents a model.
        use_continuity: bool, default=True
                        Whether a continuity correction (1/2.) should be taken
                        into account.
        alternative: {None (deprecated), 'less', 'two-sided', 'greater'},
                     default=None
                     Whether to get the p-value for the one-sided hypothesis
                     ('less' or 'greater') or for the two-sided hypothesis
                     ('two-sided'). Defaults to None, which results in a p-value
                     half the size of the 'two-sided' p-value and a different U
                     statistic. The default behavior is not the same as using
                     'less' or 'greater': it only exists for backward
                     compatibility and is deprecated.

        Returns
        -------
        comparisons: array-like
                     Strings identifier of each comparison with format 'model_i
                     vs model_j'.
        pvalues: array-like
                 The associated p-values which can be compared with a
                 significance level.
    """
    versus = list(it.combinations(range(len(models)), 2))
    comparisons = [models[vs[0]] + " vs " + models[vs[1]] for vs in versus]
    pvalues = [mannwhitneyu(samples[:, vs[0]], samples[:, vs[1]],
                            use_continuity=use_continuity,
                            alternative=alternative)[1] for vs in versus]
    return comparisons, pvalues


def hypotheses_table(samples, models, alpha=0.05, test='friedman-holm',
                     **test_args):
    """ Hypotheses table.

        Prints a table where each row represents the hypothesis that the
        ranking of each pair of models is different. Uses Friedman ranking test
        and Holm post-hoc multitest.

        Parameters
        ----------
        samples: array-like
                 Matrix of samples where each column represent a model.
        models: array-like
                Model names.
        alpha: float in [0, 1], default=0.05
               Significance level.
        test: {'friedman-holm', 'wilcoxon'. 'mannwhitneyu'},
              default='friedman-holm'
              Ranking test used.
        **test_args: dict
                     Optional ranking test arguments.

        Returns
        -------
        table: array-like
               Table of p-values and rejection/non-rejection for each
               hypothesis.
    """
    if test == 'friedman-holm':
        fvalue, pvalue, ranks, pivots = friedman_test(samples, **test_args)
        comparisons, pvalues = holm_multitest(models, pivots)
    elif test == 'wilcoxon':
        comparisons, pvalues = wilcoxon_multitest(models, samples, **test_args)
    elif test == 'mannwhitneyu':
        comparisons, pvalues = mannwhitneyu_multitest(models, samples,
                                                      **test_args)
    table = pd.DataFrame(index=comparisons, columns=['p-value', 'Hypothesis'])
    for i, d in enumerate(comparisons):
        table.loc[d] = ['{0:.2f}'.format(pvalues[i]),
                        'Rejected' if pvalues[i] < alpha else 'Not rejected']
    return table
