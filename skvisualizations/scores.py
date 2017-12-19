"""
Scikit-learn-compatible visualizations for scores and hypothesis testing.

@author: David Diaz Vico
@license: MIT
"""

import itertools as it
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, wilcoxon
from statsmodels.sandbox.stats.multicomp import multipletests


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


def hypotheses_table(samples, models, alpha=0.05, test='wilcoxon',
                     correction=None, **test_args):
    """ Hypotheses table.

        Prints a hypothesis table with a selected test and correction.

        Parameters
        ----------
        samples: array-like
                 Matrix of samples where each column represent a model.
        models: array-like
                Model names.
        alpha: float in [0, 1], default=0.05
               Significance level.
        test: {'mannwhitneyu', 'wilcoxon'},
              default='wilcoxon'
              Ranking test used.
        correction: {'bonferroni', 'sidak', 'holm-sidak', 'holm',
                     'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh',
                     'fdr_tsbky'},
              default=None
              Method used to adjust the p-values.
        **test_args: dict
                     Optional ranking test arguments.

        Returns
        -------
        table: array-like
               Table of p-values and rejection/non-rejection for each
               hypothesis.
    """
    versus = list(it.combinations(range(len(models)), 2))
    comparisons = [models[vs[0]] + " vs " + models[vs[1]] for vs in versus]
    tests = {'mannwhitneyu': mannwhitneyu, 'wilcoxon': wilcoxon}
    pvalues = [tests[test](samples[:, vs[0]], samples[:, vs[1]], **test_args)[1] for vs in versus]
    if correction is not None:
        reject, pvalues, alphac_sidak, alphac_bonf = multipletests(pvalues,
                                                                   alpha,
                                                                   method=correction)
    else:
        reject = ['Rejected' if pvalue <= alpha else 'Not rejected' for pvalue in pvalues]
    table = pd.DataFrame(index=comparisons, columns=['p-value', 'Hypothesis'])
    for i, d in enumerate(comparisons):
        table.loc[d] = ['{0:.2f}'.format(pvalues[i]),
                        'Rejected' if reject[i] else 'Not rejected']
    return table
