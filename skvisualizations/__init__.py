"""
Scikit-learn-compatible visualizations.

@author: David Diaz Vico
@license: MIT
"""

from .scores import (scores_table, friedman_test, holm_multitest,
                     hypotheses_table)
from .validation import (classifier_scatter, regressor_scatter,
                         metaparameter_plot, keras_history_plot)
