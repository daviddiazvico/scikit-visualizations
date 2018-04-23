"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.datasets import load_boston, load_iris
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV

from skvisualizations.validation import (classifier_scatter, regressor_scatter,
                                         metaparameter_plot)


def test_classifier_scatter():
    """Tests classifier scatter."""
    X, y = load_iris(return_X_y=True)
    classifier_scatter(X, y, 'classifier_scatter.pdf')


def test_regressor_scatter():
    """Tests regressor scatter."""
    X, y = load_boston(return_X_y=True)
    estimator = DummyRegressor()
    estimator.fit(X, y)
    regressor_scatter(X, y, estimator.predict(X), 'regressor_scatter.pdf')


def test_metaparameter_plot():
    """Tests metaparameter plot."""
    X, y = load_boston(return_X_y=True)
    estimator = GridSearchCV(DummyRegressor(strategy='constant'),
                             {'constant': [1.0, 2.0, 3.0]})
    estimator.fit(X, y)
    metaparameter_plot(estimator, 'constant', 'metaparameter_plot.pdf')
