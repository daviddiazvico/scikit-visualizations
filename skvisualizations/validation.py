"""
Scikit-learn-compatible visualizations for model validation.

@author: David Diaz Vico
@license: MIT
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


sns.set(style="white", palette="muted", color_codes=True)


def classifier_scatter(X, y, pca_n_components=2):
    """ Classifier scatter.

        Classifier scatter plot.

        Parameters
        ----------
        X: array-like, shape (n_samples, features_shape)
           The transformed data.
        y: numpy array of shape [n_samples]
           Target values.
        pca_n_components: integer, default=2
                          Dimension of the PCA projection of X.

        Returns
        -------
        None.
    """
    names = list(range(X.shape[1]))
    if pca_n_components is not None:
        names = list(range(pca_n_components))
        X = PCA(n_components=pca_n_components).fit_transform(X)
    names.append('class')
    data = pd.DataFrame(data=np.append(X, np.reshape(y, (len(y), 1)), axis=1),
                        columns=names)
    sns.set()
    sns.pairplot(data, hue='class', x_vars=names[:-1], y_vars=names[:-1])
    plt.show()


def regressor_scatter(X, y, preds):
    """ Regressor scatter.

        Regressor scatter plot.

        Parameters
        ----------
        X: array-like, shape (n_samples, features_shape)
           The transformed data.
        y: numpy array of shape [n_samples]
           Target values.
        preds: numpy array of shape [n_samples]
               Predicted values.

        Returns
        -------
        None.
    """
    X = PCA(n_components=1).fit_transform(X)
    names = ('X', 'y', 'source')
    data_y = np.append(X, np.reshape(y, (len(y), 1)), axis=1)
    data_y = np.append(data_y, np.reshape([0]*len(y), (len(y), 1)),
                       axis=1)
    data_preds = np.append(X, np.reshape(preds, (len(preds), 1)), axis=1)
    data_preds = np.append(data_preds, np.reshape([1]*len(preds),
                                                  (len(preds), 1)), axis=1)
    data = pd.DataFrame(data=np.append(data_y, data_preds, axis=0),
                        columns=names)
    sns.set()
    sns.lmplot(x='X', y='y', hue='source', data=data)
    plt.show()


def metaparameter_plot(search, param, log_scale=True):
    """ Metaparameter plot.

        Train and test scores plotted along a meta-parameter search space.

        Parameters
        ----------
        search: search object
                Fitted sklearn search object.
        param: string
               Name of the meta-parameter.
        log_scale: boolean, default=True
                   Wether to use a logarithmic scale.

        Returns
        -------
        None.
    """
    param_range = search.cv_results_['param_' + param].data.astype('float32')
    train_mean = search.cv_results_['mean_train_' + search.refit]
    train_std = search.cv_results_['std_train_' + search.refit]
    test_mean = search.cv_results_['mean_test_' + search.refit]
    test_std = search.cv_results_['std_test_' + search.refit]
    plt.figure()
    if log_scale:
        plt.xscale('log')
    plt.xlabel(param)
    plt.ylabel(search.refit)
    plt.plot(param_range, train_mean, 'o', label='Train', color='b')
    plt.fill_between(param_range, train_mean - train_std,
                     train_mean + train_std, alpha=0.2, color='b')
    plt.plot(param_range, test_mean, 'o', label='Test', color='g')
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std,
                     alpha=0.2, color='g')
    plt.plot(param_range[search.best_index_], test_mean[search.best_index_],
             'o', label='Best', color='r')
    plt.plot(param_range[search.best_index_], train_mean[search.best_index_],
             'o', color='r')
    plt.axvline(x=param_range[search.best_index_], color='r')
    plt.legend(loc='best')
    plt.show()


def keras_history_plot(history):
    """ Keras history plot.

        Train loss plotted for each training epoch.

        Parameters
        ----------
        history: history object
                 Keras history object.

        Returns
        -------
        None.
    """
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.legend(loc='best')
    plt.show()
