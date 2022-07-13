# scripts for creating the figures

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def pred_proba_plot(clf, X, y, cv=5, no_bins=25, x_min=0.5, x_max=1, classifier=''):
    '''
    Return figure - histogram display of correcly predicted results against incorrectly given results given the outputed probability of the classifier.
    
    Parameters
    ----------
    clf : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data..
    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d..
    y : array-like
        The target variable to try to predict in the case of
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.. The default is 5.
    no_bins : int, optional
        number of bins of histogram. The default is 25.
    x_min : int, optional
        min x on histogram plot. The default is 0.5.
    x_max : in, optional
        max x on histogram plot. The default is 1.
    output_progress : display, optional
        print no. iterations complete to console. The default is True.
    classifier : string, optional
        classifier used, will be input to title. The default is ''.
    Returns
    -------
    fig : 
    '''
    y_dup = []
    correct_guess_pred = []
    incorrect_guess_pred = []
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True)
    y_pred_cv = cross_val_predict(clf, X, y, cv=skf)
    y_pred_proba_cv = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')
    y_dup.append(list(y))
    for i in range(len(y_pred_cv)):
        if y_pred_cv[i] == list(y)[i]:
            correct_guess_pred.append(max(y_pred_proba_cv[i]))
        if y_pred_cv[i] != list(y)[i]:
            incorrect_guess_pred.append(max(y_pred_proba_cv[i]))         
    
    bins = np.linspace(x_min, x_max, no_bins)
    fig, ax = plt.subplots()
    ax.hist(incorrect_guess_pred, bins, alpha=0.5, edgecolor='#1E212A', color='red', label='Incorrect Prediction')
    ax.hist(correct_guess_pred, bins, alpha=0.5, edgecolor='#1E212A', color='green', label='Correct Prediction')
    ax.legend()
    ax.set_title(f'{classifier}', y=1, fontsize=16, fontweight='bold');
    ax.set(ylabel='Number of Occurences',
            xlabel='Prediction Probability')
    return fig


def plot_cross_val_confusion_matrix(clf, X, y, display_labels='', title='', cv=5):
    '''
    Function to plot confusion matrix given the result of cross-validation, as oposed to the standard confucion matriax on test split data.
    
    Parameters
    ----------
    clf : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d..
    y : array-like
        The target variable to try to predict in the case of supervised learning.
    display_labels : ndarray of shape (n_classes,), optional
        display labels for plot. The default is ''.
    title : string, optional
        Title to be displayed at top of plot. The default is ''.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.. The default is 5.
    Returns
    -------
    display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
    '''
    
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    cm_norm = confusion_matrix(y, y_pred, normalize='true')
    fig = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=display_labels)
    fig.plot(cmap=plt.cm.Blues)
    fig.ax_.set_title(title)
    
    return fig


def plot_learn_times(clf, title, X, y, axes=None, ylim=None, cv=5, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    
    Parameters
    ----------
    clf : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
        
    title : str
        Title for the chart.
        
    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.
        
    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.
        
    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.
        
    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        
    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        
    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve.
        
    Returns
    -------
    pltplot
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        clf,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt