"""some helper functions."""
import numpy as np
import matplotlib.pyplot as plt
import csv
from implementations import *
from errno import WSABASEERR
from matplotlib import cm


def load_data(path_dataset, sub_sample_size=None):
    """
    Load data and convert it to the metric system.
    Arguments: path_dataset: path of training or test data set.
               sub_sample_size: size for subsample. None if subsample is not considered.
    """
    label = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, dtype=str, usecols=1)
    id = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, dtype=int, usecols=0).astype(np.int)
    features = np.genfromtxt(path_dataset, delimiter=",", skip_header=1)[:, 2:]

    # convert label from string to int
    clabel = np.where(label == 's', 1, 0)

    # sub-sample
    if sub_sample_size is not None:
        clabel = clabel[:sub_sample_size]
        id = id[:sub_sample_size]
        features = features[:sub_sample_size]

    return clabel, features, id


def standardize(x):
    """
    Normalize a feature vector.
    Arguments: x: feature vector.
    """
    mean_x = np.nanmean(x)
    x = x - mean_x
    std_x = np.nanstd(x)
    if std_x != 0:
        x = x / std_x
    return x, mean_x, std_x


def standardize_data(x):
    """
    Normalize a feature matrix.
    Arguments: x: feature matrix.
    """
    for i in range(0, x.shape[1]):
        x[:, i], mean, variance = standardize(x[:, i])
    return x


def get_mean_std(x):
    """
    Get the mean and standard deviation of each feature vector in a feature matrix.
    Arguments: x: feature matrix.
    """
    mean_std_dict = dict()
    for i in range(0, x.shape[1]):
        if i != 22:
            _, mean, variance = standardize(x[:, i])
            mean_std_dict[i] = [mean, variance]
    return mean_std_dict


def pca_withpic(features):
    """
    Conduct PCA towards a feature matrix and output analysis figures.
    Arguments: features: feature matrix.
    """

    # compute the covariance matrix
    cov_mat = np.cov(features.T)
    # compute eigenvalues and eigenvectors for the from the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    # sort eigenvalues from large to small
    idx = eig_val_cov.argsort()[::-1]
    eig_val_cov_sorted = eig_val_cov[idx]
    eig_vec_cov_sorted = eig_vec_cov[:, idx]
    # transform data into new dimensions
    features_pca = features.dot(eig_vec_cov_sorted)
    # plot cumulative variance
    plt.figure()
    plt.plot(np.cumsum(eig_val_cov_sorted)/np.sum(eig_val_cov_sorted))
    plt.xticks(np.arange(0, features.shape[1], 1.0))
    # add x grid
    plt.grid(axis='x', alpha=0.75)
    # add a horizontal line wtih text
    plt.hlines(0.9, 0, features.shape[1], colors='r', linestyles='dashed', label='90%')
    plt.hlines(0.999, 0, features.shape[1], colors='m', linestyles='dashed', label='99.9%')
    plt.legend()
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    return features_pca, eig_val_cov_sorted


def pca(features, num):
    """
    Conduct PCA towards a feature matrix and output analysis figures.
    Arguments: features: feature matrix.
               num: the number of feature vectors in the output feature matrix after PCA.
    """

    # compute the covariance matrix
    cov_mat = np.cov(features.T)
    # compute eigenvalues and eigenvectors for the from the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    # sort eigenvalues from large to small
    idx = eig_val_cov.argsort()[::-1]
    eig_val_cov_sorted = eig_val_cov[idx]
    eig_vec_cov_sorted = eig_vec_cov[:, idx]
    # transform data into new dimensions
    features_pca = features.dot(eig_vec_cov_sorted)

    return features_pca[:, :num], eig_val_cov_sorted[:num]


# For cross validation
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma):
    """return the errors of logistic regression for a fold corresponding to k_indices
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()

    Returns:
        train and test errors
    """
    # get k'th subgroup in test, others in train
    k_test = k_indices[k]
    k_train = np.delete(k_indices, k, 0).flatten()

    # form data with polynomial degree
    x_train = tx[k_train, :]
    x_test = tx[k_test, :]

    # logistic regression
    w, _ = reg_logistic_regression(y[k_train], x_train, lambda_, initial_w, max_iters, gamma)
    # w, _ = logistic_regression(y[k_train], x_train, initial_w, max_iters, gamma)

    # calculate the accuracy for train and test data
    y_pred2 = logistic_regression_prediction(x_test, w)
    y_pred1 = logistic_regression_prediction(x_train, w)
    tr_error = np.sum(abs(np.array(y_pred1) - np.array(y[k_train]))) / len(y_pred1)
    val_error = np.sum(abs(np.array(y_pred2) - np.array(y[k_test]))) / len(y_pred2)

    print(f"Training error {100 * tr_error:.2f}%, validation error {100 * val_error:.2f}%")
    return tr_error, val_error, w


def cross_validation_visualization(lambds, err_tr, err_te):
    """visualization the curve results in cross validation"""

    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(16, 12))
    plt.semilogx(lambds, 1. - err_tr, marker="o", color='b', label='Training accuracy', linewidth=5.0, markersize=15.0)
    plt.semilogx(lambds, 1. - err_te, marker="o", color='r', label='Validation accuracy', linewidth=5.0, markersize=15.0)
    plt.xlabel("Lambda", fontsize=36, fontweight='bold', labelpad=20)
    plt.ylabel("Accuracy", fontsize=36, fontweight='bold', labelpad=20)
    plt.tick_params(labelsize=35)
    plt.legend(loc=3, fontsize=36)
    plt.grid(True)
    plt.savefig("cross_validation")


def cross_validation_demo(y, x_train, k_fold, lambdas, initial_w, max_iters, gamma):
    """cross validation over regularisation parameter lambda.

    Args:
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_error : scalar, the associated error for the best lambda
    """

    seed = 10

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    ws = []

    # cross validation over lambdas
    avg_err_te = np.zeros(len(lambdas))
    avg_err_tr = np.zeros(len(lambdas))
    for p in range(len(lambdas)):
        for k in range(k_fold):
            tr_error, val_error, w = cross_validation(y, x_train, k_indices, k, lambdas[p], initial_w, max_iters, gamma)
            avg_err_te[p] = avg_err_te[p] + val_error
            avg_err_tr[p] = avg_err_tr[p] + tr_error
            ws.append(w)
        avg_err_te = avg_err_te / k_fold
        avg_err_tr = avg_err_tr / k_fold

        best_lambda = lambdas[np.argmin(avg_err_te)]
        best_error = min(avg_err_te)
        best_w = ws[np.argmin(avg_err_te)]

        print(f"best_lambda, {best_lambda}")
        return lambdas, avg_err_tr, avg_err_te, best_lambda, best_error, best_w


def cross_validation_ridge(y, tx, k_indices, k, lambda_):
    """return the errors of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()

    Returns:
        train and test errors
    """

    # get k'th subgroup in test, others in train
    k_test = k_indices[k]
    k_train = np.delete(k_indices, k, 0).flatten()

    # form data with polynomial degree
    x_train = tx[k_train, :]
    x_test = tx[k_test, :]

    # ridge regression
    w, loss = ridge_regression(y[k_train], x_train, lambda_)

    # calculate the loss for train and test data
    y_pred1 = regression_prediction(x_train, w)
    y_pred2 = regression_prediction(x_test, w)
    tr_error = np.sum(abs(np.array(y_pred1) - np.array(y[k_train]))) / len(y_pred1)
    val_error = np.sum(abs(np.array(y_pred2) - np.array(y[k_test]))) / len(y_pred2)

    print(f"Training error {100 * tr_error:.2f}%, validation error {100 * val_error:.2f}%")
    return tr_error, val_error, w


def cross_validation_ridge_demo(y, x_train, k_fold, lambdas):
    """cross validation over regularisation parameter lambda.

    Args:
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_error : scalar, the associated error for the best lambda
    """

    seed = 10

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    ws = []

    # cross validation over lambdas
    avg_err_te = np.zeros(len(lambdas))
    avg_err_tr = np.zeros(len(lambdas))
    for p in range(len(lambdas)):
        for k in range(k_fold):
            tr_error, val_error, w = cross_validation_ridge(y, x_train, k_indices, k, lambdas[p])
            avg_err_te[p] = avg_err_te[p] + val_error
            avg_err_tr[p] = avg_err_tr[p] + tr_error
            ws.append(w)
    avg_err_te = avg_err_te / k_fold
    avg_err_tr = avg_err_tr / k_fold

    best_lambda = lambdas[np.argmin(avg_err_te)]
    best_error = min(avg_err_te)
    best_w = ws[np.argmin(avg_err_te)]

    print(f"best_lambda, {best_lambda}")
    return lambdas, avg_err_tr, avg_err_te, best_lambda, best_error, best_w


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids: event ids associated with each prediction
               y_pred: predicted class labels
               name: string name of .csv output file to be created
    """
    with open(name, 'w', newline="") as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def plot_correlation(features, title="Correlation matrix"):
    """
    Plot correlation heatmap for exploratory analysis.
    """
    corr = np.corrcoef(features.T)
    plt.figure(figsize=(10, 10))
    plt.imshow(corr, cmap=cm.RdBu)
    # highlight value larger that 0.9
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if abs(corr[i, j]) > 0.9:
                plt.text(j, i, str(round(corr[i, j], 2)), horizontalalignment='center',
                         verticalalignment='center', color='white')
    plt.colorbar()
    plt.xticks(list(range(features.shape[1])))
    plt.yticks(list(range(features.shape[1])))
    plt.title(title)
    plt.show()


def plot_box_groups(x, groups=[0, 1, 2, 3]):
    """
    Plotting the boxplots for all features.
    """
    fig = plt.figure(figsize=(20, 15))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for featureID in range(x.shape[1]):
        ax = fig.add_subplot(6, 5, featureID + 1)
        if groups == None:
            data = x[:, featureID]
        else:
            data = [x[x[:, 22] == groupe][:, featureID] for groupe in groups]
        ax.boxplot(data, showfliers=True)
        ax.grid(axis='y', alpha=0.75)
        ax.set(title='Feature column:{}'.format(featureID))


def plot_hist(x, plot_features=None):
    """
    Plotting the histograms of each feature.
    """
    fig = plt.figure(figsize=(20, 15))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    if plot_features == None:
        plot_features_id = range(x.shape[1])
    else:
        plot_features_id = plot_features
    for i, featureID in enumerate(plot_features_id):
        ax = fig.add_subplot(6, 5, i + 1)
        ax.hist(x=x[:, featureID], bins=100)
        ax.grid(axis='y', alpha=0.75)
        ax.set(title=f'Feature column:{featureID}')
    plt.show()


def augment_with_square_root(x):
    return np.c_[x, np.sqrt(x)]


def augment_with_power_2(x):
    return np.c_[x, x**2]


def augment_with_logarithm(x):
    return np.c_[x, np.log(x)]


def augment_with_gaussian_basis(x, basis_number):
    np.random.seed(5)
    augmented_X = x
    mu_j = np.random.uniform(-2, 2, basis_number)
    sigma_j = np.ones(basis_number)
    for feature_count in range(x.shape[1]):
        for j in range(basis_number):
            augmented_X = np.c_[
                augmented_X, np.exp(-(x[:, feature_count] - mu_j[j]) ** 2 / (2 * sigma_j[j] ** 2))]
    return augmented_X