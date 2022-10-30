import numpy as np


def compute_loss_mse(y, tx, w):

    """Calculate the loss using either MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - np.dot(tx, w)
    return np.array((np.dot(np.transpose(e), e) / (2 * len(y))).item())


def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx.dot(w)
    return -np.transpose(tx).dot(e) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD 
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    loss = compute_loss_mse(y, tx, w)
    for n_iter in range(max_iters):

        grad = compute_gradient(y, tx, w) # gradient
        loss = compute_loss_mse(y, tx, w)  # loss
        w = w - gamma * grad


        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    loss = compute_loss_mse(y, tx, w) 
    return w, loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    
    # ***************************************************
    e = y - tx.dot(w)
    return -np.transpose(tx).dot(e) / len(y)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD 
    """
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 1
    
    for n_iter in range(max_iters):
        '''
        n_rand_row = np.random.randint(len(y), size=batch_size)
        y_rand = np.random.choice(y, batch_size)
        tx_rand = tx[n_rand_row]
        grad = compute_stoch_gradient(y_rand, tx_rand, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * grad
        '''

        for yi, txi in batch_iter(y, tx, batch_size):
            grad = compute_stoch_gradient(yi, txi, w)
            loss = compute_loss_mse(y, tx, w)
            w = w - gamma * grad

        ws.append(w)
        losses.append(loss)
        
        print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    loss = compute_loss_mse(y, tx, w)
    return w, loss


# least squares
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """

    temp1 = np.linalg.inv(np.transpose(tx).dot(tx))
    temp2 = np.transpose(tx).dot(y)
    w = temp1.dot(temp2)

    # e = y - tx.dot(w)
    # mse = (np.transpose(e).dot(e) / (2 * len(y))).item()
    mse = compute_loss_mse(y, tx, w)
    return w, mse


# ridge regression
def ridge_regression(y, tx, lambda_):
    """
    implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """

    D = tx.shape[1]
    w = np.dot(np.linalg.inv(np.transpose(tx).dot(tx) + lambda_ * 2 * D * np.eye(D)), 
              np.transpose(tx).dot(y))
    loss = compute_loss_mse(y, tx, w)

    return w, loss


# logistic regression
def sigmoid(z):
    # transform y into probability
    sigmoid_z = 1.0 / (1.0 + np.exp(-z))
    sigmoid_z = np.where(sigmoid_z < 1e-7, 1e-7, sigmoid_z)
    sigmoid_z = np.where(sigmoid_z > 1 - 1e-7, 1 - 1e-7, sigmoid_z)

    return sigmoid_z


def cost_logistic(y, tx, w):
    # cost function for logistic regression
    z = np.dot(tx, w)
    temp1 = y.T.dot(np.log(sigmoid(z)))
    temp2 = (1 - y).T.dot(np.log(1 - sigmoid(z)))
    cost = -(temp1 + temp2) / len(y)

    return np.array(cost.item())


def gradient_logistic(y, tx, w):
    # gradient for logistic regression
    return np.dot(tx.T, sigmoid(np.dot(tx, w)) - y) / len(y)


def gradient_logistic_reg(y, tx, w, lambda_):
    # gradient for logistic regression
    D = tx.shape[0]
    return (np.dot(tx.T, sigmoid(np.dot(tx, w)) - y) + lambda_ * 2 * D * w) / len(y)


def logistic_regression(y, tx, w, max_iters, gamma):
    """implement logistic regression regression.
        
        Args:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N,D), D is the number of features.
            lambda_: scalar.
        
        Returns:
            w: optimal weights, numpy array of shape(D,), D is the number of features.

    """
    if (w is None) or (len(w) != tx.shape[1]):
        w = np.zeros(tx.shape[1])
    for n_iter in range(max_iters):
        # if n_iter > 0 and n_iter % 2000 == 0:
        #     gamma = 0.9 * gamma
        grad = gradient_logistic(y, tx, w)  # gradient
        loss = cost_logistic(y, tx, w)  # loss
        # w = w - gamma * grad
        w = w - gamma * grad

        # print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #       bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        if n_iter % 10 == 0:
            print("GD iter. {bi}/{ti}: loss={l}, gamma = {g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=gamma))
    loss = cost_logistic(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, w, max_iters, gamma):
    """implement regularized logistic regression regression (l2 norm)
        
        Args:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N,D), D is the number of features.
            lambda_: scalar.
        
        Returns:
            w: optimal weights, numpy array of shape(D,), D is the number of features.
            loss: loss of final weights.
    """
    if (w is None) or (len(w) != tx.shape[1]):
        w = np.zeros(tx.shape[1])
    for n_iter in range(max_iters):
        # if n_iter > 0 and n_iter % 2000 == 0:
        #     gamma = 0.9 * gamma
        # for yi, txi in batch_iter(y, tx, batch_size=1):
        grad = gradient_logistic_reg(y, tx, w, lambda_)
        loss = cost_logistic(y, tx, w)
        w = w - gamma * grad

        if n_iter % 10 == 0:
            print("GD iter. {bi}/{ti}: loss={l}, gamma = {g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=gamma))

        # if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
        # #break

    loss = cost_logistic(y, tx, w)
    return w, loss


def logistic_regression_prediction(tx, w):

    y_pred = np.zeros(tx.shape[0])
    results = sigmoid(np.dot(tx, w))
    for i in range(len(results)):
        if results[i] >= 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred


def regression_prediction(tx, w):

    y_pred = np.zeros(tx.shape[0])
    results = np.dot(tx, w)
    for i in range(len(results)):
        if results[i] >= 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred

