import numpy as np


class AdalineGD(object):
    """ADAptive LInear NEuron classifer with Gradient Decent.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Number of passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ------------
        X : {array-like}, shape = [n_examples, m_features]
            Training vector, where n_examples is the number of examples
            m_features is the number of features.
        y : {array-like}, shape = [n_examples]
            Target values.

        Returns
        ------------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0 # This is the cost function J(w) page 37
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute Linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


class AdalineSGD(object):
    """ADAptive LInear NEuron classifier with Stocastic Gradient Decent.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes orver the training dataset.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes
    ------------
    w_ : 1d-array
        weights after fitting.
    cost_ : list
        Sum-of-squares cost function value averaged over all
        training examples in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state


    def fit(self, X, y):
        """Fit training data.

        Parameters
        ------------
        X : {array-like}, shape = [n_examples, m_features]
            Training vecots, where n_examples is the number of
            examples and m_features is the number of featues.
        y : {array-like}, shape = [n_examples]
            Target values.

        Returns
        ------------
        self : object

        """
        # initialize the weights
        self._initialize_weights(X.shape[1])
        # start a list to save the values of the cost function.
        self.cost_ = []

        # start the training process
        for i in range(self.n_iter):
            # shuffle the training set if needed.
            if self.shuffle:
                X, y = self._shuffle(X, y)
            # save the value of the cost function on each example.
            cost = []
            # loop over the examples
            for xi, target in zip(X, y):
                # update the weight and save the cost in the above cost on each example.
                cost.append(self._update_weights(xi, target))

            # calculate the average cost for this training iteration.
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""

        # if the weights have not been initialized, the do so.
        if not self.w_initialized:
            self._initialized_weights(X.shape[1])

        # update the weights based on the partial fit.
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2

        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class lable after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
