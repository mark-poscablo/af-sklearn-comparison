# ArrayFire vs scikit-learn logistic regression

import time
import argparse

import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression

import arrayfire as af
from arrayfire.algorithm import count, imax, sum
from arrayfire.arith import abs, log, sigmoid
from arrayfire.blas import matmul, matmulTN
from arrayfire.data import constant, join
from arrayfire.device import eval, sync
from arrayfire.interop import from_ndarray

from util import ints_to_onehots
from util import onehots_to_ints
from util import read_and_preprocess_iris_data
from util import read_and_preprocess_mnist_data
from util import read_and_preprocess_notmnist_data
from util import demo_simple
from util import demo_pred
from util import demo_bench


def accuracy(predicted, target):
    _, tlabels = af.imax(target, 1)
    _, plabels = af.imax(predicted, 1)
    return 100 * af.count(plabels == tlabels) / tlabels.elements()


def abserr(predicted, target):
    return 100 * af.sum(af.abs(predicted - target)) / predicted.elements()


class AfLogisticRegression:
    def __init__(self, alpha=0.1, lambda_param=1.0, maxerr=0.01, maxiter=1000, verbose=False):
        self.__alpha = alpha
        self.__lambda_param = lambda_param
        self.__maxerr = maxerr
        self.__maxiter = maxiter
        self.__verbose = verbose
        self.__weights = None


    def predict_proba(self, X):
        Z = af.matmul(X, self.__weights)
        return af.sigmoid(Z)


    def predict_log_proba(self, X):
        return af.log(self.predict_proba(X))


    def predict(self, X):
        probs = self.predict_proba(X)
        _, classes = af.imax(probs, 1)
        return classes


    def cost(self, X, Y):
        # Number of samples
        m = Y.dims()[0]

        dim0 = self.__weights.dims()[0]
        dim1 = self.__weights.dims()[1] if len(self.__weights.dims()) > 1 else None
        dim2 = self.__weights.dims()[2] if len(self.__weights.dims()) > 2 else None
        dim3 = self.__weights.dims()[3] if len(self.__weights.dims()) > 3 else None
        # Make the lambda corresponding to self.__weights(0) == 0
        lambdat = af.constant(self.__lambda_param, dim0, dim1, dim2, dim3)

        # No regularization for bias weights
        lambdat[0, :] = 0

        # Get the prediction
        H = self.predict_proba(X)

        # Cost of misprediction
        Jerr = -1 * af.sum(Y * af.log(H) + (1 - Y) * af.log(1 - H), dim=0)

        # Regularization cost
        Jreg = 0.5 * af.sum(lambdat * self.__weights * self.__weights, dim=0)

        # Total cost
        J = (Jerr + Jreg) / m

        # Find the gradient of cost
        D = (H - Y)
        dJ = (af.matmulTN(X, D) + lambdat * self.__weights) / m

        return J, dJ


    def train(self, X, Y):
        # Initialize parameters to 0
        self.__weights = af.constant(0, X.dims()[1], Y.dims()[1])

        for i in range(self.__maxiter):
            # Get the cost and gradient
            J, dJ = self.cost(X, Y)
            err = af.max(af.abs(J))
            if err < self.__maxerr:
                print('Iteration {0:4d} Err: {1:4f}'.format(i + 1, err))
                print('Training converged')
                return self.__weights

            if self.__verbose and ((i+1) % 10 == 0):
                print('Iteration {0:4d} Err: {1:4f}'.format(i + 1, err))

            # Update the weights via gradient descent
            self.__weights = self.__weights - self.__alpha * dJ

        if self.__verbose:
            print('Training stopped after {0:d} iterations'.format(self.__maxiter))


    def eval(self):
        af.eval(self.__weights)
        af.sync()


def arrayfire_logit_demo(dataset, num_classes=None):
    # Determine number of classes if not provided
    if num_classes is None:
        num_classes = np.amax(dataset[1] + 1)

    # Convert numpy array to af array (and convert labels/targets from ints to
    # one-hot encodings)
    train_feats = af.from_ndarray(dataset[0])
    train_targets = af.from_ndarray(ints_to_onehots(dataset[1], num_classes))
    test_feats = af.from_ndarray(dataset[2])
    test_targets = af.from_ndarray(ints_to_onehots(dataset[3], num_classes))

    num_train = train_feats.dims()[0]
    num_test = test_feats.dims()[0]

    # Add bias
    train_bias = af.constant(1, num_train, 1)
    test_bias = af.constant(1, num_test, 1)
    train_feats = af.join(1, train_bias, train_feats)
    test_feats = af.join(1, test_bias, test_feats)

    print('arrayfire logit classifier implementation')

    clf = AfLogisticRegression(alpha=0.1,          # learning rate
                               lambda_param = 1.0, # regularization constant
                               maxerr=0.01,        # max error
                               maxiter=1000,       # max iters
                               verbose=False       # verbose mode
    )

    # Benchmark training
    t0 = time.time()
    clf.train(train_feats, train_targets)
    clf.eval()
    t1 = time.time()
    dt_train = t1 - t0
    print('Training time: {0:4.4f} s'.format(dt_train))

    # Benchmark prediction
    iters = 100
    test_outputs = None
    t0 = time.time()
    for i in range(iters):
        test_outputs = clf.predict_proba(test_feats)
        af.eval(test_outputs)
        af.sync()
    t1 = time.time()
    dt = t1 - t0
    print('Prediction time: {0:4.4f} s'.format(dt / iters))
    print('Accuracy (test data): {0:2.2f}'.format(accuracy(test_outputs, test_targets)))

    # print('Accuracy on training data: {0:2.2f}'.format(accuracy(train_outputs, train_targets)))
    # print('Accuracy on testing data: {0:2.2f}'.format(accuracy(test_outputs, test_targets)))
    # print('Maximum error on testing data: {0:2.2f}'.format(abserr(test_outputs, test_targets)))

    return clf, dt_train


def sklearn_logit_demo(dataset):
    X_train = dataset[0]
    y_train = dataset[1]
    X_test = dataset[2]
    y_test = dataset[3]

    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                             random_state=42, verbose=0, max_iter=1000, n_jobs=-1)

    print('sklearn logit classifier implementation')
    t0 = time.time()
    clf.fit(X_train, y_train)
    t1 = time.time()
    dt_train = t1 - t0
    print('Training time: {0:4.4f} s'.format(dt_train))

    t0 = time.time()
    y_pred = clf.predict(X_test)
    t1 = time.time()
    dt = t1 - t0
    print('Prediction time: {0:4.4f} s'.format(dt))
    print('Accuracy (test data): {0:2.2f}'.format(clf.score(X_test, y_test) * 100))

    return clf, dt_train


def main():
    parser = argparse.ArgumentParser(description='af vs sklearn logit comparison')
    parser.add_argument('-b', '--backend',
                        choices=['default', 'cpu', 'cuda', 'opencl'],
                        default='default',
                        action='store',
                        help='ArrayFire backend to be used')
    parser.add_argument('-v', '--device',
                        type=int,
                        default=0,
                        action='store',
                        help='ArrayFire backend device to be used')
    parser.add_argument('-d', '--dataset',
                        choices=['iris', 'mnist', 'notmnist'],
                        default='iris',
                        action='store',
                        help='Dataset to be used')
    parser.add_argument('-t', '--type',
                        choices=['simple', 'predict', 'benchmark'],
                        default='simple',
                        action='store',
                        help='Demo type')
    args = parser.parse_args()

    af.set_backend(args.backend)
    af.set_device(args.device)

    af.info()

    dataset = None
    if args.dataset == 'iris':
        dataset = read_and_preprocess_iris_data()
    elif args.dataset == 'mnist':
        dataset = read_and_preprocess_mnist_data()
    elif args.dataset == 'notmnist':
        dataset = read_and_preprocess_notmnist_data()
    else:
        parser.print_help()
        return -1

    print('------------')

    if args.type == 'simple':
        demo_simple(arrayfire_logit_demo, sklearn_logit_demo, dataset)
    elif args.type == 'predict':
        demo_pred(arrayfire_logit_demo, sklearn_logit_demo, dataset)
    elif args.type == 'benchmark':
        demo_bench(arrayfire_logit_demo, sklearn_logit_demo, dataset)
    else:
        parser.print_help()
        return -1


if __name__ == '__main__':
    main()
