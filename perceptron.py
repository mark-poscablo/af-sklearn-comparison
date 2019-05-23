# ArrayFire vs scikit-learn perceptron regression

import time
import argparse

import numpy as np
import sklearn as sk
from sklearn.linear_model import Perceptron

import arrayfire as af
from arrayfire.algorithm import count, imax, sum
from arrayfire.arith import abs, log, sigmoid
from arrayfire.blas import matmul, matmulTN
from arrayfire.data import constant, join
from arrayfire.device import eval, sync
from arrayfire.interop import from_ndarray
from arrayfire.statistics import mean

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


class AfPerceptron:
    def __init__(self, alpha=0.1, maxerr=0.05, maxiter=1000, verbose=False):
        self.__alpha = alpha
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


    def init_weights(self, X, Y):
        # Initialize parameters to 0
        self.__weights = af.constant(0, X.dims()[1], Y.dims()[1])


    def train(self, X, Y):
        # Initialize parameters to 0
        self.__weights = af.constant(0, X.dims()[1], Y.dims()[1])
        # self.__weights = af.randu(X.dims()[1], Y.dims()[1])

        for i in range(self.__maxiter):
            P = self.predict_proba(X)
            err = Y - P

            mean_abs_err = af.mean(af.abs(err))
            if mean_abs_err < self.__maxerr:
                break

            if self.__verbose and ((i + 1) % 25 == 0):
                print("Iter: {}, Err: {}".format(i+1, mean_abs_err))

            self.__weights = self.__weights + self.__alpha * af.matmulTN(X, err)


    def eval(self):
        af.eval(self.__weights)
        af.sync()


def arrayfire_perceptron_demo(dataset, num_classes=None):
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

    print('arrayfire perceptron classifier implementation')

    clf = AfPerceptron(alpha=0.1, maxerr=0.01, maxiter=1000, verbose=False)
    # Initial run to avoid overhead in training
    clf.train(train_feats, train_targets)
    clf.init_weights(train_feats, train_targets)

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

    print('Accuracy (test data): {0:2.2f}'.format(
        accuracy(test_outputs, test_targets)))

    # print('Accuracy on training data: {0:2.2f}'.format(accuracy(train_outputs, train_targets)))
    # print('Accuracy on testing data: {0:2.2f}'.format(accuracy(test_outputs, test_targets)))
    # print('Maximum error on testing data: {0:2.2f}'.format(abserr(test_outputs, test_targets)))

    return clf, dt_train


def sklearn_perceptron_demo(dataset):
    X_train = dataset[0]
    y_train = dataset[1]
    X_test = dataset[2]
    y_test = dataset[3]

    clf = Perceptron(alpha=0.1, tol=0.01, max_iter=1000, random_state=0, verbose=0, n_jobs=-1)

    print('sklearn perceptron classifier implementation')
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

    print('Accuracy (test data): {0:2.2f}'.format(
        clf.score(X_test, y_test) * 100))

    return clf, dt_train


def main():
    parser = argparse.ArgumentParser(description='af vs sklearn perceptron comparison')
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
        demo_simple(arrayfire_perceptron_demo, sklearn_perceptron_demo, dataset)
    elif args.type == 'predict':
        demo_pred(arrayfire_perceptron_demo, sklearn_perceptron_demo, dataset)
    elif args.type == 'benchmark':
        demo_bench(arrayfire_perceptron_demo, sklearn_perceptron_demo, dataset)
    else:
        parser.print_help()
        return -1


if __name__ == '__main__':
    main()
