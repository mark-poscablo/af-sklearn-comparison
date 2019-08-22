# ArrayFire vs scikit-learn logistic regression

import time
import argparse

import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

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


def accuracy2(predicted, target):
    # print(plabels.dims())
    # print(plabels[:, :10])
    # _, tlabels = af.imax(target, 1)
    return 100 * af.count(predicted == target) / target.elements()


def abserr(predicted, target):
    return 100 * af.sum(af.abs(predicted - target)) / predicted.elements()


class AfKNearestNeighbors:
    def __init__(self, dim=1, weight_by_dist=False, num_nearest=5, match_type=af.MATCH.SSD, verbose=False):
        self._dim = dim
        self._weight_by_dist = weight_by_dist
        self._num_nearest = num_nearest
        self._match_type = match_type
        self._verbose = verbose

        self._data = None
        self._labels = None


    def train(self, X, Y):
        # "fit" data
        self._data = X
        self._labels = Y


    def predict(self, X):
        near_locs, near_dists = af.vision.nearest_neighbour(X, self._data, self._dim, \
                                                            self._num_nearest, self._match_type)

        if self._weight_by_dist:
            weights = 1./near_dists
        else:
            weights = af.Array.copy(near_dists)
            weights[:] = 1

        top_labels = af.moddims(self._labels[near_locs], \
                                near_locs.dims()[0], near_locs.dims()[1])
        weighted_votes = af.scan_by_key(top_labels, weights) # reduce by key would be more ideal
        _, max_vote_locs = af.imax(weighted_votes, dim=0)
        pred_idxs = af.range(weighted_votes.dims()[1]) * weighted_votes.dims()[0] + max_vote_locs.T
        top_labels_flat = af.flat(top_labels)
        pred_classes = top_labels_flat[pred_idxs]
        return pred_classes


def arrayfire_knn_demo(dataset, num_classes=None):
    # Determine number of classes if not provided
    if num_classes is None:
        num_classes = np.amax(dataset[1] + 1)

    # Convert numpy array to af array (and convert labels/targets from ints to
    # one-hot encodings)
    train_feats = af.from_ndarray(dataset[0])
    train_targets = af.from_ndarray(dataset[1].astype('int32'))
    test_feats = af.from_ndarray(dataset[2])
    test_targets = af.from_ndarray(dataset[3].astype('int32'))

    num_train = train_feats.dims()[0]
    num_test = test_feats.dims()[0]

    print('arrayfire knn classifier implementation')

    clf = AfKNearestNeighbors()

    # Benchmark training
    t0 = time.time()
    clf.train(train_feats, train_targets)
    t1 = time.time()
    dt_train = t1 - t0
    print('Training time: {0:4.4f} s'.format(dt_train))

    # Benchmark prediction
    iters = 20
    test_outputs = None
    t0 = time.time()
    for i in range(iters):
        test_outputs = clf.predict(test_feats)
        af.eval(test_outputs)
        af.sync()
    t1 = time.time()
    dt = t1 - t0
    print('Prediction time: {0:4.4f} s'.format(dt / iters))
    print('Accuracy (test data): {0:2.2f}'.format(accuracy2(test_outputs, test_targets)))

    # print('Accuracy on training data: {0:2.2f}'.format(accuracy(train_outputs, train_targets)))
    # print('Accuracy on testing data: {0:2.2f}'.format(accuracy(test_outputs, test_targets)))
    # print('Maximum error on testing data: {0:2.2f}'.format(abserr(test_outputs, test_targets)))

    return clf, dt_train


def sklearn_knn_demo(dataset):
    X_train = dataset[0]
    y_train = dataset[1]
    X_test = dataset[2]
    y_test = dataset[3]

    clf = KNeighborsClassifier()

    print('sklearn KNN classifier implementation')
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
        demo_simple(arrayfire_knn_demo, sklearn_knn_demo, dataset)
    elif args.type == 'predict':
        demo_pred(arrayfire_knn_demo, sklearn_knn_demo, dataset)
    elif args.type == 'benchmark':
        demo_bench(arrayfire_knn_demo, sklearn_knn_demo, dataset)
    else:
        parser.print_help()
        return -1


if __name__ == '__main__':
    main()
