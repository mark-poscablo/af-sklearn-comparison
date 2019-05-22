import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import copy
import pickle

import numpy as np
from sklearn.datasets import load_iris

import arrayfire as af
from arrayfire.array import read_array, transpose
from arrayfire.data import moddims
from arrayfire.interop import from_ndarray


def ints_to_onehots(ints, num_classes):
    onehots = np.zeros((ints.shape[0], num_classes), dtype='float32')
    onehots[np.arange(ints.shape[0]), ints] = 1
    return onehots


def onehots_to_ints(onehots):
    return np.argmax(onehots, axis=1)


def read_and_preprocess_iris_data():
    print('Reading and preprocessing iris data (small)...')
    X, y = load_iris(return_X_y=True)
    X = X.astype('float32')
    y = y.astype('uint32')
    print('Reading and preprocessing iris data (small) DONE')
    return (X, y, X, y)


def read_and_preprocess_mnist_data():
    print('Reading and preprocessing MNIST data...')

    train_images = af.read_array('train_images.af', key='train_images')
    train_targets = af.read_array('train_targets.af', key='train_targets')
    test_images = af.read_array('test_images.af', key='test_images')
    test_targets = af.read_array('test_targets.af', key='test_targets')

    num_train = train_images.dims()[2]
    num_classes = train_targets.dims()[0]
    num_test = test_images.dims()[2]

    feature_length = int(train_images.elements() / num_train)
    train_feats = af.transpose(
        af.moddims(train_images, feature_length, num_train))
    test_feats = af.transpose(af.moddims(test_images, feature_length,
                                         num_test))

    train_targets = af.transpose(train_targets)
    test_targets = af.transpose(test_targets)

    X_train = train_feats.to_ndarray()
    y_train = train_targets.to_ndarray()
    X_test = test_feats.to_ndarray()
    y_test = test_targets.to_ndarray()

    y_train = onehots_to_ints(y_train)
    y_test = onehots_to_ints(y_test)

    y_train = y_train.astype('uint32')
    y_test = y_test.astype('uint32')

    print('Reading and preprocessing MNIST data DONE')
    return (X_train, y_train, X_test, y_test)


def read_and_preprocess_notmnist_data():
    print('Reading and preprocessing notMNIST data...')

    with open('notMNIST.pickle', 'rb') as f:
        data = pickle.load(f)

    train_images = af.from_ndarray(data['train_dataset'])
    train_targets = af.from_ndarray(ints_to_onehots(data['train_labels'], 10))
    test_images = af.from_ndarray(data['test_dataset'])
    test_targets = af.from_ndarray(ints_to_onehots(data['test_labels'], 10))

    num_train = train_images.dims()[0]
    num_test = test_images.dims()[0]

    feature_length = int(train_images.elements() / num_train)
    train_feats = af.moddims(train_images, num_train, feature_length)
    test_feats = af.moddims(test_images, num_test, feature_length)

    X_train = train_feats.to_ndarray()
    y_train = train_targets.to_ndarray()
    X_test = test_feats.to_ndarray()
    y_test = test_targets.to_ndarray()

    y_train = onehots_to_ints(y_train).astype('uint32')
    y_test = onehots_to_ints(y_test).astype('uint32')

    print('Reading and preprocessing notMNIST data DONE')
    return (X_train, y_train, X_test, y_test)

# The following demo code assumes that the dataset is a
# tuple of numpy ndarrays, containing (in this order):
# training samples, training labels, test samples, test labels,
# where samples are of type float32, and labels are uint32, and
# where each sample is represented as an array of features, and
# each label is represented as an integer that maps to a class
# Thus, training/test samples has the shape: m samples x n features
# training/test labels has the shape: m samples x 1

def demo_simple(af_demo, sklearn_demo, dataset):
    af_demo(dataset)
    print('------------')
    sklearn_demo(dataset)


def demo_pred(af_demo, sklearn_demo, dataset):
    af_clf, _ = af_demo(dataset)
    print('------------')
    sk_clf, _ = sklearn_demo(dataset)
    print('------------')

    test_feats = dataset[2]
    for _ in range(10):
        rand_idx = np.random.randint(0, test_feats.shape[0])
        test_sample = test_feats[rand_idx].reshape(1, test_feats.shape[1])

        af_test_sample = np.ones((1, test_sample.shape[1] + 1), dtype='float32')
        af_test_sample[0, :-1] = test_sample
        af_test_sample = af.from_ndarray(af_test_sample)

        af_pred = af_clf.predict(af_test_sample).scalar()
        print("arrayfire prediction: {}".format(af_pred))

        sk_pred = sk_clf.predict(test_sample)[0]
        print("sklearn prediction: {}".format(sk_pred))

        test_sample_square = test_sample.reshape(28, 28)
        plt.imshow(test_sample_square, interpolation='nearest')
        plt.show()

        print('--')


def demo_bench(af_demo, sklearn_demo, dataset):
    # data_sizes = [500, 1000, 2500, 5000, 10000, 50000, 100000, 200000]
    # data_sizes = [500, 1000, 2500, 5000, 10000]
    # data_sizes = [500, 1000, 2500]
    data_sizes = [100, 1000, 10000, 100000]
    af_times = []
    sk_times = []

    for size in data_sizes:
        print('Data size: {}'.format(size)
        )
        trunc_dataset = (dataset[0][:size], dataset[1][:size], dataset[2], dataset[3])
        _, af_time = af_demo(trunc_dataset)
        print('--')
        _, sk_time = sklearn_demo(trunc_dataset)
        print('------------')

        af_times.append(af_time)
        sk_times.append(sk_time)

    x = np.array(data_sizes)
    y1 = np.array(af_times)
    y2 = np.array(sk_times)
    s = y2 / y1

    plt.subplot(2, 1, 1)
    plt.plot(x, y1, label='arrayfire', marker='o', color='#eb7242')
    plt.plot(x, y2, label='sklearn', marker='o', color='#48a0e8')
    plt.ylabel('Time (s)')
    plt.legend()
    for i in range(len(data_sizes)-1, len(data_sizes)):
        plt.annotate('{0:4.2f}s'.format(y1[i]),
                     (data_sizes[i], y1[i]),
                     (data_sizes[i] + 20, y1[i]))
        plt.annotate('{0:4.2f}s'.format(y2[i]),
                     (data_sizes[i], y2[i]),
                     (data_sizes[i] + 20, y2[i]))

    plt.subplot(2, 1, 2)
    plt.plot(x, s, label='speedup', marker='o', color='#eb7242')
    plt.ylabel('Speedup')
    for i in range(len(data_sizes)-1, len(data_sizes)):
        plt.annotate('{0:4.2f}x'.format(s[i]),
                     (data_sizes[i], s[i]),
                     (data_sizes[i] + 20, s[i]))


    plt.xlabel('Training size')

    plt.show()
