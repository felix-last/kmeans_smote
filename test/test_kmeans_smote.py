"""Test oversampling method k-means SMOTE"""
# Authors: Felix Last
# License: MIT

import warnings
import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
    assert_raises_regex)

from kmeans_smote import KMeansSMOTE
from imblearn.over_sampling import SMOTE, RandomOverSampler

RND_SEED = 0
R_TOL = 1e-4
X = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141],
              [1.25192108, -0.22367336], [0.53366841, -0.30312976],
              [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
              [0.83680821, 1.72827342], [0.3084254, 0.33299982],
              [0.70472253, -0.73309052], [0.28893132, -0.38761769],
              [1.15514042, 0.0129463], [0.88407872, 0.35454207],
              [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
              [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
              [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
              [0.08711622, 0.93259929], [1.70580611, -0.11219234]])
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])

# y_resampled should look like Y_EXPECTED
Y_EXPECTED = np.array([
    0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
])
X_SHAPE_EXPECTED = (X.shape[0] + (Y_EXPECTED.size - Y.size), X.shape[1])

X_MULTICLASS = np.array([[5.8, 7.15], [6.3, 6.5], [5.5, 6.4], [6.2, 7.95],
                        [6.85, 7.5], [7.7, 7.5], [7.15, 6.95], [7.1, 8.6],
                        [8.85, 8], [7.85, 8.15], [8.6, 7.55], [8.2, 8.9],
                        [4.1, 9.75], [3.95, 10.55], [5.85, 11.6], [7.65, 5.95],
                        [7.2, 5.35], [8.25, 5.35], [10.3, 9.05], [10.95, 10.85],
                        [10.95, 9.75], [11.85, 9.95]])
Y_MULTICLASS = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
Y_MULTICLASS_EXPECTED = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
X_MULTICLASS_SHAPE_EXPECTED = (
    X_MULTICLASS.shape[0] + (Y_MULTICLASS_EXPECTED.size - Y_MULTICLASS.size),
    X_MULTICLASS.shape[1])


def test_smoke(plot=False):
    """Execute k-means SMOTE with default parameters"""
    kmeans_smote = KMeansSMOTE(random_state=RND_SEED)
    X_resampled, y_resampled = kmeans_smote.fit_sample(X, Y)

    assert (np.unique(y_resampled, return_counts=True)[1]
            == np.unique(Y_EXPECTED, return_counts=True)[1]).all()
    assert (X_resampled.shape == X_SHAPE_EXPECTED)
    if plot:
        plot_resampled(X, X_resampled, Y, y_resampled, 'smoke_test')


def test_smoke_regular_kmeans(plot=False):
    """Execute k-means SMOTE with default parameters using regular k-means (not minibatch)"""
    kmeans_smote = KMeansSMOTE(
        random_state=RND_SEED, use_minibatch_kmeans=False)
    X_resampled, y_resampled = kmeans_smote.fit_sample(X, Y)

    assert (np.unique(y_resampled, return_counts=True)[1]
            == np.unique(Y_EXPECTED, return_counts=True)[1]).all()
    assert (X_resampled.shape == X_SHAPE_EXPECTED)
    if plot:
        plot_resampled(X, X_resampled, Y, y_resampled, 'smoke_test')


def test_smote_limit_case(plot=False):
    """Execute k-means SMOTE with parameters equivalent to SMOTE"""
    kmeans_smote = KMeansSMOTE(
        random_state=RND_SEED,
        imbalance_ratio_threshold=float('Inf'),
        kmeans_args={
            'n_clusters': 1
        }
    )
    smote = SMOTE(random_state=RND_SEED)
    X_resampled, y_resampled = kmeans_smote.fit_sample(X, Y)
    X_resampled_smote, y_resampled_smote = smote.fit_sample(X, Y)

    if plot:
        plot_resampled(X, X_resampled, Y, y_resampled,
                       'smote_limit_case_test_kmeans_smote')
        plot_resampled(X, X_resampled_smote, Y, y_resampled_smote,
                       'smote_limit_case_test_smote')

    assert_array_equal(X_resampled, X_resampled_smote)
    assert_array_equal(y_resampled, y_resampled_smote)


def test_random_oversampling_limit_case(plot=False):
    """Execute k-means SMOTE with parameters equivalent to random oversampling"""
    kmeans_smote = KMeansSMOTE(
        random_state=RND_SEED,
        imbalance_ratio_threshold=float('Inf'),
        kmeans_args={
            'n_clusters': 1
        },
        smote_args={
            'k_neighbors': 0
        }
    )
    random_oversampler = RandomOverSampler(random_state=RND_SEED)
    X_resampled, y_resampled = kmeans_smote.fit_sample(X, Y)
    X_resampled_random_oversampler, y_resampled_random_oversampler = random_oversampler.fit_sample(
        X, Y)

    if plot:
        plot_resampled(X, X_resampled, Y, y_resampled,
                       'random_oversampling_limit_case_test_kmeans_smote')
        plot_resampled(X, X_resampled_random_oversampler, Y, y_resampled_random_oversampler,
                       'random_oversampling_limit_case_test_random_oversampling')

    assert_array_equal(X_resampled, X_resampled_random_oversampler)
    assert_array_equal(y_resampled, y_resampled_random_oversampler)


def test_smote_fallback(plot=False):
    """Assert that regular SMOTE is applied if no minority clusters are found."""
    kmeans_smote = KMeansSMOTE(
        random_state=RND_SEED,
        kmeans_args={
            'n_clusters': 1
        }
    )
    smote = SMOTE(random_state=RND_SEED)
    with warnings.catch_warnings(record=True) as w:
        X_resampled, y_resampled = kmeans_smote.fit_sample(X, Y)

        assert len(w) == 1
        assert "No minority clusters found" in str(w[0].message)
        assert "Performing regular SMOTE" in str(w[0].message)
        assert issubclass(w[0].category, UserWarning)

        X_resampled_smote, y_resampled_smote = smote.fit_sample(X, Y)

        if plot:
            plot_resampled(X, X_resampled, Y, y_resampled,
                           'smote_fallback_test_kmeans_smote')
            plot_resampled(X, X_resampled_smote, Y, y_resampled_smote,
                           'smote_fallback_test_smote')

        assert_array_equal(X_resampled, X_resampled_smote)
        assert_array_equal(y_resampled, y_resampled_smote)

def test_smoke_multiclass(plot=False):
    """Execute k-means SMOTE with default parameters for multi-class dataset"""
    kmeans_smote = KMeansSMOTE(random_state=RND_SEED)
    X_resampled, y_resampled = kmeans_smote.fit_sample(X_MULTICLASS, Y_MULTICLASS)

    assert (np.unique(y_resampled, return_counts=True)[1]
            == np.unique(Y_MULTICLASS_EXPECTED, return_counts=True)[1]).all()
    assert (X_resampled.shape == X_MULTICLASS_SHAPE_EXPECTED)
    if plot:
        plot_resampled(X_MULTICLASS, X_resampled, Y_MULTICLASS, y_resampled, 'smoke_multiclass_test')


def test_multiclass(plot=False):
    """Execute k-means SMOTE for multi-class dataset with user-defined n_clusters"""
    kmeans_smote = KMeansSMOTE(random_state=RND_SEED, kmeans_args={'n_clusters': 10})
    X_resampled, y_resampled = kmeans_smote.fit_sample(X_MULTICLASS, Y_MULTICLASS)

    assert (np.unique(y_resampled, return_counts=True)[1]
            == np.unique(Y_MULTICLASS_EXPECTED, return_counts=True)[1]).all()
    assert (X_resampled.shape == X_MULTICLASS_SHAPE_EXPECTED)
    if plot:
        plot_resampled(X_MULTICLASS, X_resampled, Y_MULTICLASS,
                       y_resampled, 'multiclass_test')


def test_smote_limit_case_multiclass(plot=False):
    """Execute k-means SMOTE with parameters equivalent to SMOTE"""
    kmeans_smote = KMeansSMOTE(
        random_state=RND_SEED,
        imbalance_ratio_threshold=float('Inf'),
        kmeans_args={
            'n_clusters': 1
        },
        smote_args={'k_neighbors':3}
    )
    smote = SMOTE(random_state=RND_SEED, k_neighbors=3)
    X_resampled, y_resampled = kmeans_smote.fit_sample(X_MULTICLASS, Y_MULTICLASS)
    X_resampled_smote, y_resampled_smote = smote.fit_sample(X_MULTICLASS, Y_MULTICLASS)

    if plot:
        plot_resampled(X_MULTICLASS, X_resampled, Y_MULTICLASS, y_resampled,
                       'smote_limit_case_multiclass_test_kmeans_smote')
        plot_resampled(X_MULTICLASS, X_resampled_smote, Y_MULTICLASS, y_resampled_smote,
                       'smote_limit_case_multiclass_test_smote')

    assert_array_equal(X_resampled, X_resampled_smote)
    assert_array_equal(y_resampled, y_resampled_smote)

def plot_resampled(X_original, X_resampled, y_original, y_resampled, test_name, save_path='.'):
    """Create a colored scatter plot of X_resampled and save the image to disk"""
    import matplotlib.pyplot as plt
    y_resampled[y_original.size:] = y_resampled[y_original.size:] + 100
    plt.subplots()
    for label in np.unique(y_resampled):
        if label < 100:
            color = ['r', 'b', 'g'][label]
            marker = 'o'
        else:
            color = ['r', 'b', 'g'][label-100]
            marker = '+'
        plt.scatter(
            X_resampled[y_resampled == label, 0],
            X_resampled[y_resampled == label, 1],
            c=color,
            marker=marker
        )
    plt.gcf().savefig('{}/scatter_{}.png'.format(save_path, test_name))
