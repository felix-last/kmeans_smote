"""Test oversampling method k-means SMOTE"""
# Authors: Felix Last
# License: MIT

# from unittest import assertTrue
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

def test_smoke(plot=False):
    """Execute k-means SMOTE with default parameters"""
    kmeans_smote = KMeansSMOTE(random_state=RND_SEED)
    X_resampled, y_resampled = kmeans_smote.fit_sample(X, Y)

    assert (np.unique(y_resampled, return_counts=True)[1]
        == np.unique(Y_EXPECTED, return_counts=True)[1]).all()
    assert (X_resampled.shape == X_SHAPE_EXPECTED)
    if plot:
        plot_resampled(X_resampled, y_resampled, 'smoke_test')


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
        plot_resampled(X_resampled, y_resampled,
                       'smote_limit_case_test_kmeans_smote')
        plot_resampled(X_resampled_smote, y_resampled_smote,
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
        plot_resampled(X_resampled, y_resampled,
                       'random_oversampling_limit_case_test_kmeans_smote')
        plot_resampled(X_resampled_random_oversampler, y_resampled_random_oversampler,
                       'random_oversampling_limit_case_test_random_oversampling')

    assert_array_equal(X_resampled, X_resampled_random_oversampler)
    assert_array_equal(y_resampled, y_resampled_random_oversampler)


def plot_resampled(X_resampled, y_resampled, test_name, save_path='.'):
    """Create a colored scatter plot of X_resampled and save the image to disk"""
    import matplotlib.pyplot as plt
    y_resampled[[i for i, obs in enumerate(X_resampled) if obs not in X]] = 2
    plt.subplots()
    plt.scatter(
        X_resampled[:, 0],
        X_resampled[:, 1],
        c=np.asarray(['r', 'b', 'g'])[y_resampled.tolist()]
    )
    plt.gcf().savefig('{}/scatter_{}.png'.format(save_path, test_name))
