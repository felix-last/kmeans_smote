"""K-Means SMOTE oversampling method for class-imbalanced data"""

# Authors: Felix Last
# License: MIT

import warnings
import math
import copy
import numpy as np

from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.exceptions import raise_isinstance_error
from imblearn.utils import check_neighbors_object
from imblearn.utils.deprecation import deprecate_parameter

class KMeansSMOTE(BaseOverSampler):
    """Class to perform oversampling using K-Means SMOTE.

    K-Means SMOTE works in three steps:

    1. Cluster the entire input space using k-means.
    2. Distribute the number of samples to generate across clusters:

        1. Select clusters which have a high number of minority class samples.
        2. Assign more synthetic samples to clusters where minority class samples are sparsely distributed.

    3. Oversample each filtered cluster using SMOTE.

    The method implements SMOTE and random oversampling as limit cases. Therefore, the following configurations
    may be used to achieve the behavior of ...

    ... SMOTE: ``imbalance_ratio_threshold=float('Inf'), kmeans_args={'n_clusters':1}``

    ... random oversampling: ``imbalance_ratio_threshold=float('Inf'), kmeans_args={'n_clusters':1}, smote_args={'k_neighbors':0})``

    Parameters
    ----------
    sampling_strategy : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.

        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for oversampling methods and ``'not
          minority'`` for undersampling methods. The classes targeted will be
          oversampled or undersampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.
        Will be copied to kmeans_args and smote_args if not explicitly passed there.

    kmeans_args : dict, optional (default={})
        Parameters to be passed to ``sklearn.cluster.KMeans`` or ``sklearn.cluster.MiniBatchKMeans``
        (see ``use_minibatch_kmeans``). If n_clusters is not explicitly set, scikit-learn's
        default will apply.

    smote_args : dict, optional (default={})
        Parameters to be passed to ``imblearn.over_sampling.SMOTE``. Note that ``k_neighbors`` is automatically
        adapted without warning when a cluster is smaller than the number of neighbors specified.
        `sampling_strategy` will be overwritten according to sampling_strategy passed to this class. `random_state`
        will be passed from this class if none is specified.

    imbalance_ratio_threshold : float or dict, optional (default=1.0)
        Specify a threshold for a cluster's imbalance ratio  ``((majority_count + 1) / (minority_count + 1))``.
        Only clusters with an imbalance ratio less than the threshold are oversampled. Use a dictionary to specify
        different thresholds for different minority classes.

    density_power : float, optional (default=None)
        Used to compute the density of minority samples within each cluster. By default, the number of features will be used.

    use_minibatch_kmeans : boolean, optional (default=True)
        If False, use ``sklearn.cluster.KMeans``. If True, use ``sklearn.cluster.MiniBatchKMeans``.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible. This parameter will be copied to ``kmeans_args`` and
        ``smote_args`` if not explicitly passed there. Note: ``MiniBatchKMeans`` does not accept ``n_jobs``.

    Examples
    --------

    >>> import numpy as np
    >>> from imblearn.datasets import fetch_datasets
    >>> from kmeans_smote import KMeansSMOTE
    >>>
    >>> datasets = fetch_datasets(filter_data=['oil'])
    >>> X, y = datasets['oil']['data'], datasets['oil']['target']
    >>>
    >>> [print('Class {} has {} instances'.format(label, count))
    ...  for label, count in zip(*np.unique(y, return_counts=True))]
    >>>
    >>> kmeans_smote = KMeansSMOTE(
    ...     kmeans_args={
    ...         'n_clusters': 100
    ...     },
    ...     smote_args={
    ...        'k_neighbors': 10
    ...     }
    ... )
    >>> X_resampled, y_resampled = kmeans_smote.fit_sample(X, y)
    >>>
    >>> [print('Class {} has {} instances after oversampling'.format(label, count))
    ...  for label, count in zip(*np.unique(y_resampled, return_counts=True))]
    """

    def __init__(self,
                sampling_strategy='auto',
                random_state=None,
                kmeans_args={},
                smote_args={},
                imbalance_ratio_threshold=1.0,
                density_power=None,
                use_minibatch_kmeans=True,
                n_jobs=1,
                **kwargs):
        super(KMeansSMOTE, self).__init__(sampling_strategy=sampling_strategy, **kwargs)
        self.imbalance_ratio_threshold = imbalance_ratio_threshold
        self.kmeans_args = copy.deepcopy(kmeans_args)
        self.smote_args = copy.deepcopy(smote_args)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_minibatch_kmeans = use_minibatch_kmeans

        self.density_power = density_power

    def _cluster(self, X):
        """Run k-means to cluster the dataset

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        Returns
        -------
        cluster_assignment : ndarray, shape (n_samples)
            The corresponding cluster labels of ``X``.
        """

        if self.use_minibatch_kmeans:
            from sklearn.cluster import MiniBatchKMeans as KMeans
        else:
            from sklearn.cluster import KMeans as KMeans

        kmeans = KMeans(**self.kmeans_args)
        if self.use_minibatch_kmeans and 'init_size' not in self.kmeans_args:
            self.kmeans_args['init_size'] = min(2 * kmeans.n_clusters, X.shape[0])
            kmeans = KMeans(**self.kmeans_args)

        kmeans.fit_transform(X)
        cluster_assignment = kmeans.labels_
        # kmeans.labels_ does not use continuous labels,
        # i.e. some labels in 0..n_clusters may not exist. Tidy up this mess.
        return cluster_assignment

    def _filter_clusters(self, X, y, cluster_assignment, minority_class_label):
        """Determine sampling weight for each cluster.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.
        cluster_assignment : ndarray, shape (n_samples)
            The corresponding cluster labels of ``X``.
        minority_class_label : int
            Label of the minority class to filter by.

        Returns
        -------
        sampling_weights : ndarray, shape (np.max(np.unique(cluster_assignment)),)
            Vector of sampling weights for each cluster
        """
        # compute the shape of the density factors
        # since the cluster labels are not continuous, make it large enough
        # to fit all values up to the largest cluster label
        largest_cluster_label = np.max(np.unique(cluster_assignment))
        sparsity_factors = np.zeros((largest_cluster_label + 1,), dtype=np.float64)
        minority_mask = (y == minority_class_label)
        sparsity_sum = 0
        imbalance_ratio_threshold = self.imbalance_ratio_threshold
        if isinstance(imbalance_ratio_threshold, dict):
            imbalance_ratio_threshold = imbalance_ratio_threshold[minority_class_label]

        for i in np.unique(cluster_assignment):
            cluster = X[cluster_assignment == i]
            mask = minority_mask[cluster_assignment == i]
            minority_count = cluster[mask].shape[0]
            majority_count = cluster[~mask].shape[0]
            imbalance_ratio = (majority_count + 1) / (minority_count + 1)
            if (imbalance_ratio < imbalance_ratio_threshold) and (minority_count > 1):
                distances = euclidean_distances(cluster[mask])
                non_diagonal_distances = distances[
                    ~np.eye(distances.shape[0], dtype=np.bool)
                ]
                average_minority_distance = np.mean( non_diagonal_distances )
                if average_minority_distance is 0: average_minority_distance = 1e-1 # to avoid division by 0
                density_factor = minority_count / (average_minority_distance ** self.density_power)
                sparsity_factors[i] = 1 / density_factor

        # prevent division by zero; set zero weights in majority clusters
        sparsity_sum = sparsity_factors.sum()
        if sparsity_sum == 0:
            sparsity_sum = 1 # to avoid division by zero
        sparsity_sum = np.full(sparsity_factors.shape, sparsity_sum, np.asarray(sparsity_sum).dtype)
        sampling_weights = (sparsity_factors / sparsity_sum)

        return sampling_weights


    def _fit_resample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding labels of ``X_resampled``

        """
        self._set_subalgorithm_params()

        if self.density_power is None:
            self.density_power = X.shape[1]

        resampled = [ (X.copy(), y.copy()) ]
        sampling_ratio = {k: v for k, v in self.sampling_strategy_.items()}
        # sampling_strategy_ does not contain classes where n_samples 0
        for class_label in np.unique(y):
            if class_label not in sampling_ratio:
                sampling_ratio[class_label] = 0
        for minority_class_label, n_samples in sampling_ratio.items():
            if n_samples == 0:
                continue

            cluster_assignment = self._cluster(X)
            sampling_weights = self._filter_clusters(X, y, cluster_assignment, minority_class_label)
            smote_args = self.smote_args.copy()
            if np.count_nonzero(sampling_weights) > 0:
                # perform k-means smote
                for i in np.unique(cluster_assignment):
                    cluster_X = X[cluster_assignment == i]
                    cluster_y = y[cluster_assignment == i]
                    if sampling_weights[i] > 0:
                        # determine ratio for oversampling the current cluster
                        target_ratio = {label: np.count_nonzero(cluster_y == label) for label in sampling_ratio}
                        cluster_minority_count = np.count_nonzero(cluster_y == minority_class_label)
                        generate_count = int(round(n_samples * sampling_weights[i]))
                        target_ratio[minority_class_label] = generate_count + cluster_minority_count

                        # make sure that cluster_y has more than 1 class, adding a random point otherwise
                        remove_index = -1
                        if np.unique(cluster_y).size < 2:
                            remove_index = cluster_y.size
                            cluster_X = np.append(cluster_X, np.zeros((1,cluster_X.shape[1])), axis=0)
                            majority_class_label = next( key for key in sampling_ratio.keys() if key != minority_class_label )
                            target_ratio[majority_class_label] = 1 + target_ratio[majority_class_label]
                            cluster_y = np.append(cluster_y, np.asarray(majority_class_label).reshape((1,)), axis=0)

                        # clear target ratio of labels not present in cluster
                        for label in list(target_ratio.keys()):
                            if label not in cluster_y:
                                del target_ratio[label]

                        # modify copy of the user defined smote_args to reflect computed parameters
                        smote_args['sampling_strategy'] = target_ratio

                        smote_args = self._validate_smote_args(smote_args, cluster_minority_count)
                        oversampler = SMOTE(**smote_args)

                        # if k_neighbors is 0, perform random oversampling instead of smote
                        if 'k_neighbors' in smote_args and smote_args['k_neighbors'] == 0:
                                oversampler_args = {}
                                if 'random_state' in smote_args:
                                    oversampler_args['random_state'] = smote_args['random_state']
                                oversampler = RandomOverSampler(**oversampler_args)

                        # finally, apply smote to cluster
                        with warnings.catch_warnings():
                            # ignore warnings about minority class getting bigger than majority class
                            # since this would only be true within this cluster
                            warnings.filterwarnings(action='ignore', category=UserWarning, message=r'After over-sampling, the number of samples \(.*\) in class .* will be larger than the number of samples in the majority class \(class #.* \-\> .*\)')
                            cluster_resampled_X, cluster_resampled_y = oversampler.fit_sample(cluster_X, cluster_y)

                        if remove_index > -1:
                            # since SMOTE's results are ordered the same way as the data passed into it,
                            # the temporarily added point is at the same index position as it was added.
                            for l in [cluster_resampled_X, cluster_resampled_y, cluster_X, cluster_y]:
                                np.delete(l, remove_index, 0)

                        # add new generated samples to resampled
                        resampled.append( (
                            cluster_resampled_X[cluster_y.size:,:],
                            cluster_resampled_y[cluster_y.size:]))
            else:
                # all weights are zero -> perform regular smote
                warnings.warn('No minority clusters found for class {}. Performing regular SMOTE. Try changing the number of clusters.'.format(minority_class_label))
                target_ratio = {label: np.count_nonzero(y == label) for label in sampling_ratio}
                target_ratio[minority_class_label] = sampling_ratio[minority_class_label]
                minority_count = np.count_nonzero(y == minority_class_label)
                smote_args = self._validate_smote_args(smote_args, minority_count)
                oversampler = SMOTE(**smote_args)
                X_smote, y_smote = oversampler.fit_sample(X, y)
                resampled.append((
                    X_smote[y.size:,:],
                    y_smote[y.size:]))


        resampled = list(zip(*resampled))
        if(len(resampled) > 0):
            X_resampled = np.concatenate(resampled[0], axis=0)
            y_resampled = np.concatenate(resampled[1], axis=0)
        return X_resampled, y_resampled


    def _validate_smote_args(self, smote_args, minority_count):
        # determine max number of nearest neighbors considering sample size
        max_k_neighbors =  minority_count - 1
        # check if max_k_neighbors is violated also considering smote's default
        smote = SMOTE(**smote_args)
        if smote.k_neighbors > max_k_neighbors:
            smote_args['k_neighbors'] = max_k_neighbors
            smote = SMOTE(**smote_args)
        return smote_args

    def _set_subalgorithm_params(self):
        # copy random_state to sub-algorithms
        if self.random_state is not None:
            if 'random_state' not in self.smote_args:
                    self.smote_args['random_state'] = self.random_state
            if 'random_state' not in self.kmeans_args:
                self.kmeans_args['random_state'] = self.random_state

        # copy n_jobs to sub-algorithms
        if self.n_jobs is not None:
            if 'n_jobs' not in self.smote_args:
                    self.smote_args['n_jobs'] = self.n_jobs
            if 'n_jobs' not in self.kmeans_args:
                if not self.use_minibatch_kmeans:
                    self.kmeans_args['n_jobs'] = self.n_jobs
