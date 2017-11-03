Oversampling for Imbalanced Learning based on K-Means and SMOTE
---------------------------------------------------------------

K-means SMOTE is an oversampling method for class-imbalanced data. It
aids classification by generating minority class samples in safe and
crucial areas of the input space. The method avoids the generation of
noise and effectively overcomes imbalances between and within classes.

This project is a python implementation of k-means SMOTE. It is
compatible with the scikit-learn-contrib project
`imbalanced-learn <https://github.com/scikit-learn-contrib/imbalanced-learn>`__.

Installation
------------

Dependencies
~~~~~~~~~~~~

The implementation is tested under python 3.6 and works with the latest
release of the imbalanced-learn framework:

-  imbalanced-learn (>= 0.3.1)

Installation
~~~~~~~~~~~~

Pypi
^^^^

.. code:: sh

    pip install kmeans_smote

From Source
^^^^^^^^^^^

Clone this repository and run the setup.py file. Use the following
commands to get a copy from GitHub and install all dependencies:

.. code:: sh

    git clone https://github.com/felix-last/kmeans_smote.git
    cd kmeans-smote
    pip install .

Documentation
-------------

Find the API documentation at https://kmeans_smote.readthedocs.io. As
this project follows the imbalanced-learn API, the `imbalanced-learn
documentation <http://contrib.scikit-learn.org/imbalanced-learn>`__
might also prove helpful.

Example Usage
~~~~~~~~~~~~~

.. code:: python

    import numpy as np
    from imblearn.datasets import fetch_datasets
    from kmeans_smote import KMeansSMOTE

    datasets = fetch_datasets(filter_data=['oil'])
    X, y = datasets['oil']['data'], datasets['oil']['target']

    [print('Class {} has {} instances'.format(label, count))
     for label, count in zip(*np.unique(y, return_counts=True))]

    kmeans_smote = KMeansSMOTE(
        kmeans_args={
            'n_clusters': 100
        },
        smote_args={
            'k_neighbors': 10
        }
    )
    X_resampled, y_resampled = kmeans_smote.fit_sample(X, y)

    [print('Class {} has {} instances after oversampling'.format(label, count))
     for label, count in zip(*np.unique(y_resampled, return_counts=True))]

Expected Output:

::

    Class -1 has 896 instances
    Class 1 has 41 instances
    Class -1 has 896 instances after oversampling
    Class 1 has 896 instances after oversampling

Take a look at `imbalanced-learn
pipelines <http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.pipeline.Pipeline.html>`__
for efficient usage with cross-validation.

About
-----

K-means SMOTE works in three steps:

1. Cluster the entire input space using k-means [1].
2. Distribute the number of samples to generate across clusters:

   1. Filter out clusters which have a high number of majority class
      samples.
   2. Assign more synthetic samples to clusters where minority class
      samples are sparsely distributed.

3. Oversample each filtered cluster using SMOTE [2].

Contributing
~~~~~~~~~~~~

Please feel free to submit an issue if things work differently than
expected. Pull requests are also welcome.

Citation
~~~~~~~~

If you use k-means SMOTE in a scientific publication, we would
appreciate citations to the following
`paper <https://arxiv.org/abs/1711.00837>`__:

::

    @article{kmeans_smote,
        title = {Oversampling for Imbalanced Learning Based on K-Means and SMOTE},
        author = {Last, Felix and Douzas, Georgios and Bacao, Fernando},
        year = {2017},
        archivePrefix = "arXiv",
        eprint = "1711.00837",
        primaryClass = "cs.LG"
    }

References
~~~~~~~~~~

[1] MacQueen, J. “Some Methods for Classification and Analysis of
Multivariate Observations.” Proceedings of the Fifth Berkeley Symposium
on Mathematical Statistics and Probability, 1967, p. 281297.

[2] Chawla, Nitesh V., et al. “SMOTE: Synthetic Minority over-Sampling
Technique.” Journal of Artificial Intelligence Research, vol. 16, Jan.
2002, p. 321357, doi:10.1613/jair.953.
