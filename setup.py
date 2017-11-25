import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()


tests_require = [
    'pytest>=3.2.5, < 3.3'
],

setup(
    name='kmeans_smote',
    version="0.0.2",
    py_modules=['kmeans_smote'],
    install_requires=[
        'imbalanced-learn>=0.3.1, <0.4',
        'numpy>=1.13, <1.14',
        'scikit-learn>=0.19.0, <0.20'
    ],
    tests_require=tests_require,
    extras_require={
        'test':tests_require
    },
    author="Felix Last",
    author_email="mail@felixlast.de",
    url="https://github.com/felix-last/kmeans_smote",
    description=("Oversampling for imbalanced learning based on k-means and SMOTE"),
    long_description=read('README.rst'),
    license="MIT",
    keywords=[
        'Class-imbalanced Learning',
        'Oversampling',
        'Classification',
        'Clustering',
        'Supervised Learning'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ]
)
