import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='kmeans_smote',
    version="0.0.1",
    py_modules=['kmeans_smote'],
    install_requires=['imbalanced-learn>=0.3.1','scipy>=0.13.3','numpy>=1.8.2','scikit-learn>=0.19.0'],
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
