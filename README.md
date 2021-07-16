# IsoTree

Fast and multi-threaded implementation of Extended Isolation Forest, Fair-Cut Forest, SCiForest (a.k.a. Split-Criterion iForest), and regular Isolation Forest, for outlier/anomaly detection, plus additions for imputation of missing values, distance/similarity calculation between observations, and handling of categorical data. Written in C++ with interfaces for Python and R. An additional wrapper for Ruby can be found [here](https://github.com/ankane/isotree).

The new concepts in this software are described in:
* [Distance approximation using Isolation Forests](https://arxiv.org/abs/1910.12362)
* [Imputing missing values with unsupervised random trees](https://arxiv.org/abs/1911.06646)

# Description

Isolation Forest is an algorithm originally developed for outlier detection that consists in splitting sub-samples of the data according to some attribute/feature/column at random. The idea is that, the rarer the observation, the more likely it is that a random uniform split on some feature would put outliers alone in one branch, and the fewer splits it will take to isolate an outlier observation like this. The concept is extended to splitting hyperplanes in the extended model (i.e. splitting by more than one column at a time), and to guided (not entirely random) splits in the SCiForest model that aim at isolating outliers faster and finding clustered outliers.

Note that this is a black-box model that will not produce explanations or importances - for a different take on explainable outlier detection see [OutlierTree](https://www.github.com/david-cortes/outliertree).

![image](image/density_regions.png "density regions")

_(Code to produce these plots can be found in the R examples in the documentation)_

# Comparison against other libraries

The folder [timings](https://github.com/david-cortes/isotree/blob/master/timings) contains a speed comparison against other Isolation Forest implementations in Python (SciKit-Learn, EIF) and R (IsolationForest, isofor, solitude). From the benchmarks, IsoTree tends to be at least 1 order of magnitude faster than the libraries compared against in both single-threaded and multi-threaded mode.

Example timings for 100 trees and different sample sizes, CovType dataset - see the link above for full benchmark and details:

| Library         |  Model | Time (s) 256 | Time (s) 1024 | Time (s) 10k  |
| :---:           |  :---: | :---:        | :---:         | :---:         |
| isotree         | orig   |  0.00161     | 0.00631       | 0.0848        |
| isotree         | ext    |  0.00326     | 0.0123        | 0.168         |
| eif             | orig   |  0.149       | 0.398         | 4.99          |
| eif             | ext    |  0.16        | 0.428         | 5.06          |
| scikit-learn    | orig   |  8.3         | 8.01          | 6.89          |
| solitude        | orig   |  32.612      | 34.01         | 41.01         |


Example AUC as outlier detector in typical datasets (notebook to produce results [here](https://github.com/david-cortes/isotree/blob/master/example/comparison_model_quality.ipynb)):

* Satellite dataset:

| Library      | AUC defaults | AUC grid search |
| :---:        | :---:        | :---:           |
| isotree      | 0.70         | 0.84            |
| eif          | -            | 0.714           |
| scikit-learn | 0.687        | 0.74            |

* Antthyroid dataset:

| Library      | AUC defaults | AUC grid search |
| :---:        | :---:        | :---:           |
| isotree      | 0.80         | 0.982           |
| eif          | -            | 0.808           |
| scikit-learn | 0.836        | 0.836           |

*(Disclaimer: these are rather small datasets and thus these AUC estimates have high variance)*

# Distance / similarity calculations

General idea was extended to produce distance (alternatively, similarity) between observations according to how many random splits it takes to separate them - idea is described in ["Distance approximation using Isolation Forests"](https://arxiv.org/abs/1910.12362).

# Imputation of missing values

The model can also be used to impute missing values in a similar fashion as kNN, by taking the values from observations in the terminal nodes of each tree in which an observation with missing values falls at prediction time, combining the non-missing values of the other observations as a weighted average according to the depth of the node and the number of observations that fall there. This is not related to how the model handles missing values internally, but is rather meant as a faster way of imputing by similarity. Quality is usually not as good as chained equations, but the method is a lot faster and more scalable. Recommended to use non-random splits when used as an imputer. Details are described in ["Imputing missing values with unsupervised random trees"](https://arxiv.org/abs/1911.06646).

# Highlights

There's already many available implementations of isolation forests for both Python and R (such as [the one from the original paper's authors'](https://sourceforge.net/projects/iforest/) or [the one in SciKit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)), but at the time of writing, all of them are lacking some important functionality and/or offer sub-optimal speed. This particular implementation offers the following:

* Implements the extended model (with splitting hyperplanes) and split-criterion model (with non-random splits).
* Can handle missing values (but performance with them is not so good).
* Can handle categorical variables (one-hot/dummy encoding does not produce the same result).
* Can use a mixture of random and non-random splits, and can split by weighted/pooled gain (in addition to simple average).
* Can produce approximated pairwise distances between observations according to how many steps it takes on average to separate them down the tree.
* Can produce missing value imputations according to observations that fall on each terminal node.
* Can work with sparse matrices.
* Supports sample/observation weights, either as sampling importance or as distribution density measurement.
* Supports user-provided column sample weights.
* Can sample columns randomly with weights given by kurtosis.
* Uses exact formula (not approximation as others do) for harmonic numbers at lower sample and remainder sizes, and a higher-order approximation for larger sizes.
* Can fit trees incrementally to user-provided data samples.
* Produces serializable model objects with reasonable file sizes.
* Can translate the generated trees into SQL statements.
* Fast and multi-threaded C++ code. Can be wrapped in languages other than Python/R/Ruby.

(Note that categoricals, NAs, and density-like sample weights, are treated heuristically with different options as there is no single logical extension of the original idea to them, and having them present might degrade performance/accuracy for regular numerical non-missing observations)

# Installation

* Python:
```python
pip install isotree
```
or if that fails:
```
pip install --no-use-pep517 isotree
```

**Note for macOS users:** on macOS, the Python version of this package will compile **without** multi-threading capabilities. This is due to default apple's redistribution of `clang` not providing OpenMP modules, and aliasing it to `gcc` which causes confusions in build scripts. If you have a non-apple version of `clang` with the OpenMP modules, or if you have `gcc` installed, you can compile this package with multi-threading enabled by setting up an environment variable `ENABLE_OMP=1`:
```
export ENABLE_OMP=1
pip install isotree
```
(Alternatively, can also pass argument `enable-omp` to the `setup.py` file: `python setup.py install enable-omp`)

* R:

```r
install.packages("isotree")
```

* C++:
```
git clone --recursive https://www.github.com/david-cortes/isotree.git
cd isotree
mkdir build
cd build
cmake ..
make

### for a system-wide install in linux
sudo make install
sudo ldconfig
```

(Will build as a shared object - linkage is then done with `-lisotree`)

* Ruby

See [external repository with wrapper](https://github.com/ankane/isotree).

# Sample usage

**Warning: default parameters in this implementation are very different from default parameters in others such as SciKit-Learn's, and these defaults won't scale to large datasets (see documentation for details).**

* Python:

(Library is SciKit-Learn compatible)

```python
import numpy as np
from isotree import IsolationForest

### Random data from a standard normal distribution
np.random.seed(1)
n = 100
m = 2
X = np.random.normal(size = (n, m))

### Will now add obvious outlier point (3, 3) to the data
X = np.r_[X, np.array([3, 3]).reshape((1, m))]

### Fit a small isolation forest model
iso = IsolationForest(ntrees = 10, ndim = 2, nthreads = 1)
iso.fit(X)

### Check which row has the highest outlier score
pred = iso.predict(X)
print("Point with highest outlier score: ",
      X[np.argsort(-pred)[0], ])
```

* R:
(see documentation for more examples - `help(isotree::isolation.forest)`)
```r
### Random data from a standard normal distribution
library(isotree)
set.seed(1)
n <- 100
m <- 2
X <- matrix(rnorm(n * m), nrow = n)

### Will now add obvious outlier point (3, 3) to the data
X <- rbind(X, c(3, 3))

### Fit a small isolation forest model
iso <- isolation.forest(X, ntrees = 10, nthreads = 1)

### Check which row has the highest outlier score
pred <- predict(iso, X)
cat("Point with highest outlier score: ",
    X[which.max(pred), ], "\n")
```

* C++:

See file [isotree_cpp_ex.cpp](https://github.com/david-cortes/isotree/blob/master/example/isotree_cpp_ex.cpp).

* Ruby

See [external repository with wrapper](https://github.com/ankane/isotree).

# Examples

* Python: example notebook [here](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_example.ipynb), (also example as imputer in sklearn pipeline [here](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_impute.ipynb)).
* R: examples available in the documentation (`help(isotree::isolation.forest)`, [link to CRAN](https://cran.r-project.org/web/packages/isotree/index.html)).
* C++: see short example in the section above.
* Ruby: see [external repository with wrapper](https://github.com/ankane/isotree).

# Documentation

* Python: documentation is available at [ReadTheDocs](http://isotree.readthedocs.io/en/latest/).
* R: documentation is available internally in the package (e.g. `help(isolation.forest)`) and in [CRAN](https://cran.r-project.org/web/packages/isotree/index.html).
* C++: documentation is available in the public header (`include/isotree.hpp`) and in the source files.
* Ruby: see [external repository with wrapper](https://github.com/ankane/isotree) for the syntax and the [Python docs](http://isotree.readthedocs.io) for details about the parameters.

# References

* Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." 2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
* Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation-based anomaly detection." ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
* Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner. "Extended Isolation Forest." arXiv preprint arXiv:1811.02141 (2018).
* Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "On detecting clustered anomalies using SCiForest." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2010.
* https://sourceforge.net/projects/iforest/
* https://math.stackexchange.com/questions/3388518/expected-number-of-paths-required-to-separate-elements-in-a-binary-tree
* Quinlan, J. Ross. C4. 5: programs for machine learning. Elsevier, 2014.
* Cortes, David. "Distance approximation using Isolation Forests." arXiv preprint arXiv:1910.12362 (2019).
* Cortes, David. "Imputing missing values with unsupervised random trees." arXiv preprint arXiv:1911.06646 (2019).
