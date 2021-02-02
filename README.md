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

Speed comparison, fitting 100 trees of varying sample sizes (256, 2014, 10k) to datasets of varying sizes. The code can be found under folder [timings](https://github.com/david-cortes/isotree/blob/master/timings). The timings are taken on a CPU AMD Ryzen 7 2700 running at 3.2GHz, with 16 threads. Note that not all libraries support multi-threading or sparse inputs.

* Satellite (6435 rows, 36 columns)

| Library         |  Model | Threads | Lang  | Time (s) 256 | Time (s) 1024 | Time (s) 6435 |
| :---:           |  :---: |  :---:  | :---: | :---:        | :---:         | :---:         |
| isotree         | orig   |   1     | Py    |  0.00531     | 0.0107        | 0.0522        |
| isotree         | ext    |   1     | Py    |  0.011       | 0.0317        | 0.186         |
| scikit-learn    | orig   |   1     | Py    |  0.166       | 0.17          | 0.233         |
| eif             | orig   |   1     | Py    |  0.0989      | 0.325         | 2.18          |
| eif             | ext    |   1     | Py    |  0.0943      | 0.333         | 2.21          |
| isotree         | orig   |   1     | R     |  0.00815     | 0.0173        | 0.0728        |
| isotree         | ext    |   1     | R     |  0.0155      | 0.0435        | 0.239         |
| IsolationForest | orig   |   1     | R     |  0.146       | 0.248         | 0.845         |
| isofor          | orig   |   1     | R     |  8.34        | 22.08         | 130.26        |
| solitude        | orig   |   1     | R     |  0.691       | 1.071         | 4.158         |
| isotree         | orig   |   16    | Py    |  0.000875    | 0.00164       | 0.00641       |
| isotree         | ext    |   16    | Py    |  0.00224     | 0.00563       | 0.0254        |
| scikit-learn    | orig   |   16    | Py    |  0.305       | 0.305         | 0.277         |
| isotree         | orig   |   16    | R     |  0.00564     | 0.00999       | 0.0278        |
| isotree         | ext    |   16    | R     |  0.00789     | 0.0152        | 0.0511        |
| solitude        | orig   |   16    | R     |  0.448       | 0.523         | 0.903         |

* CovType (581,012 rows, 54 columns)

| Library         |  Model | Threads | Lang  | Time (s) 256 | Time (s) 1024 | Time (s) 10k  |
| :---:           |  :---: |  :---:  | :---: | :---:        | :---:         | :---:         |
| isotree         | orig   |   1     | Py    |  0.00772     | 0.0284        | 0.326         |
| isotree         | ext    |   1     | Py    |  0.0139      | 0.0532        | 0.604         |
| scikit-learn    | orig   |   1     | Py    |  10.1        | 10.6          | 11.1          |
| eif             | orig   |   1     | Py    |  0.149       | 0.398         | 4.99          |
| eif             | ext    |   1     | Py    |  0.16        | 0.428         | 5.06          |
| isotree         | orig   |   1     | R     |  0.0494      | 0.112         | 0.443         |
| isotree         | ext    |   1     | R     |  0.058       | 0.103         | 0.743         |
| IsolationForest | orig   |   1     | R     |  oom         | oom           | oom           |
| isofor          | orig   |   1     | R     |  timeout     | timeout       | timeout       |
| solitude        | orig   |   1     | R     |  48.4        | 51.07         | 85.5          |
| isotree         | orig   |   16    | Py    |  0.00161     | 0.00631       | 0.0848        |
| isotree         | ext    |   16    | Py    |  0.00326     | 0.0123        | 0.168         |
| scikit-learn    | orig   |   16    | Py    |  8.3         | 8.01          | 6.89          |
| isotree         | orig   |   16    | R     |  0.0454      | 0.5317        | 0.148         |
| isotree         | ext    |   16    | R     |  0.05        | 0.058         | 0.234         |
| solitude        | orig   |   16    | R     |  32.612      | 34.01         | 41.01         |

* RCV1 (804,414 rows, 47,236 columns, sparse format)

| Library         |  Model | Threads | Lang  | Time (s) 256 | Time (s) 1024 | Time (s) 10k  |
| :---:           |  :---: |  :---:  | :---: | :---:        | :---:         | :---:         |
| isotree         | orig   |   1     | Py    |  0.0677      | 0.118         | 0.49          |
| isotree         | ext    |   1     | Py    |  0.152       | 0.249         | 0.844         |
| scikit-learn    | orig   |   1     | Py    |  30.9        | 31.6          | 32.8          |
| isotree         | orig   |   16    | Py    |  0.0456      | 0.0513        | 0.0977        |
| isotree         | ext    |   16    | Py    |  0.0587      | 0.0711        | 0.145         |
| scikit-learn    | orig   |   4     | Py    |  17.8        | 18.1          | 18.5          |
| scikit-learn    | orig   |   16    | Py    |  oom         | oom           | oom           |

*Note: these datasets have mostly discrete values. Some libraries such as SciKit-Learn might perform much faster when columns have continuous values*

# Distance / similarity calculations

General idea was extended to produce distance (alternatively, similarity) between observations according to how many random splits it takes to separate them - idea is described in ["Distance approximation using Isolation Forests"](https://arxiv.org/abs/1910.12362).

# Imputation of missing values

The model can also be used to impute missing values in a similar fashion as kNN, by taking the values from observations in the terminal nodes of each tree in which an observation with missing values falls at prediction time, combining the non-missing values of the other observations as a weighted average according to the depth of the node and the number of observations that fall there. This is not related to how the model handles missing values internally, but is rather meant as a faster way of imputing by similarity. Quality is usually not as good as chained equations, but the method is a lot faster and more scalable. Recommended to use non-random splits when used as an imputer. Details are described in ["Imputing missing values with unsupervised random trees"](https://arxiv.org/abs/1911.06646).

# Highlights

There's already many available implementations of isolation forests for both Python and R (such as [the one from the original paper's authors'](https://sourceforge.net/projects/iforest/) or [the one in SciKit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)), but at the time of writing, all of them are lacking some important functionality and/or offer sub-optimal speed. This particular implementation offers the following:

* Implements the extended model (with splitting hyperplanes) and split-criterion model (with non-random splits).
* Can use a mixture of random and non-random splits, and can split by weighted/pooled gain (in addition to simple average).
* Can produce approximated pairwise distances between observations according to how many steps it takes on average to separate them down the tree.
* Can handle missing values (but performance with them is not so good).
* Can produce missing value imputations according to observations that fall on each terminal node.
* Can handle categorical variables (one-hot/dummy encoding does not produce the same result).
* Can work with sparse matrices.
* Supports sample/observation weights, either as sampling importance or as distribution density measurement.
* Supports user-provided column sample weights.
* Can sample columns randomly with weights given by kurtosis.
* Uses exact formula (not approximation as others do) for harmonic numbers at lower sample and remainder sizes.
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
git clone https://www.github.com/david-cortes/isotree.git
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


# Examples

* Python: example notebook [here](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_example.ipynb), (also example as imputer in sklearn pipeline [here](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_impute.ipynb)).
* R: examples available in the documentation (`help(isotree::isolation.forest)`, [link to CRAN](https://cran.r-project.org/web/packages/isotree/index.html)).
* C++: see short example in the section above.

# Documentation

* Python: documentation is available at [ReadTheDocs](http://isotree.readthedocs.io/en/latest/).
* R: documentation is available internally in the package (e.g. `help(isolation.forest)`) and in [CRAN](https://cran.r-project.org/web/packages/isotree/index.html).
* C++: documentation is available in the public header (`include/isotree.hpp`) and in the source files.

# Known issues

When setting a random seed and using more than one thread, the results of the imputation functions are not 100% reproducible to the last decimal. This is due to parallelized aggregations, and thus the only "fix" is to limit oneself to only one thread. The trees themselves are however not affected by this, and neither is the isolation depth (main functionality of the package).

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
