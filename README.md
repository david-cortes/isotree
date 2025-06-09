# IsoTree

Fast and multi-threaded implementation of Isolation Forest (a.k.a. iForest) and variations of it such as Extended Isolation Forest (EIF), Split-Criterion iForest (SCiForest), Fair-Cut Forest (FCF), Robust Random-Cut Forest (RRCF), and other customizable variants, aimed at outlier/anomaly detection plus additions for imputation of missing values, distance/similarity calculation between observations, and handling of categorical data. Written in C++ with interfaces for Python, R, and C. An additional wrapper for Ruby can be found [here](https://github.com/ankane/isotree).

The new concepts in this software are described in:
* [Revisiting randomized choices in isolation forests](https://arxiv.org/abs/2110.13402)
* [Isolation forests: looking beyond tree depth](https://arxiv.org/abs/2111.11639)
* [Distance approximation using Isolation Forests](https://arxiv.org/abs/1910.12362)
* [Imputing missing values with unsupervised random trees](https://arxiv.org/abs/1911.06646)

*********************

For a quick introduction to the Isolation Forest concept as used in this library, see:
* [Python introductory notebook](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/an_introduction_to_isolation_forests.ipynb).
* [R Vignette](http://htmlpreview.github.io/?https://github.com/david-cortes/isotree/blob/master/inst/doc/An_Introduction_to_Isolation_Forests.html).

Short Python example notebooks:
* [General library usage](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_example.ipynb).
* [Using it as imputer in a scikit-learn pipeline](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_impute.ipynb).
* [Using it as a kernel for SVMs](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_svm_kernel_example.ipynb).
* [Converting it to TreeLite format for faster predictions](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/treelite_example.ipynb).

(R examples are available in the internal documentation)

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
| h2o             | orig   |  9.33        | 11.21         | 14.23         |
| h2o             | ext    |  1.06        | 2.07          | 17.31         |
| scikit-learn    | orig   |  8.3         | 8.01          | 6.89          |
| solitude        | orig   |  32.612      | 34.01         | 41.01         |


Example AUC as outlier detector in typical datasets (notebook to produce results [here](https://github.com/david-cortes/isotree/blob/master/example/comparison_model_quality.ipynb)):

* Satellite dataset:

| Library      | AUROC defaults | AUROC grid search |
| :---:        | :---:          | :---:             |
| isotree      | 0.70           | 0.84              |
| eif          | -              | 0.714             |
| scikit-learn | 0.687          | 0.74              |
| h2o          | 0.662          | 0.748             |

* Annthyroid dataset:

| Library      | AUROC defaults | AUROC grid search |
| :---:        | :---:          | :---:             |
| isotree      | 0.80           | 0.982             |
| eif          | -              | 0.808             |
| scikit-learn | 0.836          | 0.836             |
| h2o          | 0.80           | 0.80              |

*(Disclaimer: these are rather small datasets and thus these AUC estimates have high variance)*

# Non-random splits

While the original idea behind isolation forests consisted in deciding splits uniformly at random, it's possible to get better performance at detecting outliers in some datasets (particularly those with multimodal distributions) by determining splits according to an information gain criterion instead. The idea is described in ["Revisiting randomized choices in isolation forests"](https://arxiv.org/abs/2110.13402) along with some comparisons of different split guiding criteria.

# Different outlier scoring criteria

Although the intuition behind the algorithm was to look at the tree depth required for isolation, this package can also produce outlier scores based on density criteria, which provide improved results in some datasets, particularly when splitting on categorical features. The idea is described in ["Isolation forests: looking beyond tree depth"](https://arxiv.org/abs/2111.11639).

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
* Can calculate isolation kernels or proximity matrix, which counts the proportion of trees in which two given observations end up in the same terminal node.
* Can produce missing value imputations according to observations that fall on each terminal node.
* Can work with sparse matrices.
* Can use either depth-based metrics or density-based metrics for calculation of outlier scores.
* Supports sample/observation weights, either as sampling importance or as distribution density measurement.
* Supports user-provided column sample weights.
* Can sample columns randomly with weights given by kurtosis.
* Uses exact formula (not approximation as others do) for harmonic numbers at lower sample and remainder sizes, and a higher-order approximation for larger sizes.
* Can fit trees incrementally to user-provided data samples.
* Produces serializable model objects with reasonable file sizes.
* Can convert the models to `treelite` format (Python-only and depending on the parameters that are used) ([example here](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/treelite_example.ipynb)).
* Can translate the generated trees into SQL statements.
* Fast and multi-threaded C++ code with an ISO C interface, which is architecture-agnostic, multi-platform, and with the only external dependency (Robin-Map) being optional. Can be wrapped in languages other than Python/R/Ruby.

(Note that categoricals, NAs, and density-like sample weights, are treated heuristically with different options as there is no single logical extension of the original idea to them, and having them present might degrade performance/accuracy for regular numerical non-missing observations)

# Installation

* R:

**Note:** This package benefits from extra optimizations that aren't enabled by default for R packages. See [this guide](https://github.com/david-cortes/installing-optimized-libraries) for instructions on how to enable them.

```r
install.packages("isotree")
```
** *


* Python:

**Note:** requires C/C++ compilers configured for Python. See [this guide](https://github.com/david-cortes/installing-optimized-libraries) for instructions.

```
pip install isotree
```
or if that fails:
```
pip install --no-use-pep517 isotree
```
** *

**Note for macOS users:** on macOS, the Python version of this package might compile **without** multi-threading capabilities. In order to enable multi-threading support, first install OpenMP:
```
brew install libomp
```
And then reinstall this package: `pip install --upgrade --no-deps --force-reinstall isotree`.

** *
**IMPORTANT:** the setup script will try to add compilation flag `-march=native`. This instructs the compiler to tune the package for the CPU in which it is being installed (by e.g. using AVX instructions if available), but the result might not be usable in other computers. If building a binary wheel of this package or putting it into a docker image which will be used in different machines, this can be overridden either by (a) defining an environment variable `DONT_SET_MARCH=1`, or by (b) manually supplying compilation `CFLAGS` as an environment variable with something related to architecture. For maximum compatibility (but slowest speed), it's possible to do something like this:

```
export DONT_SET_MARCH=1
pip install isotree
```

or, by specifying some compilation flag for architecture:
```
export CFLAGS="-march=x86-64"
export CXXFLAGS="-march=x86-64"
pip install isotree
```
** *

* C and C++:
```
git clone --recursive https://www.github.com/david-cortes/isotree.git
cd isotree
mkdir build
cd build
cmake -DUSE_MARCH_NATIVE=1 ..
cmake --build .

### for a system-wide install in linux
sudo make install
sudo ldconfig
```

(Will build as a shared object - linkage is then done with `-lisotree`)

Be aware that the snippet above includes option `-DUSE_MARCH_NATIVE=1`, which will make it use the highest-available CPU instruction set (e.g. AVX2) and will produces objects that might not run on older CPUs - to build more "portable" objects, remove this option from the cmake command.

The package has an optional dependency on the [Robin-Map](https://github.com/Tessil/robin-map) library, which is added to this repository as a linked submodule. If this library is not found under `/src`, will use the compiler's own hashmaps, which are less optimal.

* Ruby:

See [external repository with wrapper](https://github.com/ankane/isotree).

# Sample usage

**Warning: default parameters in this implementation are very different from default parameters in others such as Scikit-Learn's, and these defaults won't scale to large datasets (see documentation for details).**

* Python:

(Library is Scikit-Learn compatible)

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
iso = IsolationForest(ntrees = 10, nthreads = 1)
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

The package comes with two different C++ interfaces: (a) a struct-based interface which exposes the full library's functionalities but makes little checks on the inputs it receives and is difficult to use due to the large number of arguments that functions require; and (b) a scikit-learn-like interface in which the model exposes a single class with methods like 'fit' and 'predict', which is less flexible than the struct-based interface but easier to use and the function signatures disallow some potential errors due to invalid parameter combinations. The latter ((b)) is recommended to use unless some specific functionality from (a) is required.


See files: [isotree_cpp_oop_ex.cpp](https://github.com/david-cortes/isotree/blob/master/example/isotree_cpp_oop_ex.cpp) for an example with the scikit-learn-like interface (recommended); and [isotree_cpp_ex.cpp](https://github.com/david-cortes/isotree/blob/master/example/isotree_cpp_ex.cpp) for an example with the struct-based interface.

Note that the scikit-learn-like interface does not expose all the functionalities - for example, it only supports inputs of classes 'double' and 'int', while the struct-based interface also supports 'float'/'size_t'.

* C:

See file [isotree_c_ex.c](https://github.com/david-cortes/isotree/blob/master/example/isotree_c_ex.c).

Note that the C interface is a simple wrapper over a subset of the scikit-learn-like C++ interface, but using only ISO C bindings for better compatibility and easier wrapping in other languages.

* Ruby

See [external repository with wrapper](https://github.com/ankane/isotree).

# Examples

* Python:
    * [Example about general library usage](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_example.ipynb).
    * [Example using it as imputer in a scikit-learn pipeline](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_impute.ipynb).
    * [Example using it as a kernel for SVMs](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_svm_kernel_example.ipynb).
    * [Example converting it to TreeLite format for faster predictions](https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/treelite_example.ipynb).
* R: examples available in the documentation (`help(isotree::isolation.forest)`, [link to CRAN](https://cran.r-project.org/web/packages/isotree/index.html)).
* C and C++: see short examples in the section above.
* Ruby: see [external repository with wrapper](https://github.com/ankane/isotree).

# Documentation

* Python: documentation is available at [ReadTheDocs](http://isotree.readthedocs.io/en/latest/).
* R: documentation is available internally in the package (e.g. `help(isolation.forest)`) and in [CRAN](https://cran.r-project.org/web/packages/isotree/index.html).
* C++: documentation is available in the public header (`include/isotree.hpp`) and in the source files. See also the header for the scikit-learn-like interface (`include/isotree_oop.hpp`).
* C: interface is not documented per-se, but the same documentation from the C++ header applies to it. See also its header for some non-comprehensive comments about the parameters that functions take (`include/isotree_c.h`).
* Ruby: see [external repository with wrapper](https://github.com/ankane/isotree) for the syntax and the [Python docs](http://isotree.readthedocs.io) for details about the parameters.

# Reducing library size and compilation times

By default, this library will compile with some functionalities that are unlikely to be used and which can significantly increase the size of the library and compilation times - if using this library in e.g. embedded devices, it is highly recommended to disable some options, and if creating a docker images for serving models, one might want to make it as minimal as possible. Being a C++ templated library, it generates multiple versions of its functions that are specialized for different types (such as C `double` and `float`), and in practice not all the supported types are likely to be used.

In particular, the library supports usage of `long double` type for more precise aggregated calculations (e.g. standard deviations), which is unlikely to end up used (its usage is determined by a user-passed function argument and not available in the C or C++-OOP interfaces). For a smaller library and faster compilation, support for `long double` can be disabled by:

* Defining an environment variable `NO_LONG_DOUBLE`, which will be accepted by the Python and R build systems - e.g. first run `export NO_LONG_DOUBLE=1`, then a `pip` install; or for R, run `Sys.setenv("NO_LONG_DOUBLE" = "1")` before `install.packages`.
* Passing option `NO_LONG_DOUBLE` to the CMake script - e.g. `cmake -DNO_LONG_DOUBLE=1 ..` (only when using the CMake system, which is not used by the Python and R versions).


Additionally, the library will produce functions for different floating point and integer types of the input data. In practice, one usually ends up using only `double` and `int` types (these are the only types supported in the R interface and in the C and C++-OOP interfaces). When building it as a shared library through the CMake system, these can be disabled (leaving only `double` and `int` support) through option `NO_TEMPLATED_VERSIONS` - e.g.:
```
cmake -DNO_TEMPLATED_VERSIONS=1 ..
```
(this option is not available for the Python build system)


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
* Cortes, David. "Revisiting randomized choices in isolation forests." arXiv preprint arXiv:2110.13402 (2021).
* Guha, Sudipto, et al. "Robust random cut forest based anomaly detection on streams." International conference on machine learning. PMLR, 2016.
* Cortes, David. "Isolation forests: looking beyond tree depth." arXiv preprint arXiv:2111.11639 (2021).
* Ting, Kai Ming, Yue Zhu, and Zhi-Hua Zhou. "Isolation kernel and its effect on SVM." Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.
