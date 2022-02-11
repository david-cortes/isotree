.. isotree documentation master file, created by
   sphinx-quickstart on Wed Oct 23 21:57:56 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Isolation-based Outlier Detection
=================================

This is the documentation page for the `isotree` Python package. See project webpage for more details:

`<https://www.github.com/david-cortes/isotree>`_

For the R version, see the CRAN webpage:

`<https://cran.r-project.org/web/packages/isotree/index.html>`_

Installation
============

The Python version of this package can be easily installed from PyPI:
``
pip install isotree
``

(See the GitHub page for more details, esp. section "Reducing library size and compilation times")

Note that it is only available in source form (not in binary wheel form), which means you will need a toolchain for compiling C++ source code (e.g. GCC in linux, msys2 that comes with anaconda on windows, clang in mac).


Quick example notebooks
=======================

* `General library usage <https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_example.ipynb>`_.
* `As missing value imputer <https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_impute.ipynb>`_.
* `As kernel for SVMs <https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/isotree_svm_kernel_example.ipynb>`_.
* `Converting to treelite for faster predictions <https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/treelite_example.ipynb>`_.


Methods
=======

* `IsolationForest <#isotree.IsolationForest>`_
* `append_trees <#isotree.IsolationForest.append_trees>`_
* `build_indexer <#isotree.IsolationForest.build_indexer>`_
* `copy <#isotree.IsolationForest.copy>`_
* `decision_function <#isotree.IsolationForest.decision_function>`_
* `drop_imputer <#isotree.IsolationForest.drop_imputer>`_
* `drop_indexer <#isotree.IsolationForest.drop_indexer>`_
* `drop_reference_points <#isotree.IsolationForest.drop_reference_points>`_
* `export_model <#isotree.IsolationForest.export_model>`_
* `fit <#isotree.IsolationForest.fit>`_
* `fit_predict <#isotree.IsolationForest.fit_predict>`_
* `fit_transform <#isotree.IsolationForest.fit_transform>`_
* `generate_sql <#isotree.IsolationForest.generate_sql>`_
* `get_num_nodes <#isotree.IsolationForest.get_num_nodes>`_
* `get_params <#isotree.IsolationForest.get_params>`_
* `import_model <#isotree.IsolationForest.import_model>`_
* `partial_fit <#isotree.IsolationForest.partial_fit>`_
* `predict <#isotree.IsolationForest.predict>`_
* `predict_distance <#isotree.IsolationForest.predict_distance>`_
* `predict_kernel <#isotree.IsolationForest.predict_kernel>`_
* `set_params <#isotree.IsolationForest.set_params>`_
* `set_reference_points <#isotree.IsolationForest.set_reference_points>`_
* `subset_trees <#isotree.IsolationForest.subset_trees>`_
* `to_treelite <#isotree.IsolationForest.to_treelite>`_
* `transform <#isotree.IsolationForest.transform>`_


IsolationForest
===============

.. automodule:: isotree
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
