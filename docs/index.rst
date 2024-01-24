Isolation-based Outlier Detection
=================================

This is the documentation page for the `isotree` Python package, which provides Isolation Forest models for outlier/anomaly detection and other purposes. See project's GitHub for more details:

`<https://www.github.com/david-cortes/isotree>`_

For the R version, see the CRAN webpage:

`<https://cran.r-project.org/web/packages/isotree/index.html>`_

Installation
============

The Python version of this package can be easily installed from PyPI
::

   pip install isotree

(See the GitHub page for more details, esp. section "Reducing library size and compilation times")

Note that it is only available in source form (not in binary wheel form), which means you will need a toolchain for compiling C++ source code (e.g. GCC in linux, msys2 that comes with anaconda on windows, clang in mac).

Introduction to the library and methods
=======================================

* `An introduction to Isolation Forests <https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/an_introduction_to_isolation_forests.ipynb>`_.


Quick example notebooks
=======================

* `An introduction to Isolation Forests <https://nbviewer.jupyter.org/github/david-cortes/isotree/blob/master/example/an_introduction_to_isolation_forests.ipynb>`_.
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
* `get_num_nodes <#isotree.IsolationForest.get_num_nodes>`_
* `get_params <#isotree.IsolationForest.get_params>`_
* `import_model <#isotree.IsolationForest.import_model>`_
* `partial_fit <#isotree.IsolationForest.partial_fit>`_
* `plot_tree <#isotree.IsolationForest.plot_tree>`_
* `predict <#isotree.IsolationForest.predict>`_
* `predict_distance <#isotree.IsolationForest.predict_distance>`_
* `predict_kernel <#isotree.IsolationForest.predict_kernel>`_
* `set_params <#isotree.IsolationForest.set_params>`_
* `set_reference_points <#isotree.IsolationForest.set_reference_points>`_
* `subset_trees <#isotree.IsolationForest.subset_trees>`_
* `to_json <#isotree.IsolationForest.to_json>`_
* `to_graphviz <#isotree.IsolationForest.to_graphviz>`_
* `to_sql <#isotree.IsolationForest.to_sql>`_
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
