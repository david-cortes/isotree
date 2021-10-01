# Comparison against other libraries

Speed comparison, fitting 100 trees of varying sample sizes (256, 1024, 10k) to datasets of varying sizes. The timings are taken on a CPU AMD Ryzen 7 2700 running at 3.2GHz, with 16 threads. Note that not all libraries support multi-threading or sparse inputs.

Note about h2o: the library was configured to use the maximum number of threads, but the extended model might in practice have used only 2 of them.

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
| h2o             | orig   |   16    | Py    |  0.585       | 0.95          | 1.4           |
| h2o             | ext    |   16    | Py    |  0.456       | 1.25          | 6.2           |
| isotree         | orig   |   16    | R     |  0.00564     | 0.00999       | 0.0278        |
| isotree         | ext    |   16    | R     |  0.00789     | 0.0152        | 0.0511        |
| h2o             | orig   |   16    | R     |  1.09        | 1.08          | 2.09          |
| h2o             | ext    |   16    | R     |  1.06        | 2.07          | 7.25          |
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
| h2o             | orig   |   16    | Py    |  7.32        | 8.79          | 11.8          |
| h2o             | ext    |   16    | Py    |  0.651       | 2.14          | 18.1          |
| isotree         | orig   |   16    | R     |  0.0454      | 0.5317        | 0.148         |
| isotree         | ext    |   16    | R     |  0.05        | 0.058         | 0.234         |
| h2o             | orig   |   16    | R     |  9.33        | 11.21         | 14.23         |
| h2o             | ext    |   16    | R     |  1.06        | 2.07          | 17.31         |
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

*Disclaimer: these datasets have mostly discrete values. Some libraries such as SciKit-Learn might perform much faster when columns have continuous values*