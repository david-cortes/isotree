/*    Isolation forests and variations thereof, with adjustments for incorporation
*     of categorical variables and missing values.
*     Writen for C++11 standard and aimed at being used in R and Python.
*     
*     This library is based on the following works:
*     [1] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*         "Isolation forest."
*         2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
*     [2] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*         "Isolation-based anomaly detection."
*         ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
*     [3] Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner.
*         "Extended Isolation Forest."
*         arXiv preprint arXiv:1811.02141 (2018).
*     [4] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*         "On detecting clustered anomalies using SCiForest."
*         Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2010.
*     [5] https://sourceforge.net/projects/iforest/
*     [6] https://math.stackexchange.com/questions/3388518/expected-number-of-paths-required-to-separate-elements-in-a-binary-tree
*     [7] Quinlan, J. Ross. C4. 5: programs for machine learning. Elsevier, 2014.
*     [8] Cortes, David.
*         "Distance approximation using Isolation Forests."
*         arXiv preprint arXiv:1910.12362 (2019).
*     [9] Cortes, David.
*         "Imputing missing values with unsupervised random trees."
*         arXiv preprint arXiv:1911.06646 (2019).
*     [10] https://math.stackexchange.com/questions/3333220/expected-average-depth-in-random-binary-tree-constructed-top-to-bottom
*     [11] Cortes, David.
*          "Revisiting randomized choices in isolation forests."
*          arXiv preprint arXiv:2110.13402 (2021).
*     [12] Guha, Sudipto, et al.
*          "Robust random cut forest based anomaly detection on streams."
*          International conference on machine learning. PMLR, 2016.
*     [13] Cortes, David.
*          "Isolation forests: looking beyond tree depth."
*          arXiv preprint arXiv:2111.11639 (2021).
*     [14] Ting, Kai Ming, Yue Zhu, and Zhi-Hua Zhou.
*          "Isolation kernel and its effect on SVM"
*          Proceedings of the 24th ACM SIGKDD
*          International Conference on Knowledge Discovery & Data Mining. 2018.
* 
*     BSD 2-Clause License
*     Copyright (c) 2019-2021, David Cortes
*     All rights reserved.
*     Redistribution and use in source and binary forms, with or without
*     modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright notice, this
*       list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright notice,
*       this list of conditions and the following disclaimer in the documentation
*       and/or other materials provided with the distribution.
*     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
*     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
*     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
*     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
*     DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
*     SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*     OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
*     OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


/***********************************************************************************
    ---------------------
    IsoTree OOP interface
    ---------------------

    This is provided as an alternative easier-to-use interface for this library
    which follows scikit-learn-style methods with a single C++ class. It is a
    wrapper over the non-OOP header 'isotree.hpp', providing the same functionality
    in a perhaps more comprehensible structure, while still offering direct access
    to the underlying objects so as to allow using the functions from 'isotree.hpp'.
    
    It is a more limited interface as it does not implement all the functionality
    for serialization, distance prediction, oproducing predictions in the same call
    as the model is fit, or fitting/predicting on data with types other than
    'double' and 'int'.

    The descriptions here do not contain the full documentation, but rather only
    some hints so as to make them more comprehensible, aiming at producing function
    signatures that are self-descriptive instead (if you are familiar with the
    scikit-learn library for Python).
    
    For detailed documentation see the same or similar-looking methods in the
    'isotree.hpp' header instead.

***********************************************************************************/
#ifndef ISOTREE_OOP_H
#define ISOTREE_OOP_H

#include "isotree.hpp"

namespace isotree {

class ISOTREE_EXPORTED IsolationForest
{
public:
    /*  Note: if passing nthreads<0, will reset it to 'max_threads + nthreads + 1',
        so passing -1 means using all available threads.  */
    int nthreads = -1; /* <- May be manually changed at any time */

    uint64_t random_seed = 1;

    /*  General tree construction parameters  */
    size_t ndim = 3;
    size_t ntry = 1;
    CoefType coef_type = Uniform; /* only for ndim>1 */
    bool   with_replacement = false;
    bool   weight_as_sample = true;
    size_t sample_size = 0;
    size_t ntrees = 500;
    size_t max_depth = 0;
    size_t ncols_per_tree = 0;
    bool   limit_depth = true; /* if 'true', then 'max_depth' is ignored */
    bool   penalize_range = false;
    bool   standardize_data = true; /* only for ndim==1 */
    ScoringMetric scoring_metric = Depth;
    bool   fast_bratio = true; /* only for scoring_metric with 'Boxed' */
    bool   weigh_by_kurt = false;
    double prob_pick_by_gain_pl = 0.;
    double prob_pick_by_gain_avg = 0.;
    double prob_pick_by_full_gain = 0.;
    double prob_pick_by_dens = 0.;
    double prob_pick_col_by_range = 0.;
    double prob_pick_col_by_var = 0.;
    double prob_pick_col_by_kurt = 0.;
    double min_gain = 0.;
    MissingAction missing_action = Impute;

    /*  For categorical variables  */
    CategSplit cat_split_type = SubSet;
    NewCategAction new_cat_action = Weighted;
    bool   coef_by_prop = false;
    bool   all_perm = false;

    /*  For imputation methods (when using 'build_imputer=true' and calling 'impute')  */
    bool   build_imputer = false;
    size_t min_imp_obs = 3;
    UseDepthImp depth_imp = Higher;
    WeighImpRows weigh_imp_rows = Inverse;

    /*  Internal objects which can be used with the non-OOP interface  */
    IsoForest model;
    ExtIsoForest model_ext;
    Imputer imputer;
    TreesIndexer indexer;

    IsolationForest() = default;

    ~IsolationForest() = default;

    /*  Be aware that many combinations of parameters are invalid.
        This function will not do any validation of the inputs it receives.

        Calling 'fit' with a combination of invalid parameters *may* throw a
        runtime exception, but it will not be able to detect all the possible
        invalid parameter combinations and could potentially lead to silent
        errors like statistically incorrect models or predictions that do not
        make sense. See the documentation of the non-OOP header or of the R
        and Python interfaces for more details about the parameters and the
        valid and invalid combinations of parameters.  */
    IsolationForest
    (
        size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
        bool with_replacement, bool weight_as_sample,
        size_t sample_size, size_t ntrees,
        size_t max_depth, size_t ncols_per_tree, bool   limit_depth,
        bool penalize_range, bool standardize_data,
        ScoringMetric scoring_metric, bool fast_bratio, bool weigh_by_kurt,
        double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
        double prob_pick_col_by_range, double prob_pick_col_by_var,
        double prob_pick_col_by_kurt,
        double min_gain, MissingAction missing_action,
        CategSplit cat_split_type, NewCategAction new_cat_action,
        bool   all_perm, bool build_imputer, size_t min_imp_obs,
        UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
        uint64_t random_seed, int nthreads
    );

    /*  'X' must be in column-major order (like Fortran).  */
    void fit(double X[], size_t nrows, size_t ncols);

    /*  Model can also be fit to categorical data (must also be column-major).
        Categorical data should be passed as integers starting at zero, with
        negative values denoting missing, and must pass also the number of
        categories to expect in each column.

        Can also pass row and column weights (see the documentation for options
        on how to interpret the row weights).  */
    void fit(double numeric_data[],   size_t ncols_numeric,  size_t nrows,
             int    categ_data[],     size_t ncols_categ,    int ncat[],
             double sample_weights[], double col_weights[]);

    /*  Numeric data may also be supplied as a sparse matrix, in which case it
        must be CSC format (colum-major). Categorical data is not supported in
        sparse format.  */
    void fit(double Xc[], int Xc_ind[], int Xc_indptr[],
             size_t ncols_numeric,      size_t nrows,
             int    categ_data[],       size_t ncols_categ,   int ncat[],
             double sample_weights[],   double col_weights[]);

    /*  'predict' will return a vector with the standardized outlier scores
        (output length is the same as the number of rows in the data), in
        which higher values mean more outlierness.

        The data must again be in column-major format.

        This function will run multi-threaded if there is more than one row and
        the object has number of threads set to more than 1.  */
    std::vector<double> predict(double X[], size_t nrows, bool standardize);

    /*  Can optionally write to a non-owned array, or obtain the non-standardized
        isolation depth instead of the standardized score (also on a per-tree basis
        if desired), or get the terminal node numbers/indices for each tree. Note
        that 'tree_num' and 'per_tree_depths' are optional (pass NULL if not desired),
        while 'output_depths' should always be passed. Be aware that the outputs of
        'tree_num' will be filled in column-major order ([nrows, ntrees]), while the
        outputs of 'per_tree_depths' will be in row-major order.

        Note: 'tree_num' and 'per_tree_depths' will not be calculable when using
        'ndim==1' plus either 'missing_action==Divide' or 'new_cat_action==Weighted'.
        These can be checked through 'check_can_predict_per_tree'.
       
        Here, the data might be passed as either column-major or row-major (getting
        predictions in row-major order will be faster). If the data is in row-major
        order, must also provide the leading dimension of the array (typically this
        corresponds to the number of columns, but might be larger if using a subset
        of a larger array).  */
    void predict(double numeric_data[], int categ_data[], bool is_col_major,
                 size_t nrows, size_t ld_numeric, size_t ld_categ, bool standardize,
                 double output_depths[], int tree_num[], double per_tree_depths[]);

    /*  Numeric data may also be provided in sparse format, which can be either
        CSC (column-major) or CSR (row-major). If the number of rows is large,
        predictions in CSC format will be faster than in CSR (assuming that
        categorical data is either missing or column-major). Note that for CSC,
        parallelization is done by trees instead of by rows, and outputs are
        subject to numerical rounding error between runs.  */
    void predict(double X_sparse[], int X_ind[], int X_indptr[], bool is_csc,
                 int categ_data[], bool is_col_major, size_t ld_categ, size_t nrows, bool standardize,
                 double output_depths[], int tree_num[], double per_tree_depths[]);

    /*  Distances between observations will be returned either as a triangular matrix
        representing an upper diagonal (length is nrows*(nrows-1)/2), or as a full
        square matrix (length is nrows^2).  */
    std::vector<double> predict_distance(double X[], size_t nrows,
                                         bool as_kernel,
                                         bool assume_full_distr, bool standardize,
                                         bool triangular);

    void predict_distance(double numeric_data[], int categ_data[],
                          size_t nrows,
                          bool as_kernel,
                          bool assume_full_distr, bool standardize,
                          bool triangular,
                          double dist_matrix[]);

    /*  Sparse data is only supported in CSC format.  */
    void predict_distance(double Xc[], int Xc_ind[], int Xc_indptr[], int categ_data[],
                          size_t nrows,
                          bool as_kernel,
                          bool assume_full_distr, bool standardize,
                          bool triangular,
                          double dist_matrix[]);

    /*  This will impute missing values in-place. Data here must be in column-major order.   */
    void impute(double X[], size_t nrows);

    /*  This variation will accept data in either row-major or column-major order.
        The leading dimension must match with the number of columns for row major,
        or with the number of rows for column-major (custom leading dimensions are
        not supported).  */
    void impute(double numeric_data[], int categ_data[], bool is_col_major, size_t nrows);

    /*  Numeric data may be passed in sparse CSR format. Note however that it will
        impute the values that are NAN, not the values that are ommited from the
        sparse format.  */
    void impute(double Xr[], int Xr_ind[], int Xr_indptr[],
                int categ_data[], bool is_col_major, size_t nrows);

    void build_indexer(const bool with_distances);

    /*  Sets points as reference to later calculate distances or kernel from arbitrary points
        to these ones, without having to save these reference points's original features.  */
    void set_as_reference_points(double numeric_data[], int categ_data[], bool is_col_major,
                                 size_t nrows, size_t ld_numeric, size_t ld_categ,
                                 const bool with_distances);

    void set_as_reference_points(double Xc[], int Xc_ind[], int Xc_indptr[], int categ_data[],
                                 size_t nrows, const bool with_distances);

    size_t get_num_reference_points() const noexcept;

    /*  Must call 'set_as_reference_points' to make this method available.

        Here 'dist_matrix' should have dimension [nrows, n_references],
        and will be filled in row-major order.

        This will always take 'assume_full_distr=true'.  */
    void predict_distance_to_ref_points(double numeric_data[], int categ_data[],
                                        double Xc[], int Xc_ind[], int Xc_indptr[],
                                        size_t nrows, bool is_col_major, size_t ld_numeric, size_t ld_categ,
                                        bool as_kernel, bool standardize,
                                        double dist_matrix[]);

    /*  Serialize (save) the model to a file. See 'isotree.hpp' for compatibility
        details. Note that this does not save all the details of the object, but
        rather only those that are necessary for prediction.

        The file must be opened in binary write mode ('wb').

        Note that models serialized through this interface are not importable in
        the R and Python wrappers around this library.  */
    void serialize(FILE *out) const;

    /*  The stream must be opened in binary mode.  */
    void serialize(std::ostream &out) const;

    /*  The number of threads here does not mean 'how many threads to use while
        deserializing', but rather, 'how many threads will be set for the prediction
        functions of the resulting object'.

        The input file must be opened in binary read more ('rb').

        Note that not all the members of an 'IsolationForest' object are saved
        when serializing, so if you access members such as 'prob_pick_by_gain_avg',
        they will all be at their default values.

        These functions can de-serialize models saved from the R and Python interfaces,
        but models that are serialized from this C++ interface are not importable in
        those R and Python versions.  */
    static IsolationForest deserialize(FILE *inp, int nthreads);

    /*  The stream must be opened in binary mode.  */
    static IsolationForest deserialize(std::istream &inp, int nthreads);

    /*  To serialize and deserialize in a more idiomatic way
        ('stream << model' and 'stream >> model').
        Note that 'ist >> model' will set 'nthreads=-1', which you might
        want to modify afterwards. */
    friend std::ostream& operator<<(std::ostream &ost, const IsolationForest &model);

    friend std::istream& operator>>(std::istream &ist, IsolationForest &model);

    /*  These functions allow getting the underlying objects to use with the more
        featureful non-OOP interface.

        Note that it is also possible to use the C-interface functions with this
        object by passing a pointer to the 'IsolationForest' object instead.  */
    IsoForest& get_model();

    ExtIsoForest& get_model_ext();

    Imputer& get_imputer();

    TreesIndexer& get_indexer();

    /*  This converts from a negative 'nthreads' to the actual number (provided it
        was compiled with OpenMP support), and will set to 1 if the number is invalid.
        If the library was compiled without multi-threading and it requests more than
        one thread, will write a message to 'stderr'.  */
    void check_nthreads();

    /*  This will return the number of trees in the object. If it is not fitted, will
        throw an error instead.  */
    size_t get_ntrees() const;

    /*  This checks whether 'predict' can output 'tree_num' and 'per_tree_depths'.  */
    bool check_can_predict_per_tree() const;

private:
    bool is_fitted = false;

    void override_previous_fit();
    void check_params();
    void check_is_fitted() const;
    IsolationForest(int nthreads, size_t ndim, size_t ntrees, bool build_imputer);
    template <class otype>
    void serialize_template(otype &out) const;
    template <class itype>
    static IsolationForest deserialize_template(itype &inp, int nthreads);

};

ISOTREE_EXPORTED
std::ostream& operator<<(std::ostream &ost, const IsolationForest &model);
ISOTREE_EXPORTED
std::istream& operator>>(std::istream &ist, IsolationForest &model);

}

#endif /* ifndef ISOTREE_OOP_H */
