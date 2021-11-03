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

#if !defined(_FOR_R) && !defined(_FOR_PYTHON)

#include "isotree.hpp"

namespace isotree {

ISOTREE_EXPORTED
class IsolationForest
{
public:
    int nthreads = -1;

    uint64_t random_seed = 1;

    size_t ndim = 3;
    size_t ntry = 1;
    CoefType coef_type = Normal;
    bool   with_replacement = false;
    bool   weight_as_sample = true;
    size_t sample_size = 0;
    size_t ntrees = 500;
    size_t max_depth = 0;
    size_t ncols_per_tree = 0;
    bool   limit_depth = true;
    bool   penalize_range = false;
    bool   standardize_data = true;
    bool   weigh_by_kurt = false;
    double prob_pick_by_gain_pl = 0.;
    double prob_pick_by_gain_avg = 0.;
    double prob_pick_col_by_range = 0.;
    double prob_pick_col_by_var = 0.;
    double prob_pick_col_by_kurt = 0.;
    double min_gain = 0.;
    MissingAction missing_action = Impute;

    CategSplit cat_split_type = SubSet;
    NewCategAction new_cat_action = Weighted;
    bool   coef_by_prop = false;
    bool   all_perm = false;

    bool   build_imputer = false;
    size_t min_imp_obs = 3;
    UseDepthImp depth_imp = Higher;
    WeighImpRows weigh_imp_rows = Inverse;

    IsoForest model;
    ExtIsoForest model_ext;
    Imputer imputer;

    IsolationForest() = default;

    ~IsolationForest() = default;

    IsolationForest
    (
        size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
        bool with_replacement, bool weight_as_sample,
        size_t sample_size, size_t ntrees,
        size_t max_depth, size_t ncols_per_tree, bool   limit_depth,
        bool penalize_range, bool standardize_data, bool weigh_by_kurt,
        double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
        double prob_pick_col_by_range, double prob_pick_col_by_var,
        double prob_pick_col_by_kurt,
        double min_gain, MissingAction missing_action,
        CategSplit cat_split_type, NewCategAction new_cat_action,
        bool   all_perm, bool build_imputer, size_t min_imp_obs,
        UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
        uint64_t random_seed, int nthreads
    );

    void fit(double X[], size_t nrows, size_t ncols);

    void fit(double numeric_data[],   size_t ncols_numeric,  size_t nrows,
             int    categ_data[],     size_t ncols_categ,    int ncat[],
             double sample_weights[], double col_weights[]);

    void fit(double Xc[], int Xc_ind[], int Xc_indptr[],
             size_t ncols_numeric,      size_t nrows,
             int    categ_data[],       size_t ncols_categ,   int ncat[],
             double sample_weights[],   double col_weights[]);

    std::vector<double> predict(double X[], size_t nrows, bool standardize);

    void predict(double numeric_data[], int categ_data[], bool is_col_major,
                 size_t nrows, size_t ld_numeric, size_t ld_categ, bool standardize,
                 double output_depths[], int tree_num[], double per_tree_depths[]);

    void predict(double X_sparse[], int X_ind[], int X_indptr[], bool is_csc,
                 int categ_data[], bool is_col_major, size_t ld_categ, size_t nrows, bool standardize,
                 double output_depths[], int tree_num[], double per_tree_depths[]);

    std::vector<double> predict_distance(double X[], size_t nrows,
                                         bool assume_full_distr, bool standardize_dist,
                                         bool triangular);

    void predict_distance(double numeric_data[], int categ_data[],
                          size_t nrows,
                          bool assume_full_distr, bool standardize_dist,
                          bool triangular,
                          double dist_matrix[]);

    void predict_distance(double Xc[], int Xc_ind[], int Xc_indptr[], int categ_data[],
                          size_t nrows, bool assume_full_distr, bool standardize_dist,
                          bool triangular,
                          double dist_matrix[]);

    void impute(double X[], size_t nrows);

    void impute(double numeric_data[], int categ_data[], bool is_col_major, size_t nrows);

    void impute(double Xr[], int Xr_ind[], int Xr_indptr[],
                int categ_data[], bool is_col_major, size_t nrows);

    void serialize(FILE *out) const;

    void serialize(std::ostream &out) const;

    static IsolationForest deserialize(FILE *inp, int nthreads);

    static IsolationForest deserialize(std::istream &inp, int nthreads);

    friend std::ostream& operator<<(std::ostream &ost, const IsolationForest &model);

    friend std::istream& operator>>(std::istream &ist, IsolationForest &model);

    IsoForest& get_model();

    ExtIsoForest& get_model_ext();

    Imputer& get_imputer();

    void check_nthreads();

    size_t get_ntrees() const;

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

std::ostream& operator<<(std::ostream &ost, const IsolationForest &model);
std::istream& operator>>(std::istream &ist, IsolationForest &model);

}
#endif

