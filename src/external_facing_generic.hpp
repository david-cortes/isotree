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
*     [8] Cortes, David. "Distance approximation using Isolation Forests." arXiv preprint arXiv:1910.12362 (2019).
*     [9] Cortes, David. "Imputing missing values with unsupervised random trees." arXiv preprint arXiv:1911.06646 (2019).
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
ISOTREE_EXPORTED
int fit_iforest(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                real_t numeric_data[],  size_t ncols_numeric,
                int    categ_data[],    size_t ncols_categ,    int ncat[],
                real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
                real_t sample_weights[], bool with_replacement, bool weight_as_sample,
                size_t nrows, size_t sample_size, size_t ntrees, size_t max_depth,
                bool   limit_depth, bool penalize_range,
                bool   standardize_dist, double tmat[],
                real_t output_depths[], bool standardize_depth,
                double col_weights[], bool weigh_by_kurt,
                double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
                double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
                double min_gain, MissingAction missing_action,
                CategSplit cat_split_type, NewCategAction new_cat_action,
                bool   all_perm, Imputer *imputer, size_t min_imp_obs,
                UseDepthImp depth_imp, WeighImpRows weigh_imp_rows, bool impute_at_fit,
                uint64_t random_seed, int nthreads);
ISOTREE_EXPORTED
int add_tree(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
             real_t numeric_data[],  size_t ncols_numeric,
             int    categ_data[],    size_t ncols_categ,    int ncat[],
             real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
             size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
             real_t sample_weights[], size_t nrows, size_t max_depth,
             bool   limit_depth,   bool penalize_range,
             real_t col_weights[], bool weigh_by_kurt,
             double prob_pick_by_gain_avg, double prob_split_by_gain_avg,
             double prob_pick_by_gain_pl,  double prob_split_by_gain_pl,
             double min_gain, MissingAction missing_action,
             CategSplit cat_split_type, NewCategAction new_cat_action,
             UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
             bool   all_perm, std::vector<ImputeNode> *impute_nodes, size_t min_imp_obs,
             uint64_t random_seed);
ISOTREE_EXPORTED
void predict_iforest(real_t numeric_data[], int categ_data[],
                     bool is_col_major, size_t ncols_numeric, size_t ncols_categ,
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                     size_t nrows, int nthreads, bool standardize,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double output_depths[],   sparse_ix tree_num[]);
ISOTREE_EXPORTED void get_num_nodes(IsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads);
ISOTREE_EXPORTED void get_num_nodes(ExtIsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads);
ISOTREE_EXPORTED
void calc_similarity(real_t numeric_data[], int categ_data[],
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     size_t nrows, int nthreads, bool assume_full_distr, bool standardize_dist,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double tmat[], double rmat[], size_t n_from);
ISOTREE_EXPORTED
void impute_missing_values(real_t numeric_data[], int categ_data[], bool is_col_major,
                           real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                           size_t nrows, int nthreads,
                           IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                           Imputer &imputer);
ISOTREE_EXPORTED
void merge_models(IsoForest*     model,      IsoForest*     other,
                  ExtIsoForest*  ext_model,  ExtIsoForest*  ext_other,
                  Imputer*       imputer,    Imputer*       iother);
ISOTREE_EXPORTED
std::string generate_sql_with_select_from(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                          std::string &table_from, std::string &select_as,
                                          std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                                          std::vector<std::vector<std::string>> &categ_levels,
                                          bool index1, int nthreads);
ISOTREE_EXPORTED
std::vector<std::string> generate_sql(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                      std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                                      std::vector<std::vector<std::string>> &categ_levels,
                                      bool output_tree_num, bool index1, bool single_tree, size_t tree_num,
                                      int nthreads);

ISOTREE_EXPORTED
size_t determine_serialized_size(const IsoForest &model);
ISOTREE_EXPORTED
size_t determine_serialized_size(const ExtIsoForest &model);
ISOTREE_EXPORTED
size_t determine_serialized_size(const Imputer &model);
ISOTREE_EXPORTED
void serialize_IsoForest(const IsoForest &model, char *out);
ISOTREE_EXPORTED
void serialize_IsoForest(const IsoForest &model, FILE *out);
ISOTREE_EXPORTED
void serialize_IsoForest(const IsoForest &model, std::ostream &out);
ISOTREE_EXPORTED
std::string serialize_IsoForest(const IsoForest &model);
ISOTREE_EXPORTED
void deserialize_IsoForest(IsoForest &model, const char *in);
ISOTREE_EXPORTED
void deserialize_IsoForest(IsoForest &model, FILE *in);
ISOTREE_EXPORTED
void deserialize_IsoForest(IsoForest &model, std::istream &in);
ISOTREE_EXPORTED
void deserialize_IsoForest(IsoForest &model, const std::string &in);
ISOTREE_EXPORTED
void serialize_ExtIsoForest(const ExtIsoForest &model, char *out);
ISOTREE_EXPORTED
void serialize_ExtIsoForest(const ExtIsoForest &model, FILE *out);
ISOTREE_EXPORTED
void serialize_ExtIsoForest(const ExtIsoForest &model, std::ostream &out);
ISOTREE_EXPORTED
std::string serialize_ExtIsoForest(const ExtIsoForest &model);
ISOTREE_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, const char *in);
ISOTREE_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, FILE *in);
ISOTREE_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, std::istream &in);
ISOTREE_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, const std::string &in);
ISOTREE_EXPORTED
void serialize_Imputer(const Imputer &model, char *out);
ISOTREE_EXPORTED
void serialize_Imputer(const Imputer &model, FILE *out);
ISOTREE_EXPORTED
void serialize_Imputer(const Imputer &model, std::ostream &out);
ISOTREE_EXPORTED
std::string serialize_Imputer(const Imputer &model);
ISOTREE_EXPORTED
void deserialize_Imputer(Imputer &model, const char *in);
ISOTREE_EXPORTED
void deserialize_Imputer(Imputer &model, FILE *in);
ISOTREE_EXPORTED
void deserialize_Imputer(Imputer &model, std::istream &in);
ISOTREE_EXPORTED
void deserialize_Imputer(Imputer &model, const std::string &in);
ISOTREE_EXPORTED
size_t determine_serialized_size_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const size_t size_optional_metadata
);
ISOTREE_EXPORTED
size_t determine_serialized_size_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const size_t size_optional_metadata
);
ISOTREE_EXPORTED
void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    char *out
);
ISOTREE_EXPORTED
void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    FILE *out
);
ISOTREE_EXPORTED
void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    std::ostream &out
);
ISOTREE_EXPORTED
std::string serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata
);
ISOTREE_EXPORTED
void serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    FILE *out
);
ISOTREE_EXPORTED
void serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    std::ostream &out
);
ISOTREE_EXPORTED
std::string serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata
);
ISOTREE_EXPORTED
void deserialize_combined
(
    const char* in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
);
ISOTREE_EXPORTED
void deserialize_combined
(
    FILE* in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
);
ISOTREE_EXPORTED
void deserialize_combined
(
    std::istream &in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
);
ISOTREE_EXPORTED
void deserialize_combined
(
    const std::string &in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
);
ISOTREE_EXPORTED
bool check_can_undergo_incremental_serialization(const IsoForest &model, const char *serialized_bytes);
ISOTREE_EXPORTED
bool check_can_undergo_incremental_serialization(const ExtIsoForest &model, const char *serialized_bytes);
ISOTREE_EXPORTED
bool check_can_undergo_incremental_serialization(const Imputer &model, const char *serialized_bytes);
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const IsoForest &model, size_t old_ntrees);
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const ExtIsoForest &model, size_t old_ntrees);
ISOTREE_EXPORTED
size_t determine_serialized_size_additional_trees(const Imputer &model, size_t old_ntrees);
ISOTREE_EXPORTED
void incremental_serialize_IsoForest(const IsoForest &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_Imputer(const Imputer &model, char *old_bytes_reallocated);
ISOTREE_EXPORTED
void incremental_serialize_IsoForest(const IsoForest &model, std::string &old_bytes);
ISOTREE_EXPORTED
void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, std::string &old_bytes);
ISOTREE_EXPORTED
void incremental_serialize_Imputer(const Imputer &model, std::string &old_bytes);

