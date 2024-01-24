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
*     Copyright (c) 2019-2024, David Cortes
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
#include "isotree.hpp"

/* Generate a GraphViz 'dot' representation of model trees, as a 'digraph' structure
* 
* Parameters
* ==========
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if using an extended model. Can only pass one of 'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if using a single-variable model. Can only pass one of 'model_outputs' and 'model_outputs_ext'.
* - numeric_colnames
*       Names to use for the numerical columns.
* - categ_colnames
*       Names to use for the categorical columns.
* - categ_levels
*       Names to use for the levels/categories of each categorical column.
* - output_tree_num
*       Whether to output the terminal node number instead of the isolation depth at each node.
* - index1
*       Whether to make the node numbers start their numeration at 1 instead of 0 in the
*       resulting strings. Ignored when passing 'output_tree_num=false'.
* - single_tree
*       Whether to generate the graph representation for a single tree of the model instead of for
*       all. The tree number to generate is to be passed under 'tree_num'.
* - tree_num
*       Tree number for which to generate a graph, if passing 'single_tree=true'.
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used. Ignored when not building with
*       OpenMP support.
* 
* Returns
* =======
* A vector containing at each element the tree nodes as a 'dot' GraphViz text representation
* for the corresponding tree in the model.
* If passing 'single_tree=true', will contain only one element, corresponding to the tree given
* in 'tree_num'.
*/
std::vector<std::string> generate_dot(const IsoForest *model_outputs,
                                      const ExtIsoForest *model_outputs_ext,
                                      const TreesIndexer *indexer,
                                      const std::vector<std::string> &numeric_colnames,
                                      const std::vector<std::string> &categ_colnames,
                                      const std::vector<std::vector<std::string>> &categ_levels,
                                      bool output_tree_num, bool index1, bool single_tree, size_t tree_num,
                                      int nthreads)
{
    if (!model_outputs && !model_outputs_ext) throw std::runtime_error("'generate_dot' got a NULL pointer for model.");
    if (model_outputs && model_outputs_ext) throw std::runtime_error("'generate_dot' got two models as inputs.");

    std::vector<std::string> numeric_colnames_escaped;
    std::vector<std::string> categ_colnames_escaped;
    std::vector<std::vector<std::string>> categ_levels_escaped;
    escape_strings(
        numeric_colnames,
        categ_colnames,
        categ_levels,
        numeric_colnames_escaped,
        categ_colnames_escaped,
        categ_levels_escaped
    );

    size_t ntrees = model_outputs? model_outputs->trees.size() : model_outputs_ext->hplanes.size();

    std::vector<std::string> out;

    if (single_tree)
    {
        if (index1) tree_num--;
        out.push_back(
            generate_dot_single_tree(
                model_outputs,
                model_outputs_ext,
                indexer,
                numeric_colnames_escaped,
                categ_colnames_escaped,
                categ_levels_escaped,
                output_tree_num, index1, tree_num
            )
        );
        return out;
    }

    out.resize(ntrees);

    /* Global variable that determines if the procedure receives a stop signal */
    SignalSwitcher ss = SignalSwitcher();

    /* For exception handling */
    bool threw_exception = false;
    std::exception_ptr ex = NULL;

    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(out, index1, tree_num, threw_exception, ex)
    for (size_t_for ix = 0; ix < (size_t_for)ntrees; ix++)
    {
        if (interrupt_switch || threw_exception)
            continue; /* Cannot break with OpenMP==2.0 (MSVC) */

        try
        {
            out[ix] = generate_dot_single_tree(
                model_outputs,
                model_outputs_ext,
                indexer,
                numeric_colnames_escaped,
                categ_colnames_escaped,
                categ_levels_escaped,
                output_tree_num, index1, ix
            );
        }

        catch (...)
        {
            #pragma omp critical
            {
                if (!threw_exception)
                {
                    threw_exception = true;
                    ex = std::current_exception();
                }
            }
        }
    }

    /* check if the procedure got interrupted */
    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return EXIT_FAILURE;
    #endif

    /* check if some exception was thrown */
    if (threw_exception)
        std::rethrow_exception(ex);

    return out;
}

std::string generate_dot_single_tree(const IsoForest *model_outputs,
                                     const ExtIsoForest *model_outputs_ext,
                                     const TreesIndexer *indexer,
                                     const std::vector<std::string> &numeric_colnames,
                                     const std::vector<std::string> &categ_colnames,
                                     const std::vector<std::vector<std::string>> &categ_levels,
                                     bool output_tree_num, bool index1, size_t tree_num)
{
    std::string graph_str("");
    if (interrupt_switch) return graph_str;

    const size_t *restrict terminal_node_mappings = nullptr;
    std::unique_ptr<size_t[]> terminal_node_mappings_holder(nullptr);
    if (output_tree_num)
    {
        get_tree_mappings(
            terminal_node_mappings,
            terminal_node_mappings_holder,
            model_outputs,
            model_outputs_ext,
            indexer,
            tree_num
        );
    }

    if (model_outputs)
    {
        traverse_isoforest_graphviz(
            graph_str, 0,
            *model_outputs, model_outputs->trees[tree_num],
            terminal_node_mappings,
            numeric_colnames,
            categ_colnames,
            categ_levels,
            output_tree_num, index1, tree_num
        );
    }

    else
    {
        traverse_ext_graphviz(
            graph_str, 0,
            *model_outputs_ext, model_outputs_ext->hplanes[tree_num],
            terminal_node_mappings,
            numeric_colnames,
            categ_colnames,
            categ_levels,
            output_tree_num, index1, tree_num
        );
    }

    if (interrupt_switch) return graph_str;

    return
        "digraph {\n    graph [ rankdir=TB ]\n\n" +
        graph_str +
        "}\n";
}

void get_tree_mappings
(
    const size_t *restrict &terminal_node_mappings,
    std::unique_ptr<size_t[]> &terminal_node_mappings_holder,
    const IsoForest *model_outputs,
    const ExtIsoForest *model_outputs_ext,
    const TreesIndexer *indexer,
    size_t tree_num
)
{
    if (indexer && !indexer->indices.empty() && !indexer->indices[tree_num].terminal_node_mappings.empty())
    {
        terminal_node_mappings = indexer->indices[tree_num].terminal_node_mappings.data();
    }

    else
    {
        if (model_outputs)
        {
            const std::vector<IsoTree> *trees = &model_outputs->trees[tree_num];
            terminal_node_mappings_holder = std::unique_ptr<size_t[]>(new size_t[trees->size()]);
            size_t curr = 0;
            for (size_t ix = 0; ix < trees->size(); ix++)
            {
                if ((*trees)[ix].tree_left == 0)
                {
                    terminal_node_mappings_holder[ix] = curr++;
                }
            }
            terminal_node_mappings = terminal_node_mappings_holder.get();
        }

        else if (model_outputs_ext)
        {
            const std::vector<IsoHPlane> *hplanes = &model_outputs_ext->hplanes[tree_num];
            terminal_node_mappings_holder = std::unique_ptr<size_t[]>(new size_t[hplanes->size()]);
            size_t curr = 0;
            for (size_t ix = 0; ix < hplanes->size(); ix++)
            {
                if ((*hplanes)[ix].hplane_left == 0)
                {
                    terminal_node_mappings_holder[ix] = curr++;
                }
            }
            terminal_node_mappings = terminal_node_mappings_holder.get();
        }

        else
        {
            unexpected_error();
        }
    }
}

void escape_strings
(
    const std::vector<std::string> &numeric_colnames,
    const std::vector<std::string> &categ_colnames,
    const std::vector<std::vector<std::string>> &categ_levels,
    std::vector<std::string> &numeric_colnames_out,
    std::vector<std::string> &categ_colnames_out,
    std::vector<std::vector<std::string>> &categ_levels_out
)
{
    numeric_colnames_out.clear(); numeric_colnames_out.reserve(numeric_colnames.size());
    categ_colnames_out.clear(); categ_colnames_out.reserve(categ_colnames.size());
    categ_levels_out.clear(); categ_levels_out.resize(categ_levels.size());

    for (const std::string &s : numeric_colnames)
    {
        numeric_colnames_out.push_back(
            std::regex_replace(s, std::regex("\""), "\\\"")
        );
    }

    for (const std::string &s : categ_colnames)
    {
        categ_colnames_out.push_back(
            std::regex_replace(s, std::regex("\""), "\\\"")
        );
    }

    for (size_t ix = 0; ix < categ_levels.size(); ix++)
    {
        categ_levels_out[ix].clear();
        categ_levels_out[ix].reserve(categ_levels[ix].size());
        std::vector<std::string> *categ_levels_this = &categ_levels_out[ix];

        for (const std::string &s : categ_levels[ix])
        {
            categ_levels_this->push_back(
                std::regex_replace(s, std::regex("\""), "\\\"")
            );
        }
    }
}

std::string format_pct(double fraction)
{
    fraction = std::fmin(fraction, 1.);
    fraction = std::fmax(fraction, 0.);
    char buffer[10];
    std::snprintf(buffer, 10, "%.2f%%", 100. * fraction);
    return std::string(buffer);
}

void traverse_isoforest_graphviz
(
    std::string &curr_labels, size_t curr_node,
    const IsoForest &model, const std::vector<IsoTree> &nodes,
    const size_t *restrict terminal_node_mappings,
    const std::vector<std::string> &numeric_colnames,
    const std::vector<std::string> &categ_colnames,
    const std::vector<std::vector<std::string>> &categ_levels,
    bool output_tree_num, bool index1, size_t tree_num
)
{
    if (interrupt_switch) return;
    const std::string s_curr = std::to_string(curr_node);
    const IsoTree *tree = &nodes[curr_node];

    /* terminal node */
    if (tree->tree_left == 0)
    {
        if (!output_tree_num)
        {
            const double score_write = (model.scoring_metric != Density && model.scoring_metric != BoxedRatio)?
                                        tree->score : -tree->score;
            curr_labels.append(
                s_curr + " [ label=\"leaf=" + std::to_string(score_write) + "\" ]\n\n"
            );
        }
        else
        {
            const size_t score_write = terminal_node_mappings[curr_node];
            curr_labels.append(
                s_curr + " [ label=\"node=" + std::to_string(score_write + (size_t)index1) + "\" ]\n\n"
            );
        }

        return;
    }

    /* split condition */
    curr_labels.append(s_curr + " [ label=\"");
    switch (tree->col_type)
    {
        case Numeric:
        {
            curr_labels.append(
                numeric_colnames[tree->col_num] +
                "<=" +
                std::to_string(tree->num_split)
            );
            break;
        }

        case Categorical:
        {
            switch (model.cat_split_type)
            {
                case SingleCateg:
                {
                    curr_labels.append(
                        categ_colnames[tree->col_num] +
                        "=" +
                        categ_levels[tree->col_num][tree->chosen_cat]
                    );
                    break;
                }

                case SubSet:
                {
                    curr_labels.append(categ_colnames[tree->col_num] + "={");
                    bool added_left = false;
                    for (size_t categ = 0; categ < tree->cat_split.size(); categ++)
                    {
                        if (
                            tree->cat_split[categ] == 1 ||
                            (
                                tree->cat_split[categ] == -1 &&
                                (
                                    (model.new_cat_action == Smallest && tree->pct_tree_left < .5) ||
                                    (model.missing_action == Impute && tree->pct_tree_left >= .5)
                                )
                            )
                        )
                        {
                            curr_labels.append(
                                (added_left? "," : "") +
                                categ_levels[tree->col_num][categ]
                            );
                            added_left = true;
                        }
                    }

                    curr_labels.append("}");
                    break;
                }
            }
            break;
        }

        default:
        {
            unexpected_error();
            break;
        }
    }
    curr_labels.append("\" ]\n");

    const char *s_missing_left = (model.missing_action == Impute && tree->pct_tree_left >= .5)? ", missing" : "";
    const char *s_missing_right = (model.missing_action == Impute && tree->pct_tree_left < .5)? ", missing" : "";

    curr_labels.append(
        /* left branch */
        s_curr + " -> " +
        std::to_string(tree->tree_left) +
        " [label=\"yes" +
        s_missing_left + " (" + format_pct(tree->pct_tree_left) + ")" +
        "\" color=\"#FF0000\"]\n" +

        /* right branch */
        s_curr + " -> " +
        std::to_string(tree->tree_right) +
        " [label=\"no" +
        s_missing_right + " (" + format_pct(1. - tree->pct_tree_left) + ")" +
        "\" color=\"#0000FF\"]\n\n"
    );

    traverse_isoforest_graphviz(
        curr_labels,
        tree->tree_left,
        model, nodes,
        terminal_node_mappings,
        numeric_colnames,
        categ_colnames,
        categ_levels,
        output_tree_num, index1, tree_num
    );

    traverse_isoforest_graphviz(
        curr_labels,
        tree->tree_right,
        model, nodes,
        terminal_node_mappings,
        numeric_colnames,
        categ_colnames,
        categ_levels,
        output_tree_num, index1, tree_num
    );
}

void traverse_ext_graphviz
(
    std::string &curr_labels, size_t curr_node,
    const ExtIsoForest &model, const std::vector<IsoHPlane> &nodes,
    const size_t *restrict terminal_node_mappings,
    const std::vector<std::string> &numeric_colnames,
    const std::vector<std::string> &categ_colnames,
    const std::vector<std::vector<std::string>> &categ_levels,
    bool output_tree_num, bool index1, size_t tree_num
)
{
    if (interrupt_switch) return;
    const std::string s_curr = std::to_string(curr_node);
    const IsoHPlane *hplane = &nodes[curr_node];

    /* terminal node */
    if (hplane->hplane_left == 0)
    {
        if (!output_tree_num)
        {
            const double score_write = (model.scoring_metric != Density && model.scoring_metric != BoxedRatio)?
                                        hplane->score : -hplane->score;
            curr_labels.append(
                s_curr + " [ label=\"leaf=" + std::to_string(score_write) + "\" ]\n\n"
            );
        }
        else
        {
            const size_t score_write = terminal_node_mappings[curr_node];
            curr_labels.append(
                s_curr + " [ label=\"node=" + std::to_string(score_write + (size_t)index1) + "\" ]\n\n"
            );
        }

        return;
    }

    /* split condition */
    curr_labels.append(s_curr + " [ label=\"");
    size_t n_visited_numeric = 0;
    size_t n_visited_categ = 0;
    for (size_t ix = 0; ix < hplane->col_num.size(); ix++)
    {
        if (ix > 0) curr_labels.append(" + ");

        switch (hplane->col_type[ix])
        {
            case Numeric:
            {
                curr_labels.append(
                    "[" +
                    std::to_string(hplane->coef[n_visited_numeric]) +
                    " * (" +
                    numeric_colnames[hplane->col_num[ix]] +
                    ((hplane->mean[n_visited_numeric] >= 0.)? " - " : " - (") +
                    std::to_string(hplane->mean[n_visited_numeric]) +
                    ((hplane->mean[n_visited_numeric] >= 0.)? ")" : "))") +
                    "]"
                );
                n_visited_numeric++;
                break;
            }

            case Categorical:
            {
                switch (model.cat_split_type)
                {
                    case SingleCateg:
                    {
                        curr_labels.append(
                            "[" +
                            std::to_string(hplane->fill_new[n_visited_categ]) +
                            " * (" +
                            categ_colnames[hplane->col_num[ix]] +
                            "=" +
                            categ_levels[hplane->col_num[ix]][hplane->chosen_cat[n_visited_categ]] +
                            ")]"
                        );
                        break;
                    }

                    case SubSet:
                    {
                        curr_labels.append(
                            "[" +
                            categ_colnames[hplane->col_num[ix]] +
                            "={"
                        );
                        for (size_t categ = 0; categ < hplane->cat_coef[n_visited_categ].size(); categ++)
                        {
                            curr_labels.append(
                                categ_levels[hplane->col_num[ix]][categ] +
                                ":" +
                                std::to_string(hplane->cat_coef[n_visited_categ][categ]) +
                                (
                                    (categ < hplane->cat_coef[n_visited_categ].size() - 1)?
                                    ", " : ""
                                )
                            );
                        }
                        curr_labels.append("}]");
                        break;
                    }
                }
                n_visited_categ++;
                break;
            }

            default:
            {
                unexpected_error();
                break;
            }
        }
    }

    curr_labels.append("\" ]\n");

    curr_labels.append(
        /* left branch */
        s_curr + " -> " +
        std::to_string(hplane->hplane_left) +
        " [label=\"yes\" color=\"#FF0000\"]\n" +

        /* right branch */
        s_curr + " -> " +
        std::to_string(hplane->hplane_right) +
        " [label=\"no\" color=\"#0000FF\"]\n\n"
    );

    traverse_ext_graphviz(
        curr_labels,
        hplane->hplane_left,
        model, nodes,
        terminal_node_mappings,
        numeric_colnames,
        categ_colnames,
        categ_levels,
        output_tree_num, index1, tree_num
    );

    traverse_ext_graphviz(
        curr_labels,
        hplane->hplane_right,
        model, nodes,
        terminal_node_mappings,
        numeric_colnames,
        categ_colnames,
        categ_levels,
        output_tree_num, index1, tree_num
    );
}

/* Generate a JSON string representation of model trees
* 
* Parameters
* ==========
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if using an extended model. Can only pass one of 'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if using a single-variable model. Can only pass one of 'model_outputs' and 'model_outputs_ext'.
* - numeric_colnames
*       Names to use for the numerical columns.
* - categ_colnames
*       Names to use for the categorical columns.
* - categ_levels
*       Names to use for the levels/categories of each categorical column.
* - output_tree_num
*       Whether to output the terminal node number instead of the isolation depth at each node.
* - index1
*       Whether to make the node numbers start their numeration at 1 instead of 0 in the
*       resulting strings. Ignored when passing 'output_tree_num=false'.
* - single_tree
*       Whether to generate the representation for a single tree of the model instead of for
*       all. The tree number to generate is to be passed under 'tree_num'.
* - tree_num
*       Tree number for which to generate a JSON string, if passing 'single_tree=true'.
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used. Ignored when not building with
*       OpenMP support.
* 
* Returns
* =======
* A vector containing at each element the tree nodes as a JSON string representation
* of the corresponding tree in the model.
* If passing 'single_tree=true', will contain only one element, corresponding to the tree given
* in 'tree_num'.
*/
std::vector<std::string> generate_json(const IsoForest *model_outputs,
                                       const ExtIsoForest *model_outputs_ext,
                                       const TreesIndexer *indexer,
                                       const std::vector<std::string> &numeric_colnames,
                                       const std::vector<std::string> &categ_colnames,
                                       const std::vector<std::vector<std::string>> &categ_levels,
                                       bool output_tree_num, bool index1, bool single_tree, size_t tree_num,
                                       int nthreads)
{
    if (!model_outputs && !model_outputs_ext) throw std::runtime_error("'generate_json' got a NULL pointer for model.");
    if (model_outputs && model_outputs_ext) throw std::runtime_error("'generate_json' got two models as inputs.");

    std::vector<std::string> numeric_colnames_escaped;
    std::vector<std::string> categ_colnames_escaped;
    std::vector<std::vector<std::string>> categ_levels_escaped;
    escape_strings(
        numeric_colnames,
        categ_colnames,
        categ_levels,
        numeric_colnames_escaped,
        categ_colnames_escaped,
        categ_levels_escaped
    );

    size_t ntrees = model_outputs? model_outputs->trees.size() : model_outputs_ext->hplanes.size();

    std::vector<std::string> out;

    if (single_tree)
    {
        if (index1) tree_num--;
        out.push_back(
            generate_json_single_tree(
                model_outputs,
                model_outputs_ext,
                indexer,
                numeric_colnames_escaped,
                categ_colnames_escaped,
                categ_levels_escaped,
                output_tree_num, index1, tree_num
            )
        );
        return out;
    }

    out.resize(ntrees);

    /* Global variable that determines if the procedure receives a stop signal */
    SignalSwitcher ss = SignalSwitcher();

    /* For exception handling */
    bool threw_exception = false;
    std::exception_ptr ex = NULL;

    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(out, index1, tree_num, threw_exception, ex)
    for (size_t_for ix = 0; ix < (size_t_for)ntrees; ix++)
    {
        if (interrupt_switch || threw_exception)
            continue; /* Cannot break with OpenMP==2.0 (MSVC) */

        try
        {
            out[ix] = generate_json_single_tree(
                model_outputs,
                model_outputs_ext,
                indexer,
                numeric_colnames_escaped,
                categ_colnames_escaped,
                categ_levels_escaped,
                output_tree_num, index1, ix
            );
        }

        catch (...)
        {
            #pragma omp critical
            {
                if (!threw_exception)
                {
                    threw_exception = true;
                    ex = std::current_exception();
                }
            }
        }
    }

    /* check if the procedure got interrupted */
    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return EXIT_FAILURE;
    #endif

    /* check if some exception was thrown */
    if (threw_exception)
        std::rethrow_exception(ex);

    return out;
}

std::string generate_json_single_tree(const IsoForest *model_outputs,
                                      const ExtIsoForest *model_outputs_ext,
                                      const TreesIndexer *indexer,
                                      const std::vector<std::string> &numeric_colnames,
                                      const std::vector<std::string> &categ_colnames,
                                      const std::vector<std::vector<std::string>> &categ_levels,
                                      bool output_tree_num, bool index1, size_t tree_num)
{
    std::string json_str("");
    if (interrupt_switch) return json_str;

    const size_t *restrict terminal_node_mappings = nullptr;
    std::unique_ptr<size_t[]> terminal_node_mappings_holder(nullptr);
    get_tree_mappings(
        terminal_node_mappings,
        terminal_node_mappings_holder,
        model_outputs,
        model_outputs_ext,
        indexer,
        tree_num
    );

    if (model_outputs)
    {
        traverse_isoforest_json(
            json_str, 0,
            *model_outputs, model_outputs->trees[tree_num],
            terminal_node_mappings,
            numeric_colnames,
            categ_colnames,
            categ_levels,
            output_tree_num, index1, tree_num
        );
    }

    else
    {
        traverse_ext_json(
            json_str, 0,
            *model_outputs_ext, model_outputs_ext->hplanes[tree_num],
            terminal_node_mappings,
            numeric_colnames,
            categ_colnames,
            categ_levels,
            output_tree_num, index1, tree_num
        );
    }

    if (interrupt_switch) return json_str;

    return
        "{" +
        json_str +
        "}";
}

void traverse_isoforest_json
(
    std::string &curr_json, size_t curr_node,
    const IsoForest &model, const std::vector<IsoTree> &nodes,
    const size_t *restrict terminal_node_mappings,
    const std::vector<std::string> &numeric_colnames,
    const std::vector<std::string> &categ_colnames,
    const std::vector<std::vector<std::string>> &categ_levels,
    bool output_tree_num, bool index1, size_t tree_num
)
{
    if (interrupt_switch) return;
    const IsoTree *tree = &nodes[curr_node];
    if (curr_node > 0) curr_json.append(",");

    curr_json.append(
        "\"" +
        std::to_string(curr_node + (size_t)index1) +
        "\": {\"terminal\":\""
    );

    /* terminal node */
    if (tree->tree_left == 0)
    {
        const double score_write = (model.scoring_metric != Density && model.scoring_metric != BoxedRatio)?
                                    tree->score : -tree->score;
        const size_t terminal_idx = terminal_node_mappings[curr_node];
        curr_json.append(
            "yes\", \"score\":" +
            std::to_string(score_write) +
            ", \"leaf\":" +
            std::to_string(terminal_idx) +
            "}"
        );
        return;
    }

    curr_json.append(
        "no\", \"node_when_condition_is_met\":" +
        std::to_string(tree->tree_left + (size_t)index1) +
        ", \"node_when_condition_is_not_met\":" +
        std::to_string(tree->tree_right + (size_t)index1) +
        ", \"fraction_yes\":" +
        std::to_string(tree->pct_tree_left) +
        ", \"column\":"
    );

    switch (tree->col_type)
    {
        case Numeric:
        {
            curr_json.append(
                "\"" +
                numeric_colnames[tree->col_num] +
                "\", \"column_type\":\"numeric\", \"condition\":\"<=\", \"value\":" +
                std::to_string(tree->num_split)
            );
            if (model.has_range_penalty)
            {
                curr_json.append(
                    ", \"range_low\":" +
                    std::to_string(tree->range_low) +
                    ", \"range_high\":" +
                    std::to_string(tree->range_high)
                );
            }
            break;
        }

        case Categorical:
        {
            curr_json.append(
                "\"" +
                categ_colnames[tree->col_num] +
                "\", \"column_type\":\"categorical\", \"condition\":\""
            );
            switch (model.cat_split_type)
            {
                case SingleCateg:
                {
                    curr_json.append(
                        "=\", \"value\":\"" +
                        categ_levels[tree->col_num][tree->chosen_cat] +
                        "\""
                    );
                    break;
                }

                case SubSet:
                {
                    curr_json.append("map\", \"value\":{");
                    for (size_t categ = 0; categ < tree->cat_split.size(); categ++)
                    {
                        if (categ > 0) curr_json.append(",");
                        curr_json.append(
                            "\"" +
                            categ_levels[tree->col_num][categ] +
                            "\":\""
                        );
                        switch (tree->cat_split[categ])
                        {
                            case 1:
                            {
                                curr_json.append("yes");
                                break;
                            }

                            case 0:
                            {
                                curr_json.append("no");
                                break;
                            }

                            case -1:
                            {
                                if (model.new_cat_action == Smallest || model.missing_action == Impute)
                                {
                                    if ((model.new_cat_action == Smallest && tree->pct_tree_left < .5) ||
                                        (model.missing_action == Impute && tree->pct_tree_left >= .5))
                                    {
                                        curr_json.append("yes");
                                    }

                                    else
                                    {
                                        curr_json.append("no");
                                    }
                                }

                                else if (model.missing_action == Divide)
                                {
                                    curr_json.append("both (weighted)");
                                }

                                else
                                {
                                    curr_json.append("undefined");
                                }
                                break;
                            }

                            default:
                            {
                                unexpected_error();
                            }
                        }
                        curr_json.append("\"");
                    }
                    curr_json.append("}");
                    break;
                }
            }
            break;
        }

        default:
        {
            unexpected_error();
        }
    }

    curr_json.append(
        "}"
    );

    traverse_isoforest_json(
        curr_json,
        tree->tree_left,
        model, nodes,
        terminal_node_mappings,
        numeric_colnames,
        categ_colnames,
        categ_levels,
        output_tree_num, index1, tree_num
    );

    traverse_isoforest_json(
        curr_json,
        tree->tree_right,
        model, nodes,
        terminal_node_mappings,
        numeric_colnames,
        categ_colnames,
        categ_levels,
        output_tree_num, index1, tree_num
    );
}

void traverse_ext_json
(
    std::string &curr_json, size_t curr_node,
    const ExtIsoForest &model, const std::vector<IsoHPlane> &nodes,
    const size_t *restrict terminal_node_mappings,
    const std::vector<std::string> &numeric_colnames,
    const std::vector<std::string> &categ_colnames,
    const std::vector<std::vector<std::string>> &categ_levels,
    bool output_tree_num, bool index1, size_t tree_num
)
{
    if (interrupt_switch) return;
    const IsoHPlane *hplane = &nodes[curr_node];
    if (curr_node > 0) curr_json.append(",");

    curr_json.append(
        "\"" +
        std::to_string(curr_node + (size_t)index1) +
        "\": {\"terminal\":\""
    );

    /* terminal node */
    if (hplane->hplane_left == 0)
    {
        const double score_write = (model.scoring_metric != Density && model.scoring_metric != BoxedRatio)?
                                    hplane->score : -hplane->score;
        const size_t terminal_idx = terminal_node_mappings[curr_node];
        curr_json.append(
            "yes\", \"score\":" +
            std::to_string(score_write) +
            ", \"leaf\":" +
            std::to_string(terminal_idx) +
            "}"
        );
        return;
    }

    curr_json.append(
        "no\", \"node_when_condition_is_met\":" +
        std::to_string(hplane->hplane_left + (size_t)index1) +
        ", \"node_when_condition_is_not_met\":" +
        std::to_string(hplane->hplane_right + (size_t)index1) +
        ", \"combination\":["
    );

    size_t n_visited_numeric = 0;
    size_t n_visited_categ = 0;
    for (size_t ix = 0; ix < hplane->col_num.size(); ix++)
    {
        if (ix > 0) curr_json.append(", ");
        curr_json.append("{");
        switch (hplane->col_type[ix])
        {
            case Numeric:
            {
                curr_json.append(
                    "\"column\":\"" +
                    numeric_colnames[hplane->col_num[ix]] +
                    "\", \"column_type\":\"numeric\", \"coefficient\":" +
                    std::to_string(hplane->coef[n_visited_numeric]) +
                    ", \"centering\":" +
                    std::to_string(hplane->mean[n_visited_numeric])
                );
                if (model.missing_action != Fail)
                {
                    curr_json.append(
                        ", \"imputation_value\":" +
                        std::to_string(hplane->fill_val[ix])
                    );
                }
                n_visited_numeric++;
                break;
            }

            case Categorical:
            {
                curr_json.append(
                    "\"column\":\"" +
                    categ_colnames[hplane->col_num[ix]] +
                    "\", \"column_type\":\"categorical\", "
                );
                switch (model.cat_split_type)
                {
                    case SingleCateg:
                    {
                        curr_json.append(
                            "\"category\":\"" +
                            categ_levels[hplane->col_num[ix]][hplane->chosen_cat[n_visited_categ]] +
                            "\", \"coefficient_category\":" +
                            std::to_string(hplane->fill_new[n_visited_categ]) +
                            ", \"coefficient_other_categories\":0.0"
                        );
                        break;
                    }

                    case SubSet:
                    {
                        curr_json.append(
                            "\"coefficients\":{"
                        );
                        for (size_t categ = 0; categ < hplane->cat_coef[n_visited_categ].size(); categ++)
                        {
                            if (categ > 0) curr_json.append(", ");
                            curr_json.append(
                                "\"" +
                                categ_levels[hplane->col_num[ix]][categ] +
                                "\":" +
                                std::to_string(hplane->cat_coef[n_visited_categ][categ])
                            );
                        }
                        curr_json.append("}");
                        break;
                    }
                }
                if (model.missing_action != Fail)
                {
                    curr_json.append(
                        ", \"imputation_value\":" +
                        std::to_string(hplane->fill_val[ix])
                    );
                }
                n_visited_categ++;
                break;
            }

            default:
            {
                unexpected_error();
                break;
            }
        }
        curr_json.append("}");
    }

    curr_json.append(
        "], \"condition\":\"<=\", \"value\":" +
        std::to_string(hplane->split_point)
    );

    if (model.has_range_penalty)
    {
        curr_json.append(
            ", \"range_low\":" +
            std::to_string(hplane->range_low) +
            ", \"range_high\":" +
            std::to_string(hplane->range_high)
        );
    }

    curr_json.append("}");

    traverse_ext_json(
        curr_json,
        hplane->hplane_left,
        model, nodes,
        terminal_node_mappings,
        numeric_colnames,
        categ_colnames,
        categ_levels,
        output_tree_num, index1, tree_num
    );

    traverse_ext_json(
        curr_json,
        hplane->hplane_right,
        model, nodes,
        terminal_node_mappings,
        numeric_colnames,
        categ_colnames,
        categ_levels,
        output_tree_num, index1, tree_num
    );
}
