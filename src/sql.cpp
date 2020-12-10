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
*     Copyright (c) 2020, David Cortes
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

/* Translate isolation forest model into a single SQL select statement
* 
* Parameters
* ==========
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from an extended model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from a single-variable model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - table_from
*       Table name from where the columns used in the model will be selected.
* - select_as
*       Alias to give to the outlier score in the select statement.
* - numeric_colnames
*       Names to use for the numerical columns.
* - categ_colnames
*       Names to use for the categorical columns.
* - categ_levels
*       Names to use for the levels/categories of each categorical column. These will be enclosed
*       in single quotes.
* - index1
*       Whether to make the node numbers start their numeration at 1 instead of 0 in the
*       resulting statement. If passing 'output_tree_num=false', this will only affect the
*       commented lines which act as delimiters. If passing 'output_tree_num=true', will also
*       affect the results (which will also start at 1).
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used. Ignored when not building with
*       OpenMP support.
* 
* Returns
* =======
* A string with the corresponding SQL statement that will calculate the outlier score
* from the model.
*/
std::string generate_sql_with_select_from(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                          std::string &table_from, std::string &select_as,
                                          std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                                          std::vector<std::vector<std::string>> &categ_levels,
                                          bool index1, int nthreads)
{
    std::vector<std::string> tree_conds = generate_sql(model_outputs, model_outputs_ext,
                                                       numeric_colnames, categ_colnames,
                                                       categ_levels,
                                                       false, index1, false, 0,
                                                       nthreads);
    std::string out = std::accumulate(tree_conds.begin(), tree_conds.end(), std::string("SELECT\nPOWER(2.0, -(0.0"),
                                      [&tree_conds, &index1](std::string &a, std::string &b)
                                      {return a
                                                + std::string(" + \n---BEGIN TREE ")
                                                + std::to_string((size_t)std::distance(tree_conds.data(), &b) + (size_t)index1)
                                                + std::string("---\n")
                                                + b
                                                + std::string("\n---END OF TREE ")
                                                + std::to_string((size_t)std::distance(tree_conds.data(), &b) + (size_t)index1)
                                                + std::string("---\n");});
    size_t ntrees = (model_outputs != NULL)? (model_outputs->trees.size()) : (model_outputs_ext->hplanes.size());
   return
       out
        + std::string(") / ")
        + std::to_string((long double)ntrees * ((model_outputs != NULL)?
                                                (model_outputs->exp_avg_depth) : (model_outputs_ext->exp_avg_depth)))
        + std::string(") AS ")
        + select_as
        + std::string("\nFROM ")
        + table_from;
}

/* Translate model trees into SQL select statements
* 
* Parameters
* ==========
* - model_outputs
*       Pointer to fitted single-variable model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from an extended model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - model_outputs_ext
*       Pointer to fitted extended model object from function 'fit_iforest'. Pass NULL
*       if the predictions are to be made from a single-variable model. Can only pass one of
*       'model_outputs' and 'model_outputs_ext'.
* - numeric_colnames
*       Names to use for the numerical columns.
* - categ_colnames
*       Names to use for the categorical columns.
* - categ_levels
*       Names to use for the levels/categories of each categorical column. These will be enclosed
*       in single quotes.
* - output_tree_num
*       Whether to output the terminal node number instead of the separation depth at each node.
* - index1
*       Whether to make the node numbers start their numeration at 1 instead of 0 in the
*       resulting statement. If passing 'output_tree_num=false', this will only affect the
*       commented lines which act as delimiters. If passing 'output_tree_num=true', will also
*       affect the results (which will also start at 1).
* - single_tree
*       Whether to generate the select statement for a single tree of the model instead of for
*       all. The tree number to generate is to be passed under 'tree_num'.
* - tree_num
*       Tree number for which to generate an SQL select statement, if passing 'single_tree=true'.
* - nthreads
*       Number of parallel threads to use. Note that, the more threads, the more memory will be
*       allocated, even if the thread does not end up being used. Ignored when not building with
*       OpenMP support.
* 
* Returns
* =======
* A vector containing at each element the SQL statement for the corresponding tree in the model.
* If passing 'single_tree=true', will contain only one element, corresponding to the tree given
* in 'tree_num'. The statements will be node-by-node, with commented-out separators using '---'
* as delimiters and including the node number as part of the comment.
*/
std::vector<std::string> generate_sql(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                      std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                                      std::vector<std::vector<std::string>> &categ_levels,
                                      bool output_tree_num, bool index1, bool single_tree, size_t tree_num,
                                      int nthreads)
{
    bool output_score = !output_tree_num;
    size_t ntrees_use = single_tree?
                            1 : ((model_outputs != NULL)?
                                    model_outputs->trees.size() : model_outputs_ext->hplanes.size());
    std::string initial_str = std::string("\tWHEN\n");

    size_t_for loop_st = 0;
    size_t_for loop_end = ntrees_use;
    if (single_tree)
    {
        loop_st = tree_num;
        loop_end = loop_st + 1;
    }

    /* determine maximum number of nodes in a tree */
    size_t max_nodes = 0;
    for (size_t tree = loop_st; tree < loop_end; tree++)
        max_nodes = std::max(max_nodes,
                             (model_outputs != NULL)?
                                (model_outputs->trees[tree].size()) : (model_outputs_ext->hplanes[tree].size()));
    std::vector<std::string> conditions_left(max_nodes);
    std::vector<std::string> conditions_right(max_nodes);

    std::vector<std::vector<std::string>> all_node_rules(ntrees_use);
    std::vector<std::string> out(ntrees_use);

    size_t tree_use;

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            shared(model_outputs, model_outputs_ext, numeric_colnames, categ_colnames, categ_levels, \
                   loop_st, loop_end, index1, single_tree, all_node_rules, out) \
            firstprivate(conditions_left, conditions_right) private(tree_use)
    for (size_t_for tree = loop_st; tree < loop_end; tree++)
    {
        if (model_outputs != NULL)
        {
            for (size_t node = 0; node < model_outputs->trees[tree].size(); node++)
                extract_cond_isotree(*model_outputs, model_outputs->trees[tree][node],
                                     conditions_left[node], conditions_right[node],
                                     numeric_colnames, categ_colnames,
                                     categ_levels);
        }

        else
        {
            for (size_t node = 0; node < model_outputs_ext->hplanes[tree].size(); node++)
                extract_cond_ext_isotree(*model_outputs_ext, model_outputs_ext->hplanes[tree][node],
                                         conditions_left[node], conditions_right[node],
                                         numeric_colnames, categ_colnames,
                                         categ_levels);
        }

        generate_tree_rules(
            (model_outputs == NULL)? (NULL) : &(model_outputs->trees[tree]),
            (model_outputs_ext == NULL)? (NULL) : &(model_outputs_ext->hplanes[tree]),
            output_score,
            0, index1, initial_str, all_node_rules[single_tree? 0 : tree],
            conditions_left, conditions_right
        );

        /* Code below doesn't compile with MSVC (stuck with an OMP standard that's >20 years old) */
        // if (single_tree)
        //     tree = 0;
        tree_use = single_tree? (size_t)0 : tree;

        if (all_node_rules[tree_use].size() <= 1)
        {
            for (std::string &rule : all_node_rules[tree_use])
                rule = std::string("WHEN TRUE THEN ")
                        + std::to_string((model_outputs != NULL)?
                                            (model_outputs->exp_avg_depth) : (model_outputs_ext->exp_avg_depth))
                        + std::string(" ");
        }

        out[tree_use] = std::accumulate(all_node_rules[tree_use].begin(), all_node_rules[tree_use].end(),
                                        std::string("CASE\n"),
                                        [&all_node_rules, &tree_use, &index1](std::string &a, std::string &b)
                                        {return a
                                                    + std::string("---begin terminal node ")
                                                    + std::to_string((size_t)std::distance(&(all_node_rules[tree_use][0]), &b) + (size_t)index1)
                                                    + std::string("---\n")
                                                + b;})
                        + std::string("END\n");
        all_node_rules[tree_use].clear();
    }

    return out;
} 


void generate_tree_rules(std::vector<IsoTree> *trees, std::vector<IsoHPlane> *hplanes, bool output_score,
                         size_t curr_ix, bool index1, std::string &prev_cond, std::vector<std::string> &node_rules,
                         std::vector<std::string> &conditions_left, std::vector<std::string> &conditions_right)
{
    if ((trees != NULL && (*trees)[curr_ix].score >= 0) ||
        (hplanes != NULL && (*hplanes)[curr_ix].score >= 0))
    {
        node_rules.push_back(prev_cond
                                + std::string("\tTHEN ")
                                + (output_score?
                                    (std::to_string((trees != NULL)?
                                        ((*trees)[curr_ix].score) : ((*hplanes)[curr_ix].score)))
                                        :
                                    (std::to_string(node_rules.size() + (size_t)index1)))
                                + std::string("\n---end of terminal node ")
                                + std::to_string(node_rules.size() + (size_t)index1)
                                + std::string("---\n"));
        return;
    }

    
    std::string cond_left = prev_cond
                                + ((curr_ix > 0)? std::string("\t\tAND (") : std::string("\t\t    ("))
                                + conditions_left[curr_ix]
                                + std::string(")\n");
    generate_tree_rules(trees, hplanes, output_score,
                        (trees != NULL)?
                            ((*trees)[curr_ix].tree_left) : ((*hplanes)[curr_ix].hplane_left),
                        index1, cond_left, node_rules,
                        conditions_left, conditions_right);
    cond_left.clear();
    std::string cond_right = prev_cond
                                + ((curr_ix > 0)? std::string("\t\tAND (") : std::string("\t\t    ("))
                                + conditions_right[curr_ix]
                                + std::string(")\n");
    generate_tree_rules(trees, hplanes, output_score,
                        (trees != NULL)?
                            ((*trees)[curr_ix].tree_right) : ((*hplanes)[curr_ix].hplane_right),
                        index1, cond_right, node_rules,
                        conditions_left, conditions_right);
}


void extract_cond_isotree(IsoForest &model, IsoTree &tree,
                          std::string &cond_left, std::string &cond_right,
                          std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                          std::vector<std::vector<std::string>> &categ_levels)
{
    cond_left = std::string("");
    cond_right = std::string("");
    if (tree.score >= 0.)
        return;

    switch(tree.col_type)
    {
        case Numeric:
        {
            cond_left = ((model.missing_action != Impute)? (std::string("")) :
                            ((tree.pct_tree_left >= .5)?
                                (numeric_colnames[tree.col_num]
                                    + std::string(" IS NULL OR "))
                                :
                                (numeric_colnames[tree.col_num]
                                    + std::string(" IS NOT NULL AND "))))
                            + numeric_colnames[tree.col_num]
                            + std::string(" <= ")
                            + std::to_string(tree.num_split);
            cond_right = ((model.missing_action != Impute)? (std::string("")) :
                            ((tree.pct_tree_left >= .5)?
                                (numeric_colnames[tree.col_num]
                                    + std::string(" IS NOT NULL AND "))
                                :
                                (numeric_colnames[tree.col_num]
                                    + std::string(" IS NULL OR "))))
                            + numeric_colnames[tree.col_num]
                            + std::string(" > ")
                            + std::to_string(tree.num_split);
            break;
        }

        case Categorical:
        {
            switch(model.cat_split_type)
            {
                case SingleCateg:
                {
                    cond_left = ((model.missing_action != Impute)? (std::string("")) :
                                    ((model.missing_action == Impute && tree.pct_tree_left >= .5)?
                                        (categ_colnames[tree.col_num]
                                            + std::string(" IS NULL OR "))
                                        :
                                        (categ_colnames[tree.col_num]
                                            + std::string(" IS NOT NULL AND "))))
                                    + categ_colnames[tree.col_num]
                                    + std::string(" = '")
                                    + categ_levels[tree.col_num][tree.chosen_cat]
                                    + std::string("'");
                    cond_right = ((model.missing_action != Impute)? (std::string("")) :
                                    ((model.missing_action == Impute && tree.pct_tree_left >= .5)?
                                        (categ_colnames[tree.col_num]
                                            + std::string(" IS NOT NULL AND "))
                                        :
                                        (categ_colnames[tree.col_num]
                                            + std::string(" IS NULL OR "))))
                                    + categ_colnames[tree.col_num]
                                    + std::string(" != '")
                                    + categ_levels[tree.col_num][tree.chosen_cat]
                                    + std::string("'");
                    break;
                }

                case SubSet:
                {
                    cond_left = categ_colnames[tree.col_num] + std::string(" IN (");
                    cond_right = cond_left;
                    if (model.missing_action == Impute)
                    {
                        if (tree.pct_tree_left >= .5)
                        {
                            cond_left = categ_colnames[tree.col_num] + std::string(" IS NULL OR ") + cond_left;
                            cond_right = categ_colnames[tree.col_num] + std::string(" IS NOT NULL AND ") + cond_right;
                        }

                        else
                        {
                            cond_left = categ_colnames[tree.col_num] + std::string(" IS NOT NULL AND ") + cond_left;
                            cond_right = categ_colnames[tree.col_num] + std::string(" IS NULL OR ") + cond_right;
                        }
                    }
                    bool added_left = false;
                    bool added_right = false;
                    for (size_t categ = 0; categ < tree.cat_split.size(); categ++)
                    {
                        switch(tree.cat_split[categ])
                        {
                            case 1:
                            {
                                cond_left
                                    +=
                                ((added_left)? (std::string(", ")) : (std::string("")))
                                + std::string("'")
                                + categ_levels[tree.col_num][categ]
                                + std::string("'");
                                added_left = true;
                                break;
                            }

                            case 0:
                            {
                                cond_right
                                    +=
                                ((added_right)? (std::string(", ")) : (std::string("")))
                                + std::string("'")
                                + categ_levels[tree.col_num][categ]
                                + std::string("'");
                                added_right = true;
                                break;
                            }

                            case -1:
                            {
                                if (model.new_cat_action == Smallest || model.missing_action == Impute)
                                {
                                    if ((model.new_cat_action == Smallest && tree.pct_tree_left < .5) ||
                                        (model.missing_action == Impute && tree.pct_tree_left >= .5))
                                    {
                                        cond_left
                                            +=
                                        ((added_left)? (std::string(", ")) : (std::string("")))
                                        + std::string("'")
                                        + categ_levels[tree.col_num][categ]
                                        + std::string("'");
                                        added_left = true;
                                    }
                                    else
                                    {
                                        cond_right
                                            +=
                                        ((added_right)? (std::string(", ")) : (std::string("")))
                                        + std::string("'")
                                        + categ_levels[tree.col_num][categ]
                                        + std::string("'");
                                        added_right = true;
                                    }
                                }
                                break;
                            }
                        }
                    }
                    if (added_left)
                        cond_left += std::string(")");
                    else
                        cond_left = std::string("");
                    if (added_right)
                        cond_right += std::string(")");
                    else
                        cond_right = std::string("");

                    break;
                }
            }
            break;
        }
    }
}

void extract_cond_ext_isotree(ExtIsoForest &model, IsoHPlane &hplane,
                              std::string &cond_left, std::string &cond_right,
                              std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                              std::vector<std::vector<std::string>> &categ_levels)
{
    cond_left = std::string("");
    cond_right = std::string("");
    if (hplane.score >= 0.)
        return;

    std::string hplane_conds = std::string("");

    size_t n_visited_numeric = 0;
    size_t n_visited_categ = 0;
    for (size_t ix = 0; ix < hplane.col_num.size(); ix++)
    {
        hplane_conds
            += 
        ((hplane_conds.length())? (std::string(" + ")) : (std::string("")))
            + ((model.missing_action == Impute)? (std::string("COALESCE(")) : (std::string("")));
        switch(hplane.col_type[ix])
        {
            case Numeric:
            {
                hplane_conds
                    +=
                      std::to_string(hplane.coef[n_visited_numeric])
                    + std::string(" * (")
                    + numeric_colnames[hplane.col_num[ix]]
                    + ((hplane.mean[n_visited_numeric] >= 0.)? (std::string(" - ")) : (std::string(" - ("))) 
                    + std::to_string(hplane.mean[n_visited_numeric])
                    + ((hplane.mean[n_visited_numeric] >= 0.)? (std::string(")")) : (std::string("))")));
                n_visited_numeric++;
                break;
            }

            case Categorical:
            {
                switch(model.cat_split_type)
                {
                    case SingleCateg:
                    {
                        hplane_conds
                            +=
                          std::string("CASE WHEN ")
                        + categ_colnames[hplane.col_num[ix]] 
                        + std::string(" = '")
                        + categ_levels[hplane.col_num[ix]][hplane.chosen_cat[n_visited_categ]]
                        + std::string("' THEN ")
                        + std::to_string(hplane.fill_new[n_visited_categ])
                        + std::string(" ELSE 0.0 END");
                        break;
                    }

                    case SubSet:
                    {
                        hplane_conds += std::string("CASE ") + categ_colnames[hplane.col_num[ix]];
                        for (size_t categ = 0; categ < hplane.cat_coef[hplane.col_num[ix]].size(); categ++)
                        {
                            hplane_conds
                                +=
                              std::string(" WHEN '")
                            + categ_levels[hplane.col_num[ix]][categ]
                            + std::string("' THEN ")
                            + std::to_string( hplane.cat_coef[hplane.col_num[ix]][categ]);
                        }
                        if (model.new_cat_action == Smallest)
                            hplane_conds += std::string(" ELSE ") + std::to_string(hplane.fill_new[n_visited_categ]);
                        hplane_conds += std::string(" END");
                        break;
                    }
                }
                n_visited_categ++;
                break;
            }
        }
        hplane_conds += ((model.missing_action == Impute)?
                            (std::string(", ") + std::to_string(hplane.fill_val[ix]) + std::string(")")) : (std::string("")));
    }

    cond_left = hplane_conds + std::string(" <= ") + std::to_string(hplane.split_point);
    cond_right = hplane_conds + std::string(" > ") + std::to_string(hplane.split_point);
}
