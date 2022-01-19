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
*     Copyright (c) 2019-2022, David Cortes
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

static inline bool is_terminal_node(const IsoTree &node)
{
    return node.tree_left == 0;
}

static inline bool is_terminal_node(const IsoHPlane &node)
{
    return node.hplane_left == 0;
}

template <class Tree>
void build_terminal_node_mappings_single_tree(std::vector<size_t> &mappings, size_t &n_terminal, const std::vector<Tree> &tree)
{
    mappings.resize(tree.size());
    mappings.shrink_to_fit();
    std::fill(mappings.begin(), mappings.end(), (size_t)0);
    
    n_terminal = 0;
    for (size_t node = 0; node < tree.size(); node++)
    {
        if (is_terminal_node(tree[node]))
        {
            mappings[node] = n_terminal;
            n_terminal++;
        }
    }
}

void build_terminal_node_mappings_single_tree(std::vector<size_t> &mappings, size_t &n_terminal, const std::vector<IsoTree> &tree)
{
    build_terminal_node_mappings_single_tree<IsoTree>(mappings, n_terminal, tree);
}

void build_terminal_node_mappings_single_tree(std::vector<size_t> &mappings, size_t &n_terminal, const std::vector<IsoHPlane> &tree)
{
    build_terminal_node_mappings_single_tree<IsoHPlane>(mappings, n_terminal, tree);
}

static inline const std::vector<IsoTree>& get_tree(const IsoForest &model, size_t tree)
{
    return model.trees[tree];
}

static inline const std::vector<IsoHPlane>& get_tree(const ExtIsoForest &model, size_t tree)
{
    return model.hplanes[tree];
}

template <class Model>
void build_terminal_node_mappings(TreesIndexer &indexer, const Model &model)
{
    indexer.indices.resize(get_ntrees(model));
    indexer.indices.shrink_to_fit();

    if (!indexer.indices.empty() && !indexer.indices.front().reference_points.empty())
    {
        for (auto &ind : indexer.indices)
        {
            ind.reference_points.clear();
            ind.reference_indptr.clear();
            ind.reference_mapping.clear();
        }
    }

    for (size_t tree = 0; tree < indexer.indices.size(); tree++)
    {
        build_terminal_node_mappings_single_tree(indexer.indices[tree].terminal_node_mappings,
                                                 indexer.indices[tree].n_terminal,
                                                 get_tree(model, tree));
    }
}

static inline size_t get_idx_tree_left(const IsoTree &node)
{
    return node.tree_left;
}

static inline size_t get_idx_tree_left(const IsoHPlane &node)
{
    return node.hplane_left;
}

static inline size_t get_idx_tree_right(const IsoTree &node)
{
    return node.tree_right;
}

static inline size_t get_idx_tree_right(const IsoHPlane &node)
{
    return node.hplane_right;
}

template <class Node>
void build_dindex_recursive
(
    const size_t curr_node,
    const size_t n_terminal, const size_t ncomb,
    const size_t st, const size_t end,
    std::vector<size_t> &restrict node_indices, /* array with all terminal indices in 'tree' */
    const std::vector<size_t> &restrict node_mappings, /* tree_index : terminal_index */
    std::vector<double> &restrict node_distances, /* indexed by terminal_index */
    std::vector<double> &restrict node_depths, /* indexed by terminal_index */
    size_t curr_depth,
    const std::vector<Node> &tree
)
{
    if (end > st)
    {
        size_t i, j;
        for (size_t el1 = st; el1 < end; el1++)
        {
            for (size_t el2 = el1 + 1; el2 <= end; el2++)
            {
                i = node_mappings[node_indices[el1]];
                j = node_mappings[node_indices[el2]];
                node_distances[ix_comb(i, j, n_terminal, ncomb)]++;
            }
        }
    }

    if (!is_terminal_node(tree[curr_node]))
    {
        const size_t delim = get_idx_tree_right(tree[curr_node]);
        size_t frontier = st;
        size_t temp;
        for (size_t ix = st; ix <= end; ix++)
        {
           if (node_indices[ix] < delim)
            {
                temp = node_indices[frontier];
                node_indices[frontier] = node_indices[ix];
                node_indices[ix] = temp;
                frontier++;
            }
        }

        if (unlikely(frontier == st)) unexpected_error();

        curr_depth++;
        build_dindex_recursive<Node>(get_idx_tree_left(tree[curr_node]),
                                     n_terminal, ncomb,
                                     st, frontier-1,
                                     node_indices,
                                     node_mappings,
                                     node_distances,
                                     node_depths,
                                     curr_depth,
                                     tree);
        build_dindex_recursive<Node>(get_idx_tree_right(tree[curr_node]),
                                     n_terminal, ncomb,
                                     frontier, end,
                                     node_indices,
                                     node_mappings,
                                     node_distances,
                                     node_depths,
                                     curr_depth,
                                     tree);
    }

    else
    {
        node_depths[node_mappings[curr_node]] = curr_depth;
    }
}

template <class Node>
void build_dindex
(
    std::vector<size_t> &restrict node_indices, /* empty, but correctly sized */
    const std::vector<size_t> &restrict node_mappings, /* tree_index : terminal_index */
    std::vector<double> &restrict node_distances, /* indexed by terminal_index */
    std::vector<double> &restrict node_depths, /* indexed by terminal_index */
    const size_t n_terminal,
    const std::vector<Node> &tree
)
{
    if (tree.size() <= 1) return;

    std::fill(node_distances.begin(), node_distances.end(), 0.);

    node_indices.clear();
    for (size_t node = 0; node < tree.size(); node++)
    {
        if (is_terminal_node(tree[node]))
            node_indices.push_back(node);
    }

    node_depths.resize(n_terminal);

    build_dindex_recursive<Node>(
        (size_t)0,
        node_indices.size(), calc_ncomb(node_indices.size()),
        0, node_indices.size()-1,
        node_indices,
        node_mappings,
        node_distances,
        node_depths,
        (size_t)0,
        tree
    );
}

void build_dindex
(
    std::vector<size_t> &restrict node_indices, /* empty, but correctly sized */
    const std::vector<size_t> &restrict node_mappings, /* tree_index : terminal_index */
    std::vector<double> &restrict node_distances, /* indexed by terminal_index */
    std::vector<double> &restrict node_depths, /* indexed by terminal_index */
    const size_t n_terminal,
    const std::vector<IsoTree> &tree
)
{
    build_dindex<IsoTree>(
        node_indices,
        node_mappings,
        node_distances,
        node_depths,
        n_terminal,
        tree
    );
}

void build_dindex
(
    std::vector<size_t> &restrict node_indices, /* empty, but correctly sized */
    const std::vector<size_t> &restrict node_mappings, /* tree_index : terminal_index */
    std::vector<double> &restrict node_distances, /* indexed by terminal_index */
    std::vector<double> &restrict node_depths, /* indexed by terminal_index */
    const size_t n_terminal,
    const std::vector<IsoHPlane> &tree
)
{
    build_dindex<IsoHPlane>(
        node_indices,
        node_mappings,
        node_distances,
        node_depths,
        n_terminal,
        tree
    );
}

template <class Model>
void build_distance_mappings(TreesIndexer &indexer, const Model &model, int nthreads)
{
    build_terminal_node_mappings(indexer, model);
    SignalSwitcher ss = SignalSwitcher();

    size_t ntrees = get_ntrees(model);
    std::vector<size_t> n_terminal(ntrees);
    for (size_t tree = 0; tree < ntrees; tree++)
        n_terminal[tree] = indexer.indices[tree].n_terminal;

    size_t max_n_terminal = *std::max_element(n_terminal.begin(), n_terminal.end());
    check_interrupt_switch(ss);
    if (max_n_terminal <= 1) return;

    #ifndef _OPENMP
    nthreads = 1;
    #endif
    std::vector<std::vector<size_t>> thread_buffer_indices(nthreads);
    for (std::vector<size_t> &v : thread_buffer_indices)
        v.reserve(max_n_terminal);
    check_interrupt_switch(ss);

    
    
    bool threw_exception = false;
    std::exception_ptr ex = NULL;
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(indexer, model, n_terminal, threw_exception, ex)
    for (size_t_for tree = 0; tree < (decltype(tree))ntrees; tree++)
    {
        if (interrupt_switch || threw_exception) continue;

        try
        {
            size_t n_terminal_this = n_terminal[tree];
            size_t ncomb = calc_ncomb(n_terminal_this);
            indexer.indices[tree].node_distances.assign(ncomb, 0.);
            indexer.indices[tree].node_distances.shrink_to_fit();
            build_dindex(
                thread_buffer_indices[omp_get_thread_num()],
                indexer.indices[tree].terminal_node_mappings,
                indexer.indices[tree].node_distances,
                indexer.indices[tree].node_depths,
                n_terminal_this,
                get_tree(model, tree)
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

    if (interrupt_switch || threw_exception)
    {
        indexer.indices.clear();
    }

    check_interrupt_switch(ss);
    if (threw_exception) std::rethrow_exception(ex);
}

template <class Model>
void build_tree_indices(TreesIndexer &indexer, const Model &model, int nthreads, const bool with_distances)
{
    if (!indexer.indices.empty() && !indexer.indices.front().reference_points.empty())
    {
        for (auto &ind : indexer.indices)
        {
            ind.reference_points.clear();
            ind.reference_indptr.clear();
            ind.reference_mapping.clear();
        }
    }

    
    try
    {
        if (with_distances) {
            build_distance_mappings(indexer, model, nthreads);
        }

        else {
            if (!indexer.indices.empty() && !indexer.indices.front().node_distances.empty())
            {
                for (auto &ind : indexer.indices)
                {
                    ind.node_distances.clear();
                    ind.node_depths.clear();
                }
            }

            build_terminal_node_mappings(indexer, model);
        }
    }

    catch (...)
    {
        indexer.indices.clear();
        throw;
    }
}

void build_tree_indices(TreesIndexer &indexer, const IsoForest &model, int nthreads, const bool with_distances)
{
    if (model.trees.empty())
        throw std::runtime_error("Cannot build indexed for unfitted model.\n");
    if (model.missing_action == Divide)
        throw std::runtime_error("Cannot build tree indexer with 'missing_action=Divide'.\n");
    if (model.new_cat_action == Weighted && model.cat_split_type == SubSet)
    {
        for (const std::vector<IsoTree> &tree : model.trees)
        {
            for (const IsoTree &node : tree)
            {
                if (!is_terminal_node(node) && node.col_type == Categorical)
                    throw std::runtime_error("Cannot build tree indexer with 'new_cat_action=Weighted'.\n");
            }
        }
    }
    
    build_tree_indices<IsoForest>(indexer, model, nthreads, with_distances);
}

void build_tree_indices(TreesIndexer &indexer, const ExtIsoForest &model, int nthreads, const bool with_distances)
{
    if (model.hplanes.empty())
        throw std::runtime_error("Cannot build indexed for unfitted model.\n");
    build_tree_indices<ExtIsoForest>(indexer, model, nthreads, with_distances);
}

/* Build indexer for faster terminal node predictions and/or distance calculations
* 
* Parameters
* ==========
* - indexer
*       Pointer or reference to an indexer object which will be associated to a fitted model and in
*       which indices for terminal nodes and potentially node distances will be stored.
* - model / model_outputs / model_outputs_ext
*       Pointer or reference to a fitted model object for which an indexer will be built.
* - nthreads
*       Number of parallel threads to use. This operation will only be multi-threaded when passing
*       'with_distances=true'.
* - with_distances
*       Whether to also pre-calculate node distances in order to speed up 'calc_similarity' (distances).
*       Note that this will consume a lot more memory and make the resulting object significantly
*       heavier.
*/
void build_tree_indices
(
    TreesIndexer *indexer,
    const IsoForest *model_outputs,
    const ExtIsoForest *model_outputs_ext,
    int nthreads,
    const bool with_distances
)
{
    if (model_outputs != NULL)
        build_tree_indices(*indexer, *model_outputs, nthreads, with_distances);
    else
        build_tree_indices(*indexer, *model_outputs_ext, nthreads, with_distances);
}

/* Gets the number of reference points stored in an indexer object */
size_t get_number_of_reference_points(const TreesIndexer &indexer) noexcept
{
    if (indexer.indices.empty()) return 0;
    return indexer.indices.front().reference_points.size();
}

/* This assumes it already has the indexer and 'reference_points' were just added.
   It builds up 'reference_mapping' and 'reference_indptr' from it. */
void build_ref_node(SingleTreeIndex &node)
{
    node.reference_mapping.resize(node.reference_points.size());
    node.reference_mapping.shrink_to_fit();
    std::iota(node.reference_mapping.begin(), node.reference_mapping.end(), (size_t)0);
    std::sort(node.reference_mapping.begin(), node.reference_mapping.end(),
              [&node](const size_t a, const size_t b)
              {return node.reference_points[a] < node.reference_points[b];});

    size_t n_terminal = node.n_terminal;
    node.reference_indptr.assign(n_terminal+1, (size_t)0);
    node.reference_indptr.shrink_to_fit();

    std::vector<size_t>::iterator curr_begin = node.reference_mapping.begin();
    std::vector<size_t>::iterator new_begin;
    size_t curr_node;
    while (curr_begin != node.reference_mapping.end())
    {
        curr_node = node.reference_points[*curr_begin];
        new_begin = std::upper_bound(curr_begin, node.reference_mapping.end(), curr_node,
                                     [&node](const size_t a, const size_t b)
                                     {return a < node.reference_points[b];});
        node.reference_indptr[curr_node+1] = std::distance(curr_begin, new_begin);
        curr_begin = new_begin;
    }

    for (size_t ix = 1; ix < n_terminal; ix++)
        node.reference_indptr[ix+1] += node.reference_indptr[ix];
}
