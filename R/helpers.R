check.pos.int <- function(var) {
    if (NROW(var) != 1L || var < 1) {
        stop(paste0("'", as.character(substitute(var)), "' must be a positive integer."))
    }
}

check.max.depth <- function(max_depth) {
    if (!is.null(max_depth)) {
        if (NROW(max_depth) != 1L || max_depth < 0) {
            stop(paste0("'max_depth' must be a non-negative integer."))
        }
        return(as.integer(max_depth))
    } else {
        return(0L)
    }
}

check.str.option <- function(option, allowed) {
    if (NROW(option) != 1 || !(option %in% allowed)) {
        stop(paste0("'", as.character(substitute(option)), "' must be one of '", paste(allowed, collapse = "', '"), "'."))
    }
}

check.is.prob <- function(prob) {
    if (NROW(prob) != 1 || prob < 0 || prob > 1) {
        stop(paste0("'", as.character(substitute(prob)), "' must be a number between zero and one."))
    }
}

check.is.bool <- function(var) {
    if (NROW(var) != 1) stop(paste0("'", as.character(substitute(var)), "' must be logical (boolean)."))
}

check.nthreads <- function(nthreads) {
    if (NROW(nthreads) != 1) stop("'nthreads' must be a positive integer.")
    if (is.null(nthreads)) {
        nthreads <- 1L
    } else if (is.na(nthreads)) {
        nthreads <- 1L
    }  else if (nthreads < 1) {
        nthreads <- 1L
    }
    return(as.integer(nthreads))
}

check.categ.cols <- function(categ_cols, data) {
    if (is.null(categ_cols) || !NROW(categ_cols))
        return(NULL)
    categ_cols_i <- as.integer(categ_cols)
    if (anyNA(categ_cols_i) && NROW(colnames(data))) {
        idx <- match(categ_cols, colnames(data))
        if (anyNA(idx))
            stop("'categ_cols' contains invalid columns.")
        categ_cols <- idx
    } else {
        categ_cols <- categ_cols_i
    }
    if (any(categ_cols < 1))
        stop("'categ_cols' contains invalid column indices.")
    if (anyDuplicated(categ_cols))
        stop("'categ_cols' contains duplicted entries.")
    categ_cols <- sort.int(categ_cols)
    return(categ_cols)
}

check.is.1d <- function(var) {
    if (NCOL(var) > 1) {
        stop(paste0("'", as.character(substitute(var)), "' must be a 1-d numeric vector."))
    }
}

coerce.null <- function(x, repl) {
    if (is.null(x)) {
        return(repl)
    } else {
        return(x)
    }
}

set.list.elt <- function(lst, el_name, val) {
    if (el_name %in% names(lst)) {
        ix <- which(names(lst) == el_name) - 1L
        modify_R_list_inplace(lst, ix, val)
    } else {
        addto_R_list_inplace(lst, el_name, val)
    }
}

cast.df.alike <- function(df) {
    if (inherits(df, c("data.table", "tibble")))
        df <- as.data.frame(df)
    return(df)
}

get.types.dmat <- function() {
    return(c("matrix"))
}

get.types.spmat <- function(allow_csr = FALSE, allow_csc = TRUE, allow_vec = FALSE) {
    outp <- character()
    if (allow_csc) outp <- c(outp, "dgCMatrix", "matrix.csc")
    if (allow_csr) outp <- c(outp, "dgRMatrix", "matrix.csr")
    if (allow_vec && allow_csr) outp <- c(outp, "dsparseVector")
    return(outp)
}

cast.df.col.to.num <- function(cl) {
    if (inherits(cl, "factor"))
        cl <- as.character(cl)
    return(as.numeric(cl))
}

encode.factor <- function(cl, levs) {
    if (NROW(cl) >= 100 && is.factor(cl)) {
        if (length(levels(cl)) == length(levs) && all(levels(cl) == levs)) {
            return(cl)
        }
    }
    return(factor(cl, levs))
}

process.data <- function(data, sample_weights = NULL, column_weights = NULL, recode_categ = TRUE, categ_cols = NULL) {
    data  <-  cast.df.alike(data)
    dmatrix_types     <-  get.types.dmat()
    spmatrix_types    <-  get.types.spmat()
    supported_dtypes  <-  c("data.frame", dmatrix_types, spmatrix_types)
    if (!inherits(data, supported_dtypes))
        stop(paste0("Invalid input data. Supported types are: ", paste(supported_dtypes, collapse = ", ")))
    
    if (NROW(data) < 2L) stop("Input data must have at least 2 rows.")
    
    if (!is.null(sample_weights))  sample_weights  <- as.numeric(sample_weights)
    if (!is.null(column_weights))  column_weights  <- as.numeric(column_weights)
    if (NROW(sample_weights)  && NROW(data) != NROW(sample_weights))
        stop(sprintf("'sample_weights' has different number of rows than 'data' (%d vs. %d).",
                     NROW(data), NROW(sample_weights)))
    if (NROW(column_weights)  && NCOL(data) != NROW(column_weights))
        stop(sprintf("'column_weights' has different dimension than number of columns in 'data' (%d vs. %d).",
                     NCOL(data), NROW(column_weights)))

    if (!is.null(categ_cols) && is.data.frame(data)) {
        warning("'categ_cols' is ignored when passing a data.frame as input.")
        categ_cols <- NULL
    }

    if (ncol(data) < 1L)
        stop("'data' has no columns.")
    
    outp <- list(X_num      =  numeric(),
                 X_cat      =  integer(),
                 ncat       =  integer(),
                 cols_num   =  c(),
                 cols_cat   =  c(),
                 cat_levs   =  c(),
                 Xc         =  numeric(),
                 Xc_ind     =  integer(),
                 Xc_indptr  =  integer(),
                 nrows      =  as.integer(NROW(data)),
                 ncols_num  =  as.integer(NCOL(data)),
                 ncols_cat  =  as.integer(0L),
                 categ_cols =  NULL,
                 categ_max  =  integer(),
                 sample_weights  =  unname(as.numeric(sample_weights)),
                 column_weights  =  unname(as.numeric(column_weights))
                 )

    avoid_sparse_sort <- FALSE

    if (NROW(categ_cols)) {
        cols_num   <-  setdiff(1L:ncol(data), categ_cols)
        if (inherits(data, c("data.frame", "matrix", "dgCMatrix"))) {
            X_cat  <-  data[, categ_cols, drop=FALSE]
            data   <-  data[, cols_num,   drop=FALSE]
        } else if (inherits(data, "matrix.csc")) {
            nrows    <- nrow(data)
            data@ja  <- data@ja - 1L
            data@ia  <- data@ia - 1L
            data@ra  <- deepcopy_vector(data@ra)
            avoid_sparse_sort <- TRUE
            call_sort_csc_indices(data@ra, data@ja, data@ia)
            X_cat  <- call_take_cols_by_index_csc(data@ra,
                                                  data@ja,
                                                  data@ia,
                                                  categ_cols - 1L,
                                                  TRUE, nrows)
            X_cat  <- X_cat[["X_cat"]]
            dt_new <- call_take_cols_by_index_csc(data@ra,
                                                  data@ja,
                                                  data@ia,
                                                  cols_num - 1L,
                                                  FALSE, nrows)
            data@ra  <- dt_new[["Xc"]]
            data@ja  <- dt_new[["Xc_ind"]] + 1L
            data@ia  <- dt_new[["Xc_indptr"]] + 1L
            data@dimension <- as.integer(c(nrows, NROW(cols_num)))
        } else {
            X_cat  <-  data[, categ_cols]
            data   <-  data[, cols_num]
        }
        ncols_cat  <-  ncol(X_cat)
        categ_max  <-  as.integer(unname(apply(X_cat, 2, max, na.rm=TRUE)))
        if (inherits(X_cat, "sparseMatrix"))
            X_cat  <-  as.matrix(X_cat)
        X_cat      <-  as.integer(X_cat)
        if (anyNA(X_cat))
            X_cat[is.na(X_cat)] <- -1L
        
        outp$X_cat       <-  X_cat
        outp$categ_cols  <-  categ_cols
        outp$categ_max   <-  categ_max
        outp$ncat        <-  pmax(categ_max + 1L, integer(length(categ_max)))
        outp$cols_num    <-  cols_num
        outp$ncols_num   <-  ncol(data)
        outp$ncols_cat   <-  ncols_cat

        if (!ncol(data))
            return(outp)
    }
    
    ### Dense matrix
    if ( inherits(data, dmatrix_types) ) {
        outp$X_num      <-  unname(as.numeric(data))
        outp$ncols_num  <-  ncol(data)
        return(outp)
    }
    
    ### Sparse matrix
    if ( inherits(data, spmatrix_types) ) {
        
        if (inherits(data, "dgCMatrix")) {
            ### From package 'Matrix'
            if (!NROW(data@x))
                stop("'data' has no non-zero entries.")
            outp$Xc         <-  data@x
            outp$Xc_ind     <-  data@i
            outp$Xc_indptr  <-  data@p
        } else {
            ### From package 'SparseM'
            if (!NROW(data@ra))
                stop("'data' has no non-zero entries.")
            outp$Xc         <-  data@ra
            outp$Xc_ind     <-  data@ja - 1L
            outp$Xc_indptr  <-  data@ia - 1L
        }
        if (!avoid_sparse_sort) {
            if (!inherits(data, "dgCMatrix"))
                outp$Xc     <- deepcopy_vector(outp$Xc)
            call_sort_csc_indices(outp$Xc, outp$Xc_ind, outp$Xc_indptr)
        }
        outp$ncols_num      <-  ncol(data)
        
        return(outp)
    }
    
    ### Data Frame
    if ( "data.frame" %in% class(data) ) {
        dtypes_num  <-  c("numeric",   "integer",  "Date",  "POSIXct")
        dtypes_cat  <-  c("character", "factor",   "logical")
        supported_col_types <- c(dtypes_num, dtypes_cat)
        dt_coltypes <- Reduce(c, sapply(data, class))
        if (any(!(dt_coltypes %in% c(supported_col_types, "POSIXt")))) {
            stop(paste0("Input data contains unsupported column types. Supported types are ",
                        paste(supported_col_types, collapse = ", "), " - got the following: ",
                        paste(unique(dt_coltypes[!(dt_coltypes %in% supported_col_types)]), collapse = ", ")))
        
        }
        
        if (any(dt_coltypes %in% dtypes_num)) {
            is_num          <-  unname(as.logical(sapply(data, function(x) any(class(x) %in% dtypes_num))))
            outp$cols_num   <-  names(data)[is_num]
            outp$ncols_num  <-  as.integer(sum(is_num))
            outp$X_num      <-  unname(as.numeric(as.matrix(as.data.frame(lapply(data[, is_num, drop = FALSE], cast.df.col.to.num)))))
        } else { outp$ncols_num <- as.integer(0) }
        
        if (any(dt_coltypes %in% dtypes_cat)) {
            if (any("ordered" %in% dt_coltypes))
                warning("Data contains ordered factors. These are treated as unordered.")
            is_cat          <-  unname(as.logical(sapply(data, function(x) any(class(x) %in% dtypes_cat))))
            outp$cols_cat   <-  names(data)[is_cat]
            outp$ncols_cat  <-  as.integer(sum(is_cat))
            if (recode_categ) {
                outp$X_cat  <-  as.data.frame(lapply(data[, is_cat, drop = FALSE], factor))
            } else {
                outp$X_cat  <-  as.data.frame(lapply(data[, is_cat, drop = FALSE],
                                                     function(x) if (is.factor(x)) x else factor(x)))
            }
            outp$cat_levs   <-  lapply(outp$X_cat, levels)
            outp$ncat       <-  sapply(outp$cat_levs, NROW)
            outp$X_cat      <-  as.data.frame(lapply(outp$X_cat, function(x) ifelse(is.na(x), -1L, as.integer(x) - 1L)))
            outp$X_cat      <-  unname(as.integer(as.matrix(outp$X_cat)))
        }
        
        if (NROW(outp$cols_num) && NROW(outp$cols_cat) && NROW(outp$column_weights)) {
            outp$column_weights <- c(outp$column_weights[names(data) %in% outp$cols_num],
                                     outp$column_weights[names(data) %in% outp$cols_cat])
        }
        
        return(outp)
    }
    
    stop("Unexpected error.")
}

process.data.new <- function(data, metadata, allow_csr = FALSE, allow_csc = TRUE,
                             enforce_shape = FALSE, mix_new_categ_and_missing = FALSE) {
    if (!NROW(data)) stop("'data' contains zero rows.")
    if (inherits(data, "sparseVector") && !inherits(data, "dsparseVector"))
        stop("Sparse vectors only allowed as 'dsparseVector' class.")
    if (!inherits(data, "sparseVector")) {
        if ( NCOL(data) < (metadata$ncols_num + metadata$ncols_cat) )
            stop(sprintf("Input data contains fewer columns than expected (%d vs. %d)",
                         NCOL(data), (metadata$ncols_num + metadata$ncols_cat)))
    } else {
        if (data@length < (metadata$ncols_num + metadata$ncols_cat))
            stop(sprintf("Input data contains different columns than expected (%d vs. %d)",
                         data@length, (metadata$ncols_num + metadata$ncols_cat)))
    }
    data  <-  cast.df.alike(data)
    if (metadata$ncols_cat > 0L && !NROW(metadata$categ_cols) && !inherits(data, "data.frame"))
        stop("Model was fit to data.frame with categorical data, must pass a data.frame with new data.")
    
    dmatrix_types     <-  get.types.dmat()
    spmatrix_types    <-  get.types.spmat(allow_csr = allow_csr, allow_csc = allow_csc, TRUE)
    supported_dtypes  <-  c("data.frame", dmatrix_types, spmatrix_types)

    if (!inherits(data, supported_dtypes))
        stop(paste0("Invalid input data. Supported types are: ", paste(supported_dtypes, collapse = ", ")))

    if (!allow_csr && inherits(data, c("RsparseMatrix", "matrix.csr")))
        stop("CSR matrix not supported for this prediction type. Try converting to CSC.")
    if (!allow_csc && inherits(data, c("CsparseMatrix", "matrix.csc")))
        stop("CSC matrix not supported for this prediction type. Try converting to CSR.")

    outp <- list(
        X_num      =  numeric(),
        X_cat      =  integer(),
        nrows      =  as.integer(NROW(data)),
        Xc         =  numeric(),
        Xc_ind     =  integer(),
        Xc_indptr  =  integer(),
        Xr         =  numeric(),
        Xr_ind     =  integer(),
        Xr_indptr  =  integer(),
        cat_levs   =  list()
    )

    avoid_sparse_sort <- FALSE

    if (!NROW(metadata$categ_cols)) {
        
        if (((!NROW(metadata$cols_num) && !NROW(metadata$cols_cat)) || !inherits(data, "data.frame")) &&
            (   (inherits(data, "sparseVector") && data@length > metadata$ncols_num) ||
                (!inherits(data, "sparseVector") && (ncol(data) > metadata$ncols_num)))
            && (enforce_shape || inherits(data, c("RsparseMatrix", "matrix.csr")))
            ) {

            if (inherits(data, c("matrix", "CsparseMatrix")) ||
                (!NROW(metadata$cols_num) && inherits(data, "data.frame"))) {
                data   <- data[, 1L:metadata$ncols_num, drop=FALSE]
            } else if (inherits(data, "sparseVector")) {
                data   <- data[1L:metadata$ncols_num]
            } else if (inherits(data, "RsparseMatrix")) {
                nrows  <- nrow(data)
                avoid_sparse_sort <- TRUE
                call_sort_csc_indices(data@x, data@j, data@p)
                dt_new <- call_take_cols_by_slice_csr(
                                data@x,
                                data@j,
                                data@p,
                                metadata$ncols_num,
                                FALSE
                            )
                data@x <- dt_new[["Xr"]]
                data@j <- dt_new[["Xr_ind"]]
                data@p <- dt_new[["Xr_indptr"]]
                data@Dim <- as.integer(c(nrows, metadata$ncols_num))
            } else if (inherits(data, "matrix.csr")) {
                avoid_sparse_sort <- TRUE
                data@ja <- data@ja - 1L
                data@ia <- data@ia - 1L
                data@ra <- deepcopy_vector(data@ra)
                call_sort_csc_indices(data@ra, data@ja, data@ia)
                dt_new <- call_take_cols_by_slice_csr(
                                data@ra,
                                data@ja,
                                data@ia,
                                metadata$ncols_num,
                                FALSE
                            )
                data@ra <- dt_new[["Xr"]]
                data@ja <- dt_new[["Xr_ind"]] + 1L
                data@ia <- dt_new[["Xr_indptr"]] + 1L
                data@dimension <- as.integer(c(nrows, metadata$ncols_num))
            } else if (inherits(data, "matrix.csc")) {
                data@ia <- data@ia - 1L
                nrows   <- nrow(data)
                dt_new  <- call_take_cols_by_slice_csc(
                                data@ra,
                                data@ja,
                                data@ia,
                                metadata$ncols_num,
                                FALSE, nrows
                            )
                data@ra <- dt_new[["Xc"]]
                data@ja <- dt_new[["Xc_ind"]]
                data@ia <- dt_new[["Xc_indptr"]] + 1L
                data@dimension <- as.integer(c(nrows, metadata$ncols_num))
            } else if (!inherits(data, "data.frame")) {
                data <- data[, 1L:metadata$ncols_num]
            }

        }

    } else { ### has metadata$categ_cols

        if (!inherits(data, "sparseVector")) {

            nrows <- nrow(data)
            if (inherits(data, c("matrix", "data.frame", "dgCMatrix"))) {
                X_cat  <- data[, metadata$categ_cols,  drop=FALSE]
                data   <- data[, metadata$cols_num,    drop=FALSE]
            } else if (inherits(data, "dgRMatrix")) {
                avoid_sparse_sort <- TRUE
                call_sort_csc_indices(data@x, data@j, data@p)
                X_cat  <- call_take_cols_by_index_csr(data@x,
                                                      data@j,
                                                      data@p,
                                                      metadata$categ_cols - 1L,
                                                      TRUE)
                X_cat  <- X_cat[["X_cat"]]
                dt_new <- call_take_cols_by_index_csr(data@x,
                                                      data@j,
                                                      data@p,
                                                      metadata$cols_num - 1L,
                                                      FALSE)
                data@x   <- dt_new[["Xr"]]
                data@j   <- dt_new[["Xr_ind"]]
                data@p   <- dt_new[["Xr_indptr"]]
                data@Dim <- as.integer(c(nrows, NROW(metadata$cols_num)))
            } else if (inherits(data, "matrix.csc")) {
                avoid_sparse_sort <- TRUE
                data@ja  <- data@ja - 1L
                data@ia  <- data@ia - 1L
                data@ra  <- deepcopy_vector(data@ra)
                call_sort_csc_indices(data@ra, data@ja, data@ia)

                X_cat  <- call_take_cols_by_index_csc(data@ra,
                                                      data@ja,
                                                      data@ia,
                                                      metadata$categ_cols - 1L,
                                                      TRUE, nrows)
                X_cat  <- X_cat[["X_cat"]]
                dt_new <- call_take_cols_by_index_csc(data@ra,
                                                      data@ja,
                                                      data@ia,
                                                      metadata$cols_num - 1L,
                                                      FALSE, nrows)
                data@ra  <- dt_new[["Xc"]]
                data@ja  <- dt_new[["Xc_ind"]] + 1L
                data@ia  <- dt_new[["Xc_indptr"]] + 1L
                data@dimension <- as.integer(c(nrows, NROW(metadata$cols_num)))
            } else if (inherits(data, "matrix.csr")) {
                avoid_sparse_sort <- TRUE
                data@ja  <- data@ja - 1L
                data@ia  <- data@ia - 1L
                data@ra  <- deepcopy_vector(data@ra)
                call_sort_csc_indices(data@ra, data@ja, data@ia)

                X_cat  <- call_take_cols_by_index_csr(data@ra,
                                                      data@ja,
                                                      data@ia,
                                                      metadata$categ_cols - 1L,
                                                      TRUE)
                X_cat  <- X_cat[["X_cat"]]
                dt_new <- call_take_cols_by_index_csr(data@ra,
                                                      data@ja,
                                                      data@ia,
                                                      metadata$cols_num - 1L,
                                                      FALSE)
                data@ra  <- dt_new[["Xr"]]
                data@ja  <- dt_new[["Xr_ind"]] + 1L
                data@ia  <- dt_new[["Xr_indptr"]] + 1L
                data@dimension <- as.integer(c(nrows, NROW(metadata$cols_num)))
            } else {
                X_cat  <- data[, metadata$categ_cols]
                data   <- data[, metadata$cols_num]
            }

        } else { ### sparseVector
            X_cat <- matrix(data[metadata$categ_cols], nrow=1L)
            nrows <- 1L
            data  <- data[metadata$cols_num]
        }

        if (!inherits(X_cat, "matrix"))
            X_cat <- as.matrix(X_cat)
        X_cat <- as.integer(X_cat)
        if (anyNA(X_cat))
            X_cat[is.na(X_cat)] <- -1L
        outp$X_cat <- X_cat
        outp$nrows <- nrows

    }

    if (inherits(data, "data.frame") &&
        (NROW(metadata$categ_cols) ||
        (!NROW(metadata$cols_num) && !NROW(metadata$cols_cat)))
    ) {
        data <- as.data.frame(lapply(data, cast.df.col.to.num))
        data <- as.matrix(data)
    }

    
    if (inherits(data, "data.frame")) {
        
        if (NROW(setdiff(c(metadata$cols_num, metadata$cols_cat), names(data)))) {
            missing_cols <- setdiff(c(metadata$cols_num, metadata$cols_cat), names(data))
            stop(paste0(sprintf("Input data is missing %d columns - head: ", NROW(missing_cols)),
                        paste(head(missing_cols, 3), collapse = ", ")))
        }
        
        if (!NROW(metadata$cols_num) && !NROW(metadata$cols_cat)) {
            
            if (NCOL(data) != metadata$ncols_num)
                stop(sprintf("Input data has %d columns, but model was fit to data with %d columns.",
                             NCOL(data), (metadata$ncols_num + metadata$ncols_cat)))
            outp$X_num <- unname(as.numeric(as.matrix(as.data.frame(lapply(data, cast.df.col.to.num)))))
            
        } else {
            
            if (metadata$ncols_num > 0L) {
                outp$X_num <- unname(as.numeric(as.matrix(as.data.frame(lapply(data[, metadata$cols_num, drop = FALSE], cast.df.col.to.num)))))
            }
            
            if (metadata$ncols_cat > 0L) {
                outp$X_cat <- data[, metadata$cols_cat, drop = FALSE]
                if (mix_new_categ_and_missing) {
                    new_cat_levels <- metadata$cat_levs
                } else {
                    outp$X_cat     <- lapply(outp$X_cat, factor)
                    new_cat_levels <- lapply(outp$X_cat, levels)
                    new_cat_levels <- mapply(function(old_levs, new_levs) c(old_levs, setdiff(new_levs, old_levs)),
                                             metadata$cat_levs, new_cat_levels,
                                             SIMPLIFY = FALSE, USE.NAMES = TRUE)
                    outp$cat_levs  <- new_cat_levels
                }
                outp$X_cat <- as.data.frame(mapply(encode.factor,
                                                   outp$X_cat, new_cat_levels,
                                                   SIMPLIFY = FALSE, USE.NAMES = FALSE))
                outp$X_cat <- as.data.frame(lapply(outp$X_cat, function(x) ifelse(is.na(x), -1L, as.integer(x) - 1L)))
                outp$X_cat <- unname(as.integer(as.matrix(outp$X_cat)))
            }
            
        }
        
    } else if (inherits(data, "dsparseVector")) {
        if (allow_csr) {
            data@x   <-  data@x[order(data@i)]
            data@i   <-  data@i[order(data@i)]

            outp$Xr         <-  as.numeric(data@x)
            outp$Xr_ind     <-  as.integer(data@i - 1L)
            outp$Xr_indptr  <-  as.integer(c(0L, NROW(data@x)))
        } else {
            outp$X_num      <-  as.numeric(data)
        }
        outp$nrows          <-  1L
    } else {
        
        if (is.numeric(data) && is.null(dim(data)))
            data <- matrix(data, nrow = 1)
        
        if (NCOL(data) < metadata$ncols_num)
            stop(sprintf("Input data has %d numeric columns, but model was fit to data with %d numeric columns.",
                         NCOL(data), metadata$ncols_num))
        if (!inherits(data, spmatrix_types)) {
            outp$X_num <- as.numeric(data)
        } else {

            if (inherits(data, "dgCMatrix")) {
                ### From package 'Matrix'
                outp$Xc         <-  data@x
                outp$Xc_ind     <-  data@i
                outp$Xc_indptr  <-  data@p
                if (!avoid_sparse_sort)
                    call_sort_csc_indices(outp$Xc, outp$Xc_ind, outp$Xc_indptr)
            } else if (inherits(data, "dgRMatrix")) {
                ### From package 'Matrix'
                outp$Xr         <-  data@x
                outp$Xr_ind     <-  data@j
                outp$Xr_indptr  <-  data@p
                if (!avoid_sparse_sort)
                    call_sort_csc_indices(outp$Xr, outp$Xr_ind, outp$Xr_indptr)
            } else if (inherits(data, "matrix.csc")) {
                ### From package 'SparseM'
                outp$Xc         <-  data@ra
                outp$Xc_ind     <-  data@ja - 1L
                outp$Xc_indptr  <-  data@ia - 1L
                if (!avoid_sparse_sort) {
                    outp$Xc     <-  deepcopy_vector(outp$Xc)
                    call_sort_csc_indices(outp$Xc, outp$Xc_ind, outp$Xc_indptr)
                }
            } else if (inherits(data, "matrix.csr")) {
                ### From package 'SparseM'
                outp$Xr         <-  data@ra
                outp$Xr_ind     <-  data@ja - 1L
                outp$Xr_indptr  <-  data@ia - 1L
                if (!avoid_sparse_sort) {
                    outp$Xr     <-  deepcopy_vector(outp$Xr)
                    call_sort_csc_indices(outp$Xr, outp$Xr_ind, outp$Xr_indptr)
                }
            } else {
                stop("Invalid input type.")
            }
        }
        
    }
    
    return(outp)
}

reconstruct.from.imp <- function(imputed_num, imputed_cat, data, model_metadata, pdata) {

    if (NROW(imputed_cat))
        imputed_cat[imputed_cat < 0L] <- NA_integer_

    if (inherits(data, "RsparseMatrix")) {

        outp <- data
        if (!NROW(model_metadata$categ_cols) && ncol(data) == model_metadata$ncols_num) {
            outp@x <- imputed_num
        } else if (!NROW(model_metadata$categ_cols)) {
            outp@x <- deepcopy_vector(outp@x)
            call_reconstruct_csr_sliced(
                outp@x, outp@p,
                imputed_num, pdata$Xr_indptr,
                nrow(data)
            )
        } else {
            outp@x <- deepcopy_vector(outp@x)
            call_reconstruct_csr_with_categ(
                outp@x, outp@j, outp@p,
                imputed_num, pdata$Xr_ind, pdata$Xr_indptr,
                imputed_cat,
                model_metadata$cols_num-1L, model_metadata$categ_cols-1L,
                nrow(data), ncol(data)
            )
        }
        return(outp)

    } else if (inherits(data, "CsparseMatrix")) {

        outp       <-  data
        if (!NROW(model_metadata$categ_cols)) {
            outp@x <-  imputed_num
        } else {
            outp[, model_metadata$categ_cols] <- matrix(imputed_cat, nrow=nrow(data))
            copy_csc_cols_by_index(
                outp@x,
                outp@p,
                imputed_num,
                pdata$Xc_indptr,
                model_metadata$cols_num - 1L
            )
        }
        return(outp)

    } else if (inherits(data, "matrix.csr")) {
        
        outp <- data
        if (!NROW(model_metadata$categ_cols) && ncol(data) == model_metadata$ncols_num) {
            outp@ra <- imputed_num
        } else if (!NROW(model_metadata$categ_cols)) {
            outp@ra <- deepcopy_vector(outp@ra)
            call_reconstruct_csr_sliced(
                outp@ra, outp@ia-1L,
                imputed_num, pdata$Xr_indptr,
                nrow(data)
            )
        } else {
            outp@ra <- deepcopy_vector(outp@ra)
            call_reconstruct_csr_with_categ(
                outp@ra, outp@ja-1L, outp@ia-1L,
                imputed_num, pdata$Xr_ind, pdata$Xr_indptr,
                imputed_cat,
                model_metadata$cols_num-1L, model_metadata$categ_cols-1L,
                nrow(data), ncol(data)
            )
        }
        return(outp)

    } else if (inherits(data, "matrix.csc")) {
        
        outp <- data
        if (!NROW(model_metadata$categ_cols)) {
            outp@ra  <-  imputed_num
        } else {
            dt_new   <- assign_csc_cols(
                pdata$Xc,
                pdata$Xc_ind,
                pdata$Xc_indptr,
                imputed_cat,
                model_metadata$categ_cols - 1L,
                model_metadata$cols_num - 1L,
                nrow(data)
            )
            copy_csc_cols_by_index(
                dt_new$Xc,
                dt_new$Xc_indptr,
                imputed_num,
                pdata$Xc_indptr,
                model_metadata$cols_num - 1L
            )
            outp@ra <- dt_new$Xc
            outp@ja <- dt_new$Xc_ind + 1L
            outp@ia <- dt_new$Xc_indptr + 1L
            outp@dimension <- as.integer(c(nrow(data), length(dt_new$Xc_indptr)-1L))
        }
        return(outp)

    } else if (inherits(data, "sparseVector")) {

        if (!NROW(model_metadata$categ_cols) && data@length == model_metadata$ncols_num) {
            data@x <- imputed_num
        } else if (!NROW(model_metadata$categ_cols)) {
            data@x[1L:NROW(imputed_num)]    <- imputed_num
        } else {
            data[model_metadata$cols_num]   <- imputed_num
            data[model_metadata$categ_cols] <- imputed_cat
        }

    } else if (!inherits(data, "data.frame")) {

        if (!NROW(model_metadata$categ_cols) && (ncol(data) == model_metadata$ncols_num)) {
            return(matrix(imputed_num, nrow = NROW(data)))
        } else if (!NROW(model_metadata$categ_cols)) {
            data[, 1L:model_metadata$ncols_num]  <-  matrix(imputed_num, nrow = NROW(data))
            return(data)
        } else {
            data[, model_metadata$categ_cols]    <-  matrix(imputed_cat, nrow = NROW(data))
            if (model_metadata$ncols_num)
                data[, model_metadata$cols_num]  <-  matrix(imputed_num, nrow = NROW(data))
            return(data)
        }

    } else {
        
        dt_num <- as.data.frame(matrix(imputed_num, nrow = NROW(data)))
        dt_cat <- as.data.frame(matrix(imputed_cat, nrow = NROW(data)))
        if (!NROW(model_metadata$categ_cols)) {
            dt_cat <- as.data.frame(mapply(function(x, levs) factor(x, labels = levs),
                                           dt_cat + 1L, model_metadata$cat_levs,
                                           SIMPLIFY = FALSE))
        }

        if (NROW(model_metadata$categ_cols)) {
            data[, model_metadata$categ_cols]   <- dt_cat
            if (model_metadata$ncols_num)
                data[, model_metadata$cols_num] <- dt_num
        } else if (!NROW(model_metadata$cols_num)) {
            data[, 1L:model_metadata$ncols_num] <- dt_num
        } else {
            if (model_metadata$ncols_num)
                data[, model_metadata$cols_num] <- dt_num
            if (model_metadata$ncols_cat)
                data[, model_metadata$cols_cat] <- dt_cat
        }
        return(data)

    }
}

export.metadata <- function(model) {
    data_info <- list(
        ncols_numeric    =  model$metadata$ncols_num, ## is in c++
        ncols_categ      =  model$metadata$ncols_cat,  ## is in c++
        cols_numeric     =  as.list(model$metadata$cols_num),
        cols_categ       =  as.list(model$metadata$cols_cat),
        cat_levels       =  unname(as.list(model$metadata$cat_levs)),
        categ_cols       =  model$metadata$categ_cols,
        categ_max        =  model$metadata$categ_max,
        reference_names  =  model$metadata$reference_names
    )
    
    if (NROW(data_info$cat_levels)) {
        force.to.bool <- function(v) {
            if (NROW(v) == 2) {
                if (("TRUE" %in% v) && ("FALSE" %in% v))
                    v <- as.logical(v)
            }
            return(v)
        }
        data_info$cat_levels <- lapply(data_info$cat_levels, force.to.bool)
    }

    model_info <- list(
        ndim = model$params$ndim,
        nthreads = model$nthreads,
        build_imputer = model$params$build_imputer,
        use_long_double = coerce.null(model$use_long_double, FALSE)
    )
    
    params <- list(
        sample_size = model$params$sample_size,
        ntrees = model$params$ntrees,  ## is in c++
        ntry = model$params$ntry,
        max_depth = model$params$max_depth,
        ncols_per_tree = model$params$ncols_per_tree,
        prob_pick_avg_gain = model$params$prob_pick_avg_gain,
        prob_pick_pooled_gain = model$params$prob_pick_pooled_gain,
        prob_pick_full_gain = model$params$prob_pick_full_gain,
        prob_pick_dens = model$params$prob_pick_dens,
        prob_pick_col_by_range = model$params$prob_pick_col_by_range,
        min_gain = model$params$min_gain,
        missing_action = model$params$missing_action,  ## is in c++
        new_categ_action = model$params$new_categ_action,  ## is in c++
        categ_split_type = model$params$categ_split_type,  ## is in c++
        coefs = model$params$coefs,
        depth_imp = model$params$depth_imp,
        weigh_imp_rows = model$params$weigh_imp_rows,
        min_imp_obs = model$params$min_imp_obs,
        random_seed = model$random_seed,
        all_perm = model$params$all_perm,
        coef_by_prop = model$params$coef_by_prop,
        weights_as_sample_prob = model$params$weights_as_sample_prob,
        sample_with_replacement = model$params$sample_with_replacement,
        penalize_range = model$params$penalize_range,
        standardize_data = model$params$standardize_data,
        scoring_metric = model$params$scoring_metric,
        fast_bratio = model$params$fast_bratio,
        weigh_by_kurtosis = model$params$weigh_by_kurtosis,
        assume_full_distr = model$params$assume_full_distr
    )

    return(list(data_info = data_info, model_info = model_info, params = params))
}

take.metadata <- function(metadata) {
    this <- list(
        params  =  list(
            sample_size = metadata$params$sample_size, ntrees = metadata$params$ntrees, ndim = metadata$model_info$ndim,
            ntry = metadata$params$ntry, max_depth = metadata$params$max_depth,
            ncols_per_tree = metadata$params$ncols_per_tree,
            prob_pick_avg_gain = metadata$params$prob_pick_avg_gain,
            prob_pick_pooled_gain = metadata$params$prob_pick_pooled_gain,
            prob_pick_full_gain = metadata$params$prob_pick_full_gain,
            prob_pick_dens = metadata$params$prob_pick_dens,
            prob_pick_col_by_range = metadata$params$prob_pick_col_by_range,
            prob_pick_col_by_var = metadata$params$prob_pick_col_by_var,
            prob_pick_col_by_kurt = metadata$params$prob_pick_col_by_kurt,
            min_gain = metadata$params$min_gain, missing_action = metadata$params$missing_action,
            new_categ_action = metadata$params$new_categ_action,
            categ_split_type = metadata$params$categ_split_type,
            all_perm = metadata$params$all_perm, coef_by_prop = metadata$params$coef_by_prop,
            weights_as_sample_prob = metadata$params$weights_as_sample_prob,
            sample_with_replacement = metadata$params$sample_with_replacement,
            penalize_range = metadata$params$penalize_range,
            standardize_data = metadata$params$standardize_data,
            scoring_metric = metadata$params$scoring_metric,
            fast_bratio = metadata$params$fast_bratio,
            weigh_by_kurtosis = metadata$params$weigh_by_kurtosis,
            coefs = metadata$params$coefs, assume_full_distr = metadata$params$assume_full_distr,
            build_imputer = metadata$model_info$build_imputer, min_imp_obs = metadata$params$min_imp_obs,
            depth_imp = metadata$params$depth_imp, weigh_imp_rows = metadata$params$weigh_imp_rows
        ),
        metadata  = list(
            ncols_num  =  metadata$data_info$ncols_numeric,
            ncols_cat  =  metadata$data_info$ncols_categ,
            cols_num   =  unlist(metadata$data_info$cols_numeric),
            cols_cat   =  unlist(metadata$data_info$cols_categ),
            cat_levs   =  metadata$data_info$cat_levels,
            categ_cols =  metadata$data_info$categ_cols,
            categ_max  =  metadata$data_info$categ_max,
            reference_names = coerce.null(metadata$data_info$reference_names, character())
        ),
        use_long_double  =  coerce.null(metadata$model_info$use_long_double, FALSE),
        random_seed      =  metadata$params$random_seed,
        nthreads         =  metadata$model_info$nthreads,
        cpp_objects      =  list(
            model    =  NULL,
            imputer  =  NULL,
            indexer  =  NULL
        )
    )

    this$params$prob_pick_full_gain     <-  coerce.null(this$params$prob_pick_full_gain,    0.0)
    this$params$prob_pick_dens          <-  coerce.null(this$params$prob_pick_dens,         0.0)
    this$params$prob_pick_col_by_range  <-  coerce.null(this$params$prob_pick_col_by_range, 0.0)
    this$params$prob_pick_col_by_var    <-  coerce.null(this$params$prob_pick_col_by_var,   0.0)
    this$params$prob_pick_col_by_kurt   <-  coerce.null(this$params$prob_pick_col_by_kurt,  0.0)
    this$params$scoring_metric          <-  coerce.null(this$params$scoring_metric,         "depth")
    this$params$fast_bratio             <-  coerce.null(this$params$fast_bratio,            TRUE)

    if (!NROW(this$metadata$standardize_data))
        this$metadata$standardize_data <- TRUE
    
    if (NROW(this$metadata$cat_levels))
        names(this$metadata$cat_levels) <- this$metadata$cols_cat
    if (!NROW(this$metadata$categ_cols)) {
        this$metadata$categ_cols <- NULL
        this$metadata$categ_max  <- NULL
    }

    if ("prob_split_avg_gain" %in% names(metadata$params)) {
        msg <- "'prob_split_avg_gain' has been deprecated in favor of 'prob_pick_avg_gain' + 'ntry'."
        if (this$params$ndim == 1L) {
            msg <- paste0(msg, " Be sure to change these parameters if adding trees to this model.")
        }
    }

    if ("prob_split_pooled_gain" %in% names(metadata$params)) {
        msg <- "'prob_split_pooled_gain' has been deprecated in favor of 'prob_pick_pooled_gain' + 'ntry'."
        if (this$params$ndim == 1L) {
            msg <- paste0(msg, " Be sure to change these parameters if adding trees to this model.")
        }
    }
    
    class(this) <- "isolation_forest"
    return(this)
}

check.formatted.export.colnames <- function(model, column_names, column_names_categ) {
    if (model$metadata$ncols_num) {
        if (!is.null(column_names)) {
            if (NROW(column_names) != model$metadata$ncols_num)
                stop(sprintf("'column_names' must have length %d", model$metadata$ncols_num))
            if (!is.character(column_names))
                stop("'column_names' must be a character vector.")
            cols_num <- column_names
        } else {
            if (NROW(model$metadata$cols_num)) {
                cols_num <- model$metadata$cols_num
                if (is.integer(model$metadata$cols_num)) {
                    cols_num <- paste0("column_", cols_num)
                }
            } else {
                cols_num <- paste0("column_", seq(1L, model$metadata$ncols_num))
            }
        }
    } else {
        cols_num <- character()
    }
    
    if (model$metadata$ncols_cat) {
        if (!is.null(column_names_categ)) {
            if (NROW(column_names_categ) != model$metadata$ncols_cat)
                stop(sprintf("'column_names_categ' must have length %d", model$metadata$ncols_cat))
            if (!is.character(column_names_categ))
                stop("'column_names_categ' must be a character vector.")
            cols_cat <- column_names_categ
        } else {
            if (NROW(model$metadata$cols_cat))
                cols_cat <- model$metadata$cols_cat
            else
                cols_cat <- paste0("column_", model$metadata$categ_cols)
        }
        
        if (NROW(model$metadata$cat_levs))
            cat_levels <- model$metadata$cat_levs
        else
            cat_levels <- lapply(model$metadata$categ_max, function(x) as.character(seq(1, x + 1)))
    } else {
        cols_cat <- character()
        cat_levels <- list()
    }

    return(list(cols_num=cols_num, cols_cat=cols_cat, cat_levels=cat_levels))
}
