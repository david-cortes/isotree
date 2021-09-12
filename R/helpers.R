check.pos.int <- function(var, name) {
    if (NROW(var) != 1 || var < 1) {
        stop(paste0("'", name, "' must be a positive integer."))
    }
}

check.max.depth <- function(max_depth) {
    if (!is.null(max_depth)) {
        if (NROW(max_depth) != 1 || max_depth < 1) {
            stop(paste0("'max_depth' must be a non-negative integer."))
        }
        return(as.integer(max_depth))
    } else {
        return(0L)
    }
}

check.str.option <- function(option, name, allowed) {
    if (NROW(option) != 1 || !(option %in% allowed)) {
        stop(paste0("'", name, "' must be one of '", paste(allowed, collapse = "', '"), "'."))
    }
}

check.is.prob <- function(prob, name) {
    if (NROW(prob) != 1 || prob < 0 || prob > 1) {
        stop(paste0("'", name, "' must be a number between zero and one."))
    }
}

check.is.bool <- function(var, name) {
    if (NROW(var) != 1) stop(paste0("'", name, "' must be logical (boolean)."))
}

check.nthreads <- function(nthreads) {
    if (NROW(nthreads) != 1) stop("'nthreads' must be one of 'auto' or a positive integer.")
    if (is.null(nthreads)) {
        nthreads <- 1
    } else if (is.na(nthreads)) {
        nthreads <- 1
    } else if (nthreads == "auto") {
        nthreads <- parallel::detectCores()
    } else if (nthreads < 1) {
        nthreads <- parallel::detectCores()
    }
    return(as.integer(nthreads))
}

check.categ.cols <- function(categ_cols) {
    if (is.null(categ_cols) || !NROW(categ_cols))
        return(NULL)
    categ_cols <- as.integer(categ_cols)
    if (anyNA(categ_cols))
        stop("'categ_cols' cannot contain missing values.")
    if (any(categ_cols < 1))
        stop("'categ_cols' contains invalid column indices.")
    if (any(duplicated(categ_cols)))
        stop("'categ_cols' contains duplicted entries.")
    categ_cols <- sort.int(categ_cols)
    return(categ_cols)
}

check.is.1d <- function(var, name) {
    if (NCOL(var) > 1) {
        stop(paste0("'", name, "' must be a 1-d numeric vector."))
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

process.data <- function(df, sample_weights = NULL, column_weights = NULL, recode_categ = TRUE, categ_cols = NULL) {
    df  <-  cast.df.alike(df)
    dmatrix_types     <-  get.types.dmat()
    spmatrix_types    <-  get.types.spmat()
    supported_dtypes  <-  c("data.frame", dmatrix_types, spmatrix_types)
    if (!NROW(intersect(class(df), supported_dtypes)))
        stop(paste0("Invalid input data. Supported types are: ", paste(supported_dtypes, collapse = ", ")))
    
    if (NROW(df) < 2L) stop("Input data must have at least 2 rows.")
    
    if (!is.null(sample_weights))  sample_weights  <- as.numeric(sample_weights)
    if (!is.null(column_weights))  column_weights  <- as.numeric(column_weights)
    if (NROW(sample_weights)  && NROW(df) != NROW(sample_weights))
        stop(sprintf("'sample_weights' has different number of rows than df (%d vs. %d).",
                     NROW(df), NROW(sample_weights)))
    if (NROW(column_weights)  && NCOL(df) != NROW(column_weights))
        stop(sprintf("'column_weights' has different dimension than number of columns in df (%d vs. %d).",
                     NCOL(df), NROW(column_weights)))

    if (!is.null(categ_cols) && ("data.frame" %in% class(df))) {
        warning("'categ_cols' is ignored when passing a data.frame as input.")
        categ_cols <- NULL
    }

    if (ncol(df) < 1L)
        stop("'df' has no columns.")
    
    outp <- list(X_num      =  numeric(),
                 X_cat      =  integer(),
                 ncat       =  integer(),
                 cols_num   =  c(),
                 cols_cat   =  c(),
                 cat_levs   =  c(),
                 Xc         =  numeric(),
                 Xc_ind     =  integer(),
                 Xc_indptr  =  integer(),
                 nrows      =  as.integer(NROW(df)),
                 ncols_num  =  as.integer(NCOL(df)),
                 ncols_cat  =  as.integer(0L),
                 categ_cols =  NULL,
                 categ_max  =  integer(),
                 sample_weights  =  unname(as.numeric(sample_weights)),
                 column_weights  =  unname(as.numeric(column_weights))
                 )

    avoid_sparse_sort <- FALSE

    if (NROW(categ_cols)) {
        cols_num   <-  setdiff(1L:ncol(df), categ_cols)
        if (inherits(df, c("data.frame", "matrix", "dgCMatrix"))) {
            X_cat  <-  df[, categ_cols, drop=FALSE]
            df     <-  df[, cols_num,   drop=FALSE]
        } else if (inherits(df, "matrix.csc")) {
            nrows  <- nrow(df)
            df@ja  <- df@ja - 1L
            df@ia  <- df@ia - 1L
            df@ra  <- deepcopy_vector(df@ra)
            avoid_sparse_sort <- TRUE
            call_sort_csc_indices(df@ra, df@ja, df@ia)
            X_cat  <- call_take_cols_by_index_csc(df@ra,
                                                  df@ja,
                                                  df@ia,
                                                  categ_cols - 1L,
                                                  TRUE, nrows)
            X_cat  <- X_cat[["X_cat"]]
            df_new <- call_take_cols_by_index_csc(df@ra,
                                                  df@ja,
                                                  df@ia,
                                                  cols_num - 1L,
                                                  FALSE, nrows)
            df@ra  <- df_new[["Xc"]]
            df@ja  <- df_new[["Xc_ind"]] + 1L
            df@ia  <- df_new[["Xc_indptr"]] + 1L
            df@dimension <- as.integer(c(nrows, NROW(cols_num)))
        } else {
            X_cat  <-  df[, categ_cols]
            df     <-  df[, cols_num]
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
        outp$ncat        <-  categ_max + 1L
        outp$cols_num    <-  cols_num
        outp$ncols_num   <-  ncol(df)
        outp$ncols_cat   <-  ncols_cat

        if (!ncol(df))
            return(outp)
    }
    
    ### Dense matrix
    if ( any(class(df) %in% dmatrix_types) ) {
        outp$X_num      <-  unname(as.numeric(df))
        outp$ncols_num  <-  ncol(df)
        return(outp)
    }
    
    ### Sparse matrix
    if ( any(class(df) %in% spmatrix_types) ) {
        
        if (inherits(df, "dgCMatrix")) {
            ### From package 'Matrix'
            if (!NROW(df@x))
                stop("'df' has no non-zero entries.")
            outp$Xc         <-  df@x
            outp$Xc_ind     <-  df@i
            outp$Xc_indptr  <-  df@p
        } else {
            ### From package 'SparseM'
            if (!NROW(df@ra))
                stop("'df' has no non-zero entries.")
            outp$Xc         <-  df@ra
            outp$Xc_ind     <-  df@ja - 1L
            outp$Xc_indptr  <-  df@ia - 1L
        }
        if (!avoid_sparse_sort) {
            if (!inherits(df, "dgCMatrix"))
                outp$Xc     <- deepcopy_vector(outp$Xc)
            call_sort_csc_indices(outp$Xc, outp$Xc_ind, outp$Xc_indptr)
        }
        outp$ncols_num      <-  ncol(df)
        
        return(outp)
    }
    
    ### Data Frame
    if ( "data.frame" %in% class(df) ) {
        dtypes_num  <-  c("numeric",   "integer",  "Date",  "POSIXct")
        dtypes_cat  <-  c("character", "factor",   "logical")
        supported_col_types <- c(dtypes_num, dtypes_cat)
        df_coltypes <- Reduce(c, sapply(df, class))
        if (any(!(df_coltypes %in% c(supported_col_types, "POSIXt")))) {
            stop(paste0("Input data contains unsupported column types. Supported types are ",
                        paste(supported_col_types, collapse = ", "), " - got the following: ",
                        paste(unique(df_coltypes[!(df_coltypes %in% supported_col_types)]), collapse = ", ")))
        
        }
        
        if (any(df_coltypes %in% dtypes_num)) {
            is_num          <-  unname(as.logical(sapply(df, function(x) any(class(x) %in% dtypes_num))))
            outp$cols_num   <-  names(df)[is_num]
            outp$ncols_num  <-  as.integer(sum(is_num))
            outp$X_num      <-  unname(as.numeric(as.matrix(as.data.frame(lapply(df[, is_num, drop = FALSE], cast.df.col.to.num)))))
        } else { outp$ncols_num <- as.integer(0) }
        
        if (any(df_coltypes %in% dtypes_cat)) {
            if (any("ordered" %in% df_coltypes))
                warning("Data contains ordered factors. These are treated as unordered.")
            is_cat          <-  unname(as.logical(sapply(df, function(x) any(class(x) %in% dtypes_cat))))
            outp$cols_cat   <-  names(df)[is_cat]
            outp$ncols_cat  <-  as.integer(sum(is_cat))
            if (recode_categ) {
                outp$X_cat  <-  as.data.frame(lapply(df[, is_cat, drop = FALSE], factor))
            } else {
                outp$X_cat  <-  as.data.frame(lapply(df[, is_cat, drop = FALSE],
                                                     function(x) if("factor" %in% class(x)) x else factor(x)))
            }
            outp$cat_levs   <-  lapply(outp$X_cat, levels)
            outp$ncat       <-  sapply(outp$cat_levs, NROW)
            outp$X_cat      <-  as.data.frame(lapply(outp$X_cat, function(x) ifelse(is.na(x), -1L, as.integer(x) - 1L)))
            outp$X_cat      <-  unname(as.integer(as.matrix(outp$X_cat)))
        }
        
        if (NROW(outp$cols_num) && NROW(outp$cols_cat) && NROW(outp$column_weights)) {
            outp$column_weights <- c(outp$column_weights[names(df) %in% outp$cols_num],
                                     outp$column_weights[names(df) %in% outp$cols_cat])
        }
        
        return(outp)
    }
    
    stop("Unexpected error.")
}

process.data.new <- function(df, metadata, allow_csr = FALSE, allow_csc = TRUE, enforce_shape = FALSE) {
    if (!NROW(df)) stop("'df' contains zero rows.")
    if (inherits(df, "sparseVector") && !inherits(df, "dsparseVector"))
        stop("Sparse vectors only allowed as 'dsparseVector' class.")
    if (!inherits(df, "sparseVector")) {
        if ( NCOL(df) < (metadata$ncols_num + metadata$ncols_cat) )
            stop(sprintf("Input data contains fewer columns than expected (%d vs. %d)",
                         NCOL(df), (metadata$ncols_num + metadata$ncols_cat)))
    } else {
        if (df@length < (metadata$ncols_num + metadata$ncols_cat))
            stop(sprintf("Input data contains different columns than expected (%d vs. %d)",
                         df@length, (metadata$ncols_num + metadata$ncols_cat)))
    }
    df  <-  cast.df.alike(df)
    if (metadata$ncols_cat > 0L && !NROW(metadata$categ_cols) && !inherits(df, "data.frame"))
        stop("Model was fit to data.frame with categorical data, must pass a data.frame with new data.")
    
    dmatrix_types     <-  get.types.dmat()
    spmatrix_types    <-  get.types.spmat(allow_csr = allow_csr, allow_csc = allow_csc, TRUE)
    supported_dtypes  <-  c("data.frame", dmatrix_types, spmatrix_types)

    if (!NROW(intersect(class(df), supported_dtypes)))
        stop(paste0("Invalid input data. Supported types are: ", paste(supported_dtypes, collapse = ", ")))

    if (!allow_csr && inherits(df, c("RsparseMatrix", "matrix.csr")))
        stop("CSR matrix not supported for this prediction type. Try converting to CSC.")
    if (!allow_csc && inherits(df, c("CsparseMatrix", "matrix.csc")))
        stop("CSC matrix not supported for this prediction type. Try converting to CSR.")

    outp <- list(
        X_num      =  numeric(),
        X_cat      =  integer(),
        nrows      =  as.integer(NROW(df)),
        Xc         =  numeric(),
        Xc_ind     =  integer(),
        Xc_indptr  =  integer(),
        Xr         =  numeric(),
        Xr_ind     =  integer(),
        Xr_indptr  =  integer()
    )

    avoid_sparse_sort <- FALSE

    if (!NROW(metadata$categ_cols)) {
        
        if (((!NROW(metadata$cols_num) && !NROW(metadata$cols_cat)) || !inherits(df, "data.frame")) &&
            (   (inherits(df, "sparseVector") && df@length > metadata$ncols_num) ||
                (!inherits(df, "sparseVector") && (ncol(df) > metadata$ncols_num)))
            && (enforce_shape || inherits(df, c("RsparseMatrix", "matrix.csr")))
            ) {

            if (inherits(df, c("matrix", "CsparseMatrix")) ||
                (!NROW(metadata$cols_num) && inherits(df, "data.frame"))) {
                df <- df[, 1L:metadata$ncols_num, drop=FALSE]
            } else if (inherits(df, "sparseVector")) {
                df <- df[1L:metadata$ncols_num]
            } else if (inherits(df, "RsparseMatrix")) {
                nrows <- nrow(df)
                avoid_sparse_sort <- TRUE
                call_sort_csc_indices(df@x, df@j, df@p)
                df_new <- call_take_cols_by_slice_csr(
                                df@x,
                                df@j,
                                df@p,
                                metadata$ncols_num,
                                FALSE
                            )
                df@x <- df_new[["Xr"]]
                df@j <- df_new[["Xr_ind"]]
                df@p <- df_new[["Xr_indptr"]]
                df@Dim <- as.integer(c(nrows, metadata$ncols_num))
            } else if (inherits(df, "matrix.csr")) {
                avoid_sparse_sort <- TRUE
                df@ja <- df@ja - 1L
                df@ia <- df@ia - 1L
                df@ra <- deepcopy_vector(df@ra)
                call_sort_csc_indices(df@ra, df@ja, df@ia)
                df_new <- call_take_cols_by_slice_csr(
                                df@ra,
                                df@ja,
                                df@ia,
                                metadata$ncols_num,
                                FALSE
                            )
                df@ra <- df_new[["Xr"]]
                df@ja <- df_new[["Xr_ind"]] + 1L
                df@ia <- df_new[["Xr_indptr"]] + 1L
                df@dimension <- as.integer(c(nrows, metadata$ncols_num))
            } else if (inherits(df, "matrix.csc")) {
                df@ia <- df@ia - 1L
                nrows <- nrow(df)
                df_new <- call_take_cols_by_slice_csc(
                                df@ra,
                                df@ja,
                                df@ia,
                                metadata$ncols_num,
                                FALSE, nrows
                            )
                df@ra <- df_new[["Xc"]]
                df@ja <- df_new[["Xc_ind"]]
                df@ia <- df_new[["Xc_indptr"]] + 1L
                df@dimension <- as.integer(c(nrows, metadata$ncols_num))
            } else if (!inherits(df, "data.frame")) {
                df <- df[, 1L:metadata$ncols_num]
            }

        }

    } else { ### has metadata$categ_cols

        if (!inherits(df, "sparseVector")) {

            nrows <- nrow(df)
            if (inherits(df, c("matrix", "data.frame", "dgCMatrix"))) {
                X_cat  <- df[, metadata$categ_cols,  drop=FALSE]
                df     <- df[, metadata$cols_num,    drop=FALSE]
            } else if (inherits(df, "dgRMatrix")) {
                avoid_sparse_sort <- TRUE
                call_sort_csc_indices(df@x, df@j, df@p)
                X_cat  <- call_take_cols_by_index_csr(df@x,
                                                      df@j,
                                                      df@p,
                                                      metadata$categ_cols - 1L,
                                                      TRUE)
                X_cat  <- X_cat[["X_cat"]]
                df_new <- call_take_cols_by_index_csr(df@x,
                                                      df@j,
                                                      df@p,
                                                      metadata$cols_num - 1L,
                                                      FALSE)
                df@x   <- df_new[["Xr"]]
                df@j   <- df_new[["Xr_ind"]]
                df@p   <- df_new[["Xr_indptr"]]
                df@Dim <- as.integer(c(nrows, NROW(metadata$cols_num)))
            } else if (inherits(df, "matrix.csc")) {
                avoid_sparse_sort <- TRUE
                df@ja  <- df@ja - 1L
                df@ia  <- df@ia - 1L
                df@ra  <- deepcopy_vector(df@ra)
                call_sort_csc_indices(df@ra, df@ja, df@ia)

                X_cat  <- call_take_cols_by_index_csc(df@ra,
                                                      df@ja,
                                                      df@ia,
                                                      metadata$categ_cols - 1L,
                                                      TRUE, nrows)
                X_cat  <- X_cat[["X_cat"]]
                df_new <- call_take_cols_by_index_csc(df@ra,
                                                      df@ja,
                                                      df@ia,
                                                      metadata$cols_num - 1L,
                                                      FALSE, nrows)
                df@ra  <- df_new[["Xc"]]
                df@ja  <- df_new[["Xc_ind"]] + 1L
                df@ia  <- df_new[["Xc_indptr"]] + 1L
                df@dimension <- as.integer(c(nrows, NROW(metadata$cols_num)))
            } else if (inherits(df, "matrix.csr")) {
                avoid_sparse_sort <- TRUE
                df@ja  <- df@ja - 1L
                df@ia  <- df@ia - 1L
                df@ra  <- deepcopy_vector(df@ra)
                call_sort_csc_indices(df@ra, df@ja, df@ia)

                X_cat  <- call_take_cols_by_index_csr(df@ra,
                                                      df@ja,
                                                      df@ia,
                                                      metadata$categ_cols - 1L,
                                                      TRUE)
                X_cat  <- X_cat[["X_cat"]]
                df_new <- call_take_cols_by_index_csr(df@ra,
                                                      df@ja,
                                                      df@ia,
                                                      metadata$cols_num - 1L,
                                                      FALSE)
                df@ra  <- df_new[["Xr"]]
                df@ja  <- df_new[["Xr_ind"]] + 1L
                df@ia  <- df_new[["Xr_indptr"]] + 1L
                df@dimension <- as.integer(c(nrows, NROW(metadata$cols_num)))
            } else {
                X_cat  <- df[, metadata$categ_cols]
                df     <- df[, metadata$cols_num]
            }

        } else { ### sparseVector
            X_cat <- matrix(df[metadata$categ_cols], nrow=1L)
            nrows <- 1L
            df    <- df[metadata$cols_num]
        }

        X_cat[sweep(X_cat, 2, metadata$categ_max, ">")] <- -1L
        if (!inherits(X_cat, "matrix"))
            X_cat <- as.matrix(X_cat)
        X_cat <- as.integer(X_cat)
        if (anyNA(X_cat))
            X_cat[is.na(X_cat)] <- -1L
        outp$X_cat <- X_cat
        outp$nrows <- nrows

    }

    if (inherits(df, "data.frame") &&
        (NROW(metadata$categ_cols) ||
        (!NROW(metadata$cols_num) && !NROW(metadata$cols_cat)))
        ) {
        df <- as.data.frame(lapply(df, cast.df.col.to.num))
        df <- as.matrix(df)
    }

    
    if (inherits(df, "data.frame")) {
        
        if (NROW(setdiff(c(metadata$cols_num, metadata$cols_cat), names(df)))) {
            missing_cols <- setdiff(c(metadata$cols_num, metadata$cols_cat), names(df))
            stop(paste0(sprintf("Input data is missing %d columns - head: ", NROW(missing_cols)),
                        paste(head(missing_cols, 3), collapse = ", ")))
        }
        
        if (!NROW(metadata$cols_num) && !NROW(metadata$cols_cat)) {
            
            if (NCOL(df) != metadata$ncols_num)
                stop(sprintf("Input data has %d columns, but model was fit to data with %d columns.",
                             NCOL(df), (metadata$ncols_num + metadata$ncols_cat)))
            outp$X_num <- unname(as.numeric(as.matrix(as.data.frame(lapply(df, cast.df.col.to.num)))))
            
        } else {
            
            if (metadata$ncols_num > 0L) {
                outp$X_num <- unname(as.numeric(as.matrix(as.data.frame(lapply(df[, metadata$cols_num, drop = FALSE], cast.df.col.to.num)))))
            }
            
            if (metadata$ncols_cat > 0L) {
                outp$X_cat <- df[, metadata$cols_cat, drop = FALSE]
                outp$X_cat <- as.data.frame(mapply(function(cl, levs) factor(cl, levs),
                                                   outp$X_cat, metadata$cat_levs,
                                                   SIMPLIFY = FALSE, USE.NAMES = FALSE))
                outp$X_cat <- as.data.frame(lapply(outp$X_cat, function(x) ifelse(is.na(x), -1L, as.integer(x) - 1L)))
                outp$X_cat <- unname(as.integer(as.matrix(outp$X_cat)))
            }
            
        }
        
    } else if (inherits(df, "dsparseVector")) {
        if (allow_csr) {
            df@x   <-  df@x[order(df@i)]
            df@i   <-  df@i[order(df@i)]

            outp$Xr         <-  as.numeric(df@x)
            outp$Xr_ind     <-  as.integer(df@i - 1L)
            outp$Xr_indptr  <-  as.integer(c(0L, NROW(df@x)))
        } else {
            outp$X_num      <-  as.numeric(df)
        }
        outp$nrows          <-  1L
    } else {
        
        if ("numeric" %in% class(df) && is.null(dim(df)))
            df <- matrix(df, nrow = 1)
        
        if (NCOL(df) < metadata$ncols_num)
            stop(sprintf("Input data has %d numeric columns, but model was fit to data with %d numeric columns.",
                         NCOL(df), metadata$ncols_num))
        if (!any(class(df) %in% spmatrix_types)) {
            outp$X_num <- as.numeric(df)
        } else {

            if (inherits(df, "dgCMatrix")) {
                ### From package 'Matrix'
                outp$Xc         <-  df@x
                outp$Xc_ind     <-  df@i
                outp$Xc_indptr  <-  df@p
                if (!avoid_sparse_sort)
                    call_sort_csc_indices(outp$Xc, outp$Xc_ind, outp$Xc_indptr)
            } else if (inherits(df, "dgRMatrix")) {
                ### From package 'Matrix'
                outp$Xr         <-  df@x
                outp$Xr_ind     <-  df@j
                outp$Xr_indptr  <-  df@p
                if (!avoid_sparse_sort)
                    call_sort_csc_indices(outp$Xr, outp$Xr_ind, outp$Xr_indptr)
            } else if (inherits(df, "matrix.csc")) {
                ### From package 'SparseM'
                outp$Xc         <-  df@ra
                outp$Xc_ind     <-  df@ja - 1L
                outp$Xc_indptr  <-  df@ia - 1L
                if (!avoid_sparse_sort) {
                    outp$Xc     <-  deepcopy_vector(outp$Xc)
                    call_sort_csc_indices(outp$Xc, outp$Xc_ind, outp$Xc_indptr)
                }
            } else if (inherits(df, "matrix.csr")) {
                ### From package 'SparseM'
                outp$Xr         <-  df@ra
                outp$Xr_ind     <-  df@ja - 1L
                outp$Xr_indptr  <-  df@ia - 1L
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

reconstruct.from.imp <- function(imputed_num, imputed_cat, df, model, pdata) {

    if (NROW(imputed_cat))
        imputed_cat[imputed_cat < 0L] <- NA_integer_

    if (inherits(df, "RsparseMatrix")) {

        outp <- df
        if (!NROW(model$metadata$categ_cols) && ncol(df) == model$metadata$ncols_num) {
            outp@x <- imputed_num
        } else if (!NROW(model$metadata$categ_cols)) {
            outp@x <- deepcopy_vector(outp@x)
            call_reconstruct_csr_sliced(
                outp@x, outp@p,
                imputed_num, pdata$Xr_indptr,
                nrow(df)
            )
        } else {
            outp@x <- deepcopy_vector(outp@x)
            call_reconstruct_csr_with_categ(
                outp@x, outp@j, outp@p,
                imputed_num, pdata$Xr_ind, pdata$Xr_indptr,
                imputed_cat,
                model$metadata$cols_num-1L, model$metadata$categ_cols-1L,
                nrow(df), ncol(df)
            )
        }
        return(outp)

    } else if (inherits(df, "CsparseMatrix")) {

        outp       <-  df
        if (!NROW(model$metadata$categ_cols)) {
            outp@x <-  imputed_num
        } else {
            outp[, model$metadata$categ_cols] <- matrix(imputed_cat, nrow=nrow(df))
            copy_csc_cols_by_index(
                outp@x,
                outp@p,
                imputed_num,
                pdata$Xc_indptr,
                model$metadata$cols_num - 1L
            )
        }
        return(outp)

    } else if (inherits(df, "matrix.csr")) {
        
        outp <- df
        if (!NROW(model$metadata$categ_cols) && ncol(df) == model$metadata$ncols_num) {
            outp@ra <- imputed_num
        } else if (!NROW(model$metadata$categ_cols)) {
            outp@ra <- deepcopy_vector(outp@ra)
            call_reconstruct_csr_sliced(
                outp@ra, outp@ia-1L,
                imputed_num, pdata$Xr_indptr,
                nrow(df)
            )
        } else {
            outp@ra <- deepcopy_vector(outp@ra)
            call_reconstruct_csr_with_categ(
                outp@ra, outp@ja-1L, outp@ia-1L,
                imputed_num, pdata$Xr_ind, pdata$Xr_indptr,
                imputed_cat,
                model$metadata$cols_num-1L, model$metadata$categ_cols-1L,
                nrow(df), ncol(df)
            )
        }
        return(outp)

    } else if (inherits(df, "matrix.csc")) {
        
        outp <- df
        if (!NROW(model$metadata$categ_cols)) {
            outp@ra  <-  imputed_num
        } else {
            df_new   <- assign_csc_cols(
                pdata$Xc,
                pdata$Xc_ind,
                pdata$Xc_indptr,
                imputed_cat,
                model$metadata$categ_cols - 1L,
                model$metadata$cols_num - 1L,
                nrow(df)
            )
            copy_csc_cols_by_index(
                df_new$Xc,
                df_new$Xc_indptr,
                imputed_num,
                pdata$Xc_indptr,
                model$metadata$cols_num - 1L
            )
            outp@ra <- df_new$Xc
            outp@ja <- df_new$Xc_ind + 1L
            outp@ia <- df_new$Xc_indptr + 1L
            outp@dimension <- as.integer(c(nrow(df), length(df_new$Xc_indptr)-1L))
        }
        return(outp)

    } else if (inherits(df, "sparseVector")) {

        if (!NROW(model$metadata$categ_cols) && df@length == model$metadata$ncols_num) {
            df@x <- imputed_num
        } else if (!NROW(model$metadata$categ_cols)) {
            df@x[1L:NROW(imputed_num)]    <- imputed_num
        } else {
            df[model$metadata$cols_num]   <- imputed_num
            df[model$metadata$categ_cols] <- imputed_cat
        }

    } else if (!inherits(df, "data.frame")) {

        if (!NROW(model$metadata$categ_cols) && (ncol(df) == model$metadata$ncols_num)) {
            return(matrix(imputed_num, nrow = NROW(df)))
        } else if (!NROW(model$metadata$categ_cols)) {
            df[, 1L:model$metadata$ncols_num]  <-  matrix(imputed_num, nrow = NROW(df))
            return(df)
        } else {
            df[, model$metadata$categ_cols]    <-  matrix(imputed_cat, nrow = NROW(df))
            if (model$metadata$ncols_num)
                df[, model$metadata$cols_num]  <-  matrix(imputed_num, nrow = NROW(df))
            return(df)
        }

    } else {
        
        df_num <- as.data.frame(matrix(imputed_num, nrow = NROW(df)))
        df_cat <- as.data.frame(matrix(imputed_cat, nrow = NROW(df)))
        if (!NROW(model$metadata$categ_cols)) {
            df_cat <- as.data.frame(mapply(function(x, levs) factor(x, labels = levs),
                                           df_cat + 1L, model$metadata$cat_levs,
                                           SIMPLIFY = FALSE))
        }

        if (NROW(model$metadata$categ_cols)) {
            df[, model$metadata$categ_cols]   <- df_cat
            if (model$metadata$ncols_num)
                df[, model$metadata$cols_num] <- df_num
        } else if (!NROW(model$metadata$cols_num)) {
            df[, 1L:model$metadata$ncols_num] <- df_num
        } else {
            if (model$metadata$ncols_num)
                df[, model$metadata$cols_num] <- df_num
            if (model$metadata$ncols_cat)
                df[, model$metadata$cols_cat] <- df_cat
        }
        return(df)

    }
}

export.metadata <- function(model) {
    data_info <- list(
        ncols_numeric  =  model$metadata$ncols_num, ## is in c++
        ncols_categ    =  model$metadata$ncols_cat,  ## is in c++
        cols_numeric   =  as.list(model$metadata$cols_num),
        cols_categ     =  as.list(model$metadata$cols_cat),
        cat_levels     =  unname(as.list(model$metadata$cat_levs)),
        categ_cols     =  model$metadata$categ_cols,
        categ_max      =  model$metadata$categ_max
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
        build_imputer = model$params$build_imputer
    )
    
    params <- list(
        sample_size = model$params$sample_size,
        ntrees = model$params$ntrees,  ## is in c++
        ntry = model$params$ntry,
        max_depth = model$params$max_depth,
        ncols_per_tree = model$params$ncols_per_tree,
        prob_pick_avg_gain = model$params$prob_pick_avg_gain,
        prob_pick_pooled_gain = model$params$prob_pick_pooled_gain,
        prob_split_avg_gain = model$params$prob_split_avg_gain,
        prob_split_pooled_gain = model$params$prob_split_pooled_gain,
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
            prob_split_avg_gain = metadata$params$prob_split_avg_gain,
            prob_split_pooled_gain = metadata$params$prob_split_pooled_gain,
            min_gain = metadata$params$min_gain, missing_action = metadata$params$missing_action,
            new_categ_action = metadata$params$new_categ_action,
            categ_split_type = metadata$params$categ_split_type,
            all_perm = metadata$params$all_perm, coef_by_prop = metadata$params$coef_by_prop,
            weights_as_sample_prob = metadata$params$weights_as_sample_prob,
            sample_with_replacement = metadata$params$sample_with_replacement,
            penalize_range = metadata$params$penalize_range,
            standardize_data = metadata$params$standardize_data,
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
            categ_max  =  metadata$data_info$categ_max
        ),
        random_seed  =  metadata$params$random_seed,
        nthreads     =  metadata$model_info$nthreads,
        cpp_obj      =  as.environment(list(
            ptr         =  NULL,
            serialized  =  NULL,
            imp_ptr     =  NULL,
            imp_ser     =  NULL
        ))
    )

    if (!NROW(this$metadata$standardize_data))
        this$metadata$standardize_data <- TRUE
    
    if (NROW(this$metadata$cat_levels))
        names(this$metadata$cat_levels) <- this$metadata$cols_cat
    if (!NROW(this$metadata$categ_cols)) {
        this$metadata$categ_cols <- NULL
        this$metadata$categ_max  <- NULL
    }
    
    class(this) <- "isolation_forest"
    return(this)
}
