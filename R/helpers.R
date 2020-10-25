check.pos.int <- function(var, name) {
    if (NROW(var) != 1 || var < 1) {
        stop(paste0("'", name, "' must be a positive integer."))
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

check.is.1d <- function(var, name) {
    if (NCOL(var) > 1) {
        stop(paste0("'", name, "' must be a 1-d numeric vector."))
    }
}

get.empty.vector <- function() {
    outp <- vector("numeric", 0L)
    return(outp)
}

get.empty.int.vector <- function() {
    outp <- vector("integer", 0L)
    return(outp)
}

cast.df.alike <- function(df) {
    if ("data.table" %in% class(df))  df  <- as.data.frame(df)
    if ("tibble"     %in% class(df))  df  <- as.data.frame(df)
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

process.data <- function(df, sample_weights = NULL, column_weights = NULL, recode_categ = TRUE) {
    df  <-  cast.df.alike(df)
    dmatrix_types     <-  get.types.dmat()
    spmatrix_types    <-  get.types.spmat()
    supported_dtypes  <-  c("data.frame", dmatrix_types, spmatrix_types)
    if (!NROW(intersect(class(df), supported_dtypes)))
        stop(paste0("Invalid input data. Supported types are: ", paste(supported_dtypes, collapse = ", ")))
    
    if (NROW(df) < 5) stop("Input data must have at least 5 rows.")
    
    if (!is.null(sample_weights))  sample_weights  <- as.numeric(sample_weights)
    if (!is.null(column_weights))  column_weights  <- as.numeric(column_weights)
    if (NROW(sample_weights)  && NROW(df) != NROW(sample_weights))
        stop(sprintf("'sample_weights' has different number of rows than df (%d vs. %d).",
                     NROW(df), NROW(sample_weights)))
    if (NROW(column_weights)  && NCOL(df) != NROW(column_weights))
        stop(sprintf("'column_weights' has different dimension than number of columns in df (%d vs. %d).",
                     NCOL(df), NROW(column_weights)))
    
    
    outp <- list(X_num      =  get.empty.vector(),
                 X_cat      =  get.empty.int.vector(),
                 ncat       =  get.empty.int.vector(),
                 cols_num   =  c(),
                 cols_cat   =  c(),
                 cat_levs   =  c(),
                 Xc         =  get.empty.vector(),
                 Xc_ind     =  get.empty.int.vector(),
                 Xc_indptr  =  get.empty.int.vector(),
                 nrows      =  as.integer(NROW(df)),
                 ncols_num  =  as.integer(NCOL(df)),
                 ncols_cat  =  as.integer(0L),
                 sample_weights  =  unname(as.numeric(sample_weights)),
                 column_weights  =  unname(as.numeric(column_weights))
                 )
    
    ### Dense matrix
    if ( any(class(df) %in% dmatrix_types) ) { outp$X_num <- unname(as.numeric(df)) ; return(outp) }
    
    ### Sparse matrix
    if ( any(class(df) %in% spmatrix_types) ) {
        
        if ("dgCMatrix" %in% class(df)) {
            ### From package 'Matrix'
            if (NROW(df@x) == 0)
                stop("'df' has no non-zero entries.")
            outp$Xc         <-  as.numeric(df@x)
            outp$Xc_ind     <-  as.integer(df@i)
            outp$Xc_indptr  <-  as.integer(df@p)
        } else {
            ### From package 'SparseM'
            if (NROW(df@ra) == 0)
                stop("'df' has no non-zero entries.")
            outp$Xc         <-  as.numeric(df@ra)
            outp$Xc_ind     <-  as.integer(df@ia - 1L)
            outp$Xc_indptr  <-  as.integer(df@ja - 1L)
        }
        
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
            outp$X_num      <-  unname(as.numeric(as.matrix(as.data.frame(lapply(df[, is_num, drop = FALSE], as.numeric)))))
        } else { outp$ncols_num <- as.integer(0) }
        
        if (any(df_coltypes %in% dtypes_cat)) {
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

process.data.new <- function(df, metadata, allow_csr = FALSE, allow_csc = TRUE) {
    if (!NROW(df)) stop("'df' contains zero rows.")
    if (!("dsparseVector" %in% class(df))) {
        if ( NCOL(df) < (metadata$ncols_num + metadata$ncols_cat) )
            stop(sprintf("Input data contains fewer columns than expected (%d vs. %d)",
                         NCOL(df), (metadata$ncols_num + metadata$ncols_cat)))
    } else {
        if (df@length != metadata$ncols_num)
            stop(sprintf("Input data contains different columns than expected (%d vs. %d)",
                         df@length, (metadata$ncols_num)))
    }
    df  <-  cast.df.alike(df)
    if (metadata$ncols_cat > 0 && !("data.frame" %in% class(df)))
        stop("Model was fit to categorical data, must pass a data.frame with new data.")
    
    dmatrix_types     <-  get.types.dmat()
    spmatrix_types    <-  get.types.spmat(allow_csr = allow_csr, allow_csc = allow_csc, TRUE)
    supported_dtypes  <-  c("data.frame", dmatrix_types, spmatrix_types)

    if (!NROW(intersect(class(df), supported_dtypes)))
        stop(paste0("Invalid input data. Supported types are: ", paste(supported_dtypes, collapse = ", ")))
    
    outp <- list(
        X_num      =  get.empty.vector(),
        X_cat      =  get.empty.int.vector(),
        nrows      =  as.integer(NROW(df)),
        Xc         =  get.empty.vector(),
        Xc_ind     =  get.empty.int.vector(),
        Xc_indptr  =  get.empty.int.vector(),
        Xr         =  get.empty.vector(),
        Xr_ind     =  get.empty.int.vector(),
        Xr_indptr  =  get.empty.int.vector()
    )
    
    if ("data.frame" %in% class(df)) {
        
        if (NROW(setdiff(c(metadata$cols_num, metadata$cols_cat), names(df)))) {
            missing_cols <- setdiff(c(metadata$cols_num, metadata$cols_cat), names(df))
            stop(paste0(sprintf("Input data is missing %d columns - head: ", NROW(missing_cols)),
                        paste(head(missing_cols, 3), collapse = ", ")))
        }
        
        if (!NROW(metadata$cols_num) && !NROW(metadata$cols_cat)) {
            
            if (NCOL(df) != metadata$ncols_num)
                stop(sprintf("Input data has %d columns, but model was fit to data with %d columns.",
                             NCOL(df), (metadata$ncols_num + metadata$ncols_cat)))
            outp$X_num <- unname(as.numeric(as.matrix(as.data.frame(lapply(df, as.numeric)))))
            
        } else {
            
            if (metadata$ncols_num > 0) {
                outp$X_num <- unname(as.numeric(as.matrix(as.data.frame(lapply(df[, metadata$cols_num, drop = FALSE], as.numeric)))))
            }
            
            if (metadata$ncols_cat > 0) {
                outp$X_cat <- df[, metadata$cols_cat, drop = FALSE]
                outp$X_cat <- as.data.frame(mapply(function(cl, levs) factor(cl, levs),
                                                   outp$X_cat, metadata$cat_levs,
                                                   SIMPLIFY = FALSE, USE.NAMES = FALSE))
                outp$X_cat <- as.data.frame(lapply(outp$X_cat, function(x) ifelse(is.na(x), -1L, as.integer(x) - 1L)))
                outp$X_cat <- unname(as.integer(as.matrix(outp$X_cat)))
            }
            
        }
        
    } else if ("dsparseVector" %in% class(df)) {
        outp$Xr         <-  as.numeric(df@x)
        outp$Xr_ind     <-  as.integer(df@i) - 1L
        outp$Xr_indptr  <-  as.integer(c(0L, NROW(df@x)))
        outp$nrows      <-  1L
    } else {
        
        if ("numeric" %in% class(df) && is.null(dim(df)))
            df <- matrix(df, nrow = 1)
        
        if (NCOL(df) != (metadata$ncols_num + metadata$ncols_cat))
            stop(sprintf("Input data has %d columns, but model was fit to data with %d columns.",
                         NCOL(df), (metadata$ncols_num + metadata$ncols_cat)))
        if (!any(class(df) %in% spmatrix_types)) {
            outp$X_num <- as.numeric(df)
        } else {
            if ("dgCMatrix" %in% class(df)) {
                ### From package 'Matrix'
                if (allow_csc) {
                    outp$Xc         <-  as.numeric(df@x)
                    outp$Xc_ind     <-  as.integer(df@i)
                    outp$Xc_indptr  <-  as.integer(df@p)
                } else {
                    df <- Matrix::t(df)
                    outp$Xr         <-  as.numeric(df@x)
                    outp$Xr_ind     <-  as.integer(df@i)
                    outp$Xr_indptr  <-  as.integer(df@p)
                }
            } else if ("dgRMatrix" %in% class(df)) {
                ### From package 'Matrix'
                outp$Xr         <-  as.numeric(df@x)
                outp$Xr_ind     <-  as.integer(df@j)
                outp$Xr_indptr  <-  as.integer(df@p)
            } else {
                ### From package 'SparseM'
                if ("matrix.csc" %in% class(df)) {
                    outp$Xc         <-  as.numeric(df@ra)
                    outp$Xc_ind     <-  as.integer(df@ia - 1L)
                    outp$Xc_indptr  <-  as.integer(df@ja - 1L)
                } else {
                    outp$Xr         <-  as.numeric(df@ra)
                    outp$Xr_ind     <-  as.integer(df@ia - 1L)
                    outp$Xr_indptr  <-  as.integer(df@ja - 1L)
                }
            }
        }
        
    }
    
    return(outp)
}

reconstruct.from.imp <- function(imputed_num, imputed_cat, df, model, trans_CSC=FALSE) {
    
    if ("dgCMatrix" %in% class(df)) {
        outp     <-  df
        if (trans_CSC) outp <- Matrix::t(outp)
        outp@x   <-  imputed_num
        if (trans_CSC) outp <- Matrix::t(outp)
        return(outp)
    } else if (("dgRMatrix" %in% class(df)) || ("dsparseVector" %in% class(df))) {
        outp     <-  df
        outp@x   <-  imputed_num
        return(outp)
    } else if ( any(class(df) %in% c("matrix.csr", "matrix.csc")) ) {
        outp     <-  df
        outp@ra  <-  imputed_num
    } else if (!("data.frame" %in% class(df))) {
        return(matrix(imputed_num, nrow = NROW(df)))
    } else {
        df_num <- as.data.frame(matrix(imputed_num, nrow = NROW(df)))
        names(df_num) <- model$metadata$cols_num
        
        df_cat <- as.data.frame(matrix(ifelse(imputed_cat < 0, NA_integer_, imputed_cat) + 1L, nrow = NROW(df)))
        names(df_cat) <- model$metadata$cols_cat
        df_cat <- as.data.frame(mapply(function(x, levs) factor(x, labels = levs),
                                       df_cat, model$metadata$cat_levs,
                                       SIMPLIFY = FALSE))
        
        df_merged <- cbind(df_num, df_cat)
        df_merged <- df_merged[, names(df)]
        return(df_merged)
    }
}

export.metadata <- function(model) {
    data_info <- list(
        ncols_numeric = model$metadata$ncols_num, ## is in c++
        ncols_categ = model$metadata$ncols_cat,  ## is in c++
        cols_numeric = as.list(model$metadata$cols_num),
        cols_categ = as.list(model$metadata$cols_cat),
        cat_levels = unname(as.list(model$metadata$cat_levs))
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
            cat_levs   =  metadata$data_info$cat_levels
        ),
        random_seed  =  metadata$params$random_seed,
        nthreads     =  metadata$model_info$nthreads,
        cpp_obj      =  list(
            ptr         =  NULL,
            serialized  =  NULL,
            imp_ptr     =  NULL,
            imp_ser     =  NULL
        )
    )
    
    if (NROW(this$metadata$cat_levels))
        names(this$metadata$cat_levels) <- this$metadata$cols_cat
    
    class(this) <- "isolation_forest"
    return(this)
}
