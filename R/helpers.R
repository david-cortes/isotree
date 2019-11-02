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
    return(c("matrix", "dgTMatrix"))
}

get.types.spmat <- function(allow_csr = FALSE, allow_csc = TRUE) {
    outp <- c("dgCMatrix")
    if (allow_csc) outp <- c(outp, "matrix.csc")
    if (allow_csr) outp <- c(outp, "matrix.csr")
    return(outp)
}

process.data <- function(df, sample_weights = NULL, column_weights = NULL) {
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
    if ( max(class(df) %in% dmatrix_types) ) { outp$X_num <- unname(as.numeric(df)) ; return(outp) }
    
    ### Sparse matrix
    if ( max(class(df) %in% spmatrix_types) ) {
        
        if ("dgCMatrix" %in% class(df)) {
            ### From package 'Matrix'
            outp$Xc         <-  as.numeric(df@x)
            outp$Xc_ind     <-  as.integer(df@i)
            outp$Xc_indptr  <-  as.integer(df@p)
        } else {
            ### From package 'SparseM'
            outp$Xc         <-  as.numeric(df@ra)
            outp$Xc_ind     <-  as.integer(df@ia - 1)
            outp$Xc_indptr  <-  as.integer(df@ja - 1)
        }
        
        return(outp)
    }
    
    ### Data Frame
    if ( max(class(df) %in% "data.frame") ) {
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
            is_num          <-  unname(as.logical(sapply(df, function(x) max(class(x) %in% dtypes_num))))
            outp$cols_num   <-  names(df)[is_num]
            outp$ncols_num  <-  as.integer(sum(is_num))
            outp$X_num      <-  unname(as.numeric(as.matrix(as.data.frame(lapply(df[, is_num, drop = FALSE], as.numeric)))))
        } else { outp$ncols_num <- as.integer(0) }
        
        if (any(df_coltypes %in% dtypes_cat)) {
            is_cat          <-  unname(as.logical(sapply(df, function(x) max(class(x) %in% dtypes_cat))))
            outp$cols_cat   <-  names(df)[is_cat]
            outp$ncols_cat  <-  as.integer(sum(is_cat))
            outp$X_cat      <-  as.data.frame(lapply(df[, is_cat, drop = FALSE], factor))
            outp$cat_levs   <-  lapply(outp$X_cat, levels)
            outp$ncat       <-  sapply(outp$cat_levs, NROW)
            outp$X_cat      <-  as.data.frame(lapply(outp$X_cat, function(x) ifelse(is.na(x), -1, as.integer(x) - 1)))
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
    if ( NCOL(df) < (metadata$ncols_num + metadata$ncols_cat) )
        stop(sprintf("Input data contains fewer columns than expected (%d vs. %d)",
                     NCOL(df), (metadata$ncols_num + metadata$ncols_cat)))
    df  <-  cast.df.alike(df)
    if (metadata$ncols_cat > 0 && !("data.frame" %in% class(df)))
        stop("Model was fit to categorical data, must pass a data.frame with new data.")
    
    dmatrix_types     <-  get.types.dmat()
    spmatrix_types    <-  get.types.spmat(allow_csr = allow_csr, allow_csc = allow_csc)
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
                outp$X_cat <- as.data.frame(lapply(outp$X_cat, function(x) ifelse(is.na(x), -1, as.integer(x) - 1)))
                outp$X_cat <- unname(as.integer(as.matrix(outp$X_cat)))
            }
            
        }
        
    } else {
        
        if (NCOL(df) != (metadata$ncols_num + metadata$ncols_cat))
            stop(sprintf("Input data has %d columns, but model was fit to data with %d columns.",
                         NCOL(df), (metadata$ncols_num + metadata$ncols_cat)))
        if (!any(class(df) %in% spmatrix_types)) {
            outp$X_num <- as.numeric(df)
        } else {
            if ("dgCMatrix" %in% class(df)) {
                ### From package 'Matrix'
                outp$Xc         <-  as.numeric(df@x)
                outp$Xc_ind     <-  as.integer(df@i)
                outp$Xc_indptr  <-  as.integer(df@p)
            } else {
                ### From package 'SparseM'
                if ("matrix.csc" %in% class(df)) {
                    outp$Xc         <-  as.numeric(df@ra)
                    outp$Xc_ind     <-  as.integer(df@ia - 1)
                    outp$Xc_indptr  <-  as.integer(df@ja - 1)
                } else {
                    outp$Xr         <-  as.numeric(df@ra)
                    outp$Xr_ind     <-  as.integer(df@ia - 1)
                    outp$Xr_indptr  <-  as.integer(df@ja - 1)
                }
            }
        }
        
    }
    
    return(outp)
}

reconstruct.from.imp <- function(imputed_num, imputed_cat, df, model) {
    
    if ("dgCMatrix" %in% class(df)) {
        outp     <-  df
        outp@x   <-  imputed_num
        return(outp)
    } else if ("matrix.csc" %in% class(df)) {
        outp     <-  df
        outp@ra  <-  imputed_num
    } else if (!("data.frame" %in% class(df))) {
        return(matrix(imputed_num, nrow = NROW(df)))
    } else {
        df_num <- as.data.frame(matrix(imputed_num, nrow = NROW(df)))
        names(df_num) <- model$metadata$cols_num
        
        df_cat <- as.data.frame(matrix(ifelse(imputed_cat < 0, NA, imputed_cat) + 1, nrow = NROW(df)))
        names(df_cat) <- model$metadata$cols_cat
        df_cat <- as.data.frame(mapply(function(x, levs) factor(x, labels = levs),
                                       df_cat, model$metadata$cat_levs,
                                       SIMPLIFY = FALSE))
        
        df_merged <- cbind(df_num, df_cat)
        df_merged <- df_merged[, names(df)]
        return(df_merged)
    }
}

