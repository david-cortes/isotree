convert.from.old.format.to.new <- function(model) {
    new_format <- list(
        model    =  list(ptr = methods::new("externalptr"), ser = NULL),
        imputer  =  list(ptr = methods::new("externalptr"), ser = NULL),
        indexer  =  list(ptr = methods::new("externalptr"), ser = NULL)
    )

    check.valid.ser(model$cpp_obj$serialized)
    new_format$model$ser <- model$cpp_obj$serialized
    
    if (check.blank.pointer(model$cpp_obj$ptr)) {
        if (model$params$ndim == 1)
            new_format$model$ptr <- deserialize_IsoForest(model$cpp_obj$serialized)
        else
            new_format$model$ptr <- deserialize_ExtIsoForest(model$cpp_obj$serialized)
    } else {
        new_format$model$ptr <- model$cpp_obj$ptr
    }

    if (model$params$build_imputer) {
        check.valid.ser(model$cpp_obj$imp_ser)
        new_format$imputer$ser <- model$cpp_obj$imp_ser

        if (check.blank.pointer(model$cpp_obj$imp_ptr)) {
            new_format$imputer$ptr <- deserialize_Imputer(model$cpp_obj$imp_ser)
        } else {
            new_format$imputer$ptr <- model$cpp_obj$imp_ptr
        }
    }

    if (("indexer" %in% names(model$cpp_obj)) && NROW(model$cpp_obj$ind_ser)) {
        check.valid.ser(model$cpp_obj$ind_ser)
        new_format$indexer$ser <- model$cpp_obj$ind_ser

        if (check.blank.pointer(model$cpp_obj$indexer)) {
            new_format$indexer$ptr <- deserialize_Indexer(model$cpp_obj$ind_ser)
        } else {
            new_format$indexer$ptr <- model$cpp_obj$indexer
        }
    }

    set.list.elt(model, "cpp_objects", new_format)
    set.list.elt(model, "cpp_obj", NULL)
    return(model)
}
