####
#### THIS SCRIPT FITS A LOGISTIC REGRESSION MODEL AND MAKE PREDICTIONS
####

# Set seed
seed <- ifelse(exists('seed'), seed, 2019)
set.seed(seed)


pipeline_stack <- function(target, train_set, valid_set, test_set,
                         trControl = NULL, tuneGrid = NULL,
                         suffix = NULL, calculate = FALSE, seed = 2019,
                         n_cores = 1){
  
  # Define objects suffix
  suffix <- ifelse(is.null(suffix), NULL, paste0('_', suffix))  
  
  # Default trControl if input is NULL
  if (is.null(trControl)){
    trControl <- trainControl()
  }
  
  # Logistic Regression ----
  if (calculate == TRUE) {
    
    # Set up Multithreading
    # library(doParallel)
    # cl <- makePSOCKcluster(n_cores)
    # clusterEvalQ(cl, library(foreach))
    # registerDoParallel(cl)
    
    print(paste0(
      ifelse(exists('start_time'), paste0('[', round(
        difftime(Sys.time(), start_time, units = 'mins'), 1
      ),
      'm]: '), ''),
      'Starting Stacking Model Fit... ',
      ifelse(is.null(suffix), NULL, paste0(' ', substr(
        suffix, 2, nchar(suffix)
      )))
    ))
    
    control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid", savePredictions = "final", 
                            index = createResample(train_set$y, 5), summaryFunction = twoClassSummary, 
                            classProbs = TRUE, verboseIter = TRUE)
    # Model List Training
    time_fit_start <- Sys.time()
    assign(
      paste0('fit_model_list', suffix),
      caretList( x = train_set[, !names(train_set) %in% c(target)],
                 y = train_set[, target],
                 trControl = control,
                 tuneList=list(
                   xgbTree = caretModelSpec(method="xgbTree", tuneGrid=data.frame(nrounds = 200, 
                                                                                  max_depth = 6,
                                                                                  gamma = 0,
                                                                                  eta = 0.05,
                                                                                  colsample_bytree = 1,
                                                                                  min_child_weight = 3,
                                                                                  subsample = 1)),
                   ranger  = caretModelSpec(method="ranger",    tuneGrid=data.frame(mtry = 4,
                                                                                    splitrule = "gini",
                                                                                    min.node.size = 9)),
                   glm  = caretModelSpec(method="glm")),
                 continue_on_fail = FALSE,
                 preProcess = c('center','scale'))
      , envir = .GlobalEnv) 
    
    # Baseline (25, gini, 1 \\ 150, 3, 0.3, 0.8, 0.75)
    # Clustering (29, gini, 1 \\ 150, 3, 0.4, 0.8, 1)
    # Bining (73, gini, 1 \\ 100, 3, 0.3, 0.6, 1)
    
    
    # Save model list
    time_fit_end <- Sys.time()
    assign(paste0('time_fit_model_list', suffix),
           time_fit_end - time_fit_start, envir = .GlobalEnv)
    saveRDS(get(paste0('fit_model_list', suffix)), paste0('models/fit_model_list', suffix, '.rds'))
    saveRDS(get(paste0('time_fit_model_list', suffix)), paste0('models/time_fit_model_list', suffix, '.rds'))
    
    # Model Training Stacking
    stackControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE, classProbs = TRUE,
                                 verboseIter = TRUE)
    assign(
      paste0('fit_stack_glm', suffix),
      caretStack(
        
        get(paste0('fit_model_list', suffix)), 
        method = "glm", 
        metric = "Accuracy",
        trControl = stackControl)
      
      , envir = .GlobalEnv)
    
    assign(
      paste0('fit_stack_rf', suffix),
      caretStack(
        
        get(paste0('fit_model_list', suffix)), 
        method = "ranger", 
        metric = "Accuracy",
        trControl = stackControl)
      
      , envir = .GlobalEnv)
    
    assign(
      paste0('fit_stack_xgbTree', suffix),
      caretStack(
        
        get(paste0('fit_model_list', suffix)), 
        method = "xgbTree", 
        metric = "Accuracy",
        trControl = stackControl)
      
      , envir = .GlobalEnv)
    
    # Stop Multithreading
    #stopCluster(cl)
    
    # Save model stack
    time_fit_end_0 <- Sys.time()
    assign(paste0('time_fit_stack', suffix),
           time_fit_end_0 - time_fit_start, envir = .GlobalEnv)
    saveRDS(get(paste0('fit_stack_glm', suffix)), paste0('models/fit_stack_glm', suffix, '.rds'))
    saveRDS(get(paste0('fit_stack_rf', suffix)), paste0('models/fit_stack_rf', suffix, '.rds'))
    saveRDS(get(paste0('fit_stack_xgbTree', suffix)), paste0('models/fit_stack_xgbTree', suffix, '.rds'))
    saveRDS(get(paste0('time_fit_stack', suffix)), paste0('models/time_fit_stack_glm', suffix, '.rds'))
  }
  
  # Load Model
  assign(paste0('fit_model_list', suffix),
         readRDS(paste0('models/fit_model_list', suffix, '.rds')), envir = .GlobalEnv)
  assign(paste0('fit_stack_glm', suffix),
         readRDS(paste0('models/fit_stack_glm', suffix, '.rds')), envir = .GlobalEnv)
  assign(paste0('fit_stack_rf', suffix),
         readRDS(paste0('models/fit_stack_rf', suffix, '.rds')), envir = .GlobalEnv)
  assign(paste0('fit_stack_xgbTree', suffix),
         readRDS(paste0('models/fit_stack_xgbTree', suffix, '.rds')), envir = .GlobalEnv)
  assign(paste0('time_fit_stack', suffix),
         readRDS(paste0('models/time_fit_stack_glm', suffix, '.rds')), envir = .GlobalEnv)
  
  # Predicting against Valid Set with transformed target
  assign(paste0('pred_glm_stack', suffix),
         predict(get(paste0('fit_stack_glm', suffix)), valid_set, type = 'prob'), envir = .GlobalEnv)
  assign(paste0('pred_glm_stack_prob', suffix), get(paste0('pred_glm_stack', suffix)), envir = .GlobalEnv)
  assign(paste0('pred_glm_stack', suffix), ifelse(get(paste0('pred_glm_stack', suffix)) > 0.5, yes = 1,no = 0), envir = .GlobalEnv)
  
  assign(paste0('pred_rf_stack', suffix),
         predict(get(paste0('fit_stack_rf', suffix)), valid_set, type = 'prob'), envir = .GlobalEnv)
  assign(paste0('pred_rf_stack_prob', suffix), get(paste0('pred_rf_stack', suffix)), envir = .GlobalEnv)
  assign(paste0('pred_rf_stack', suffix), ifelse(get(paste0('pred_rf_stack', suffix)) > 0.5, yes = 1,no = 0), envir = .GlobalEnv)
  
  assign(paste0('pred_xgbTree_stack', suffix),
         predict(get(paste0('fit_stack_xgbTree', suffix)), valid_set, type = 'prob'), envir = .GlobalEnv)
  assign(paste0('pred_xgbTree_stack_prob', suffix), get(paste0('pred_xgbTree_stack', suffix)), envir = .GlobalEnv)
  assign(paste0('pred_xgbTree_stack', suffix), ifelse(get(paste0('pred_xgbTree_stack', suffix)) > 0.5,yes =  1,no =  0), envir = .GlobalEnv)
  
  # Storing Sensitivity for different thresholds GLM
  sens_temp <- data.frame(rbind(rep_len(0, length(seq(from = 0.05, to = 1, by = 0.05)))))
  temp_cols <- c()
  for (t in seq(from = 0.05, to = 1, by =0.05)){
    temp_cols <- cbind(temp_cols, paste0('t_', format(t, nsmall=2)))
  }
  colnames(sens_temp) <- temp_cols
  
  valid_set[,target] <- ifelse(valid_set[,target]=='No',0,1)
  
  for (t in seq(from = 0.05, to = 1, by = 0.05)){
    assign(paste0('tres_pred_glm_stack', suffix, '_', format(t, nsmall=2)), ifelse(get(paste0('pred_glm_stack_prob', suffix)) > t, no = 0, yes = 1), envir = .GlobalEnv)
    sens_temp[, paste0('t_', format(t, nsmall=2))] <- Sensitivity(y_pred = get(paste0('tres_pred_glm_stack', suffix, '_', format(t, nsmall=2))),
                                                                  y_true = valid_set[,target], positive = '1')
    
  }  
  assign(paste0('sens_temp_glm_stack', suffix), sens_temp, envir = .GlobalEnv)
  
  # Storing Sensitivity for different thresholds RANGER
  sens_temp <- data.frame(rbind(rep_len(0, length(seq(from = 0.05, to = 1, by = 0.05)))))
  temp_cols <- c()
  for (t in seq(from = 0.05, to = 1, by =0.05)){
    temp_cols <- cbind(temp_cols, paste0('t_', format(t, nsmall=2)))
  }
  colnames(sens_temp) <- temp_cols
  
  for (t in seq(from = 0.05, to = 1, by = 0.05)){
    assign(paste0('tres_pred_rf_stack', suffix, '_', format(t, nsmall=2)), ifelse(get(paste0('pred_rf_stack_prob', suffix)) > t, no = 0, yes = 1), envir = .GlobalEnv)
    sens_temp[, paste0('t_', format(t, nsmall=2))] <- Sensitivity(y_pred = get(paste0('tres_pred_rf_stack', suffix, '_', format(t, nsmall=2))),
                                                                  y_true = valid_set[,target], positive = '1')
    
  }  
  assign(paste0('sens_temp_rf_stack', suffix), sens_temp, envir = .GlobalEnv)
  
  # Storing Sensitivity for different thresholds XGBTREE
  sens_temp <- data.frame(rbind(rep_len(0, length(seq(from = 0.05, to = 1, by = 0.05)))))
  temp_cols <- c()
  for (t in seq(from = 0.05, to = 1, by =0.05)){
    temp_cols <- cbind(temp_cols, paste0('t_', format(t, nsmall=2)))
  }
  colnames(sens_temp) <- temp_cols
  
  for (t in seq(from = 0.05, to = 1, by = 0.05)){
    assign(paste0('tres_pred_xgbTree_stack', suffix, '_', format(t, nsmall=2)), ifelse(get(paste0('pred_xgbTree_stack_prob', suffix)) > t, no = 0, yes = 1), envir = .GlobalEnv)
    sens_temp[, paste0('t_', format(t, nsmall=2))] <- Sensitivity(y_pred = get(paste0('tres_pred_xgbTree_stack', suffix, '_', format(t, nsmall=2))),
                                                                  y_true = valid_set[,target], positive = '1')
    
  }  
  assign(paste0('sens_temp_xgbTree_stack', suffix), sens_temp, envir = .GlobalEnv)
  
  
  # Compare Predictions and Valid Set
  assign(paste0('comp_stack', suffix),
         data.frame(obs = valid_set[,target],
                    pred_glm = get(paste0('pred_glm_stack', suffix)),
                    pred_rf = get(paste0('pred_rf_stack', suffix)),
                    pred_xgbTree = get(paste0('pred_xgbTree_stack', suffix))
                    ), envir = .GlobalEnv)
  
  # Generate results with transformed target
  assign(paste0('results', suffix),
         as.data.frame(
           rbind(   cbind(
                   'Accuracy' = Accuracy(y_pred = get(paste0('pred_glm_stack', suffix)),
                                         y_true = valid_set[,target]),
                   'Sensitivity' = Sensitivity(y_pred = get(paste0('pred_glm_stack', suffix)),positive = '1',
                                               y_true = valid_set[,target]),
                   'Precision' = Precision(y_pred = get(paste0('pred_glm_stack', suffix)),positive = '1',
                                           y_true = valid_set[,target]),
                   'Specificity' = Specificity(y_pred = get(paste0('pred_glm_stack', suffix)),positive = '1',
                                     y_true = valid_set[,target]),
                   'F1 Score' = F1_Score(y_pred = get(paste0('pred_glm_stack', suffix)),positive = '1',
                                         y_true = valid_set[,target]),
                   'AUC'      = AUC::auc(AUC::roc(as.numeric(valid_set[,target]), as.factor(get(paste0('pred_glm_stack', suffix))))),
                   'Coefficients' = length(get(paste0('fit_stack_glm', suffix))$ens_model$finalModel$coefficients),
                   'Train Time (min)' = round(as.numeric(get(paste0('time_fit_stack', suffix)), units = 'mins'), 1),
                   'CV | Accuracy' = get(paste0('fit_stack_glm', suffix))$error[, 'Accuracy'],
                   'CV | Kappa' = get(paste0('fit_stack_glm', suffix))$error[, 'Kappa'],
                   'CV | AccuracySD' = get(paste0('fit_stack_glm', suffix))$error[, 'AccuracySD'],
                   'CV | KappaSD' = get(paste0('fit_stack_glm', suffix))$error[, 'KappaSD']),
                   cbind(
                   'Accuracy' = Accuracy(y_pred = get(paste0('pred_rf_stack', suffix)),
                                         y_true = valid_set[,target]),
                   'Sensitivity' = Sensitivity(y_pred = get(paste0('pred_rf_stack', suffix)),positive = '1',
                                               y_true = valid_set[,target]),
                   'Precision' = Precision(y_pred = get(paste0('pred_rf_stack', suffix)),positive = '1',
                                           y_true = valid_set[,target]),
                   'Specificity' = Specificity(y_pred = get(paste0('pred_rf_stack', suffix)),positive = '1',
                                     y_true = valid_set[,target]),
                   'F1 Score' = F1_Score(y_pred = get(paste0('pred_rf_stack', suffix)),positive = '1',
                                         y_true = valid_set[,target]),
                   'AUC'      = AUC::auc(AUC::roc(as.numeric(valid_set[,target]), as.factor(get(paste0('pred_rf_stack', suffix))))),
                   'Coefficients' = length(get(paste0('fit_stack_rf', suffix))$ens_model$finalModel$num.independent.variables),
                   'Train Time (min)' = round(as.numeric(get(paste0('time_fit_stack', suffix)), units = 'mins'), 1),
                   'CV | Accuracy' = mean(get(paste0('fit_stack_rf', suffix))$error[, 'Accuracy']),
                   'CV | Kappa' = mean(get(paste0('fit_stack_rf', suffix))$error[, 'Kappa']),
                   'CV | AccuracySD' = mean(get(paste0('fit_stack_rf', suffix))$error[, 'AccuracySD']),
                   'CV | KappaSD' = mean(get(paste0('fit_stack_rf', suffix))$error[, 'KappaSD']))
                   ,cbind(
                   'Accuracy' = Accuracy(y_pred = get(paste0('pred_xgbTree_stack', suffix)),
                                         y_true = valid_set[,target]),
                   'Sensitivity' = Sensitivity(y_pred = get(paste0('pred_xgbTree_stack', suffix)),positive = '1',
                                               y_true = valid_set[,target]),
                   'Precision' = Precision(y_pred = get(paste0('pred_xgbTree_stack', suffix)),positive = '1',
                                           y_true = valid_set[,target]),
                   'Specificity' = Specificity(y_pred = get(paste0('pred_xgbTree_stack', suffix)),positive = '1',
                                     y_true = valid_set[,target]),
                   'F1 Score' = F1_Score(y_pred = get(paste0('pred_xgbTree_stack', suffix)),positive = '1',
                                         y_true = valid_set[,target]),
                   'AUC'      = AUC::auc(AUC::roc(as.numeric(valid_set[,target]), as.factor(get(paste0('pred_xgbTree_stack', suffix))))),
                   'Coefficients' = length(get(paste0('fit_stack_xgbTree', suffix))$ens_model$finalModel$nfeatures),
                   'Train Time (min)' = round(as.numeric(get(paste0('time_fit_stack', suffix)), units = 'mins'), 1),
                   'CV | Accuracy' = mean(get(paste0('fit_stack_xgbTree', suffix))$error[, 'Accuracy']),
                   'CV | Kappa' = mean(get(paste0('fit_stack_xgbTree', suffix))$error[, 'Kappa']),
                   'CV | AccuracySD' = mean(get(paste0('fit_stack_xgbTree', suffix))$error[, 'AccuracySD']),
                   'CV | KappaSD' = mean(get(paste0('fit_stack_xgbTree', suffix))$error[, 'KappaSD']))

           ), envir = .GlobalEnv
         )
  )

  # Generate all_results table | with CV and transformed target
  results_title_glm <- paste0('Stacking glm', ifelse(is.null(suffix), NULL, paste0(' ', substr(suffix,2, nchar(suffix)))))
  results_title_rf <- paste0('Stacking rf', ifelse(is.null(suffix), NULL, paste0(' ', substr(suffix,2, nchar(suffix)))))
  results_title_xgb <- paste0('Stacking xgb', ifelse(is.null(suffix), NULL, paste0(' ', substr(suffix,2, nchar(suffix)))))
  
  if (exists('all_results')){
    a <- rownames(all_results)
    assign('all_results', rbind(all_results, get(paste0('results', suffix))), envir = .GlobalEnv)
    rownames(all_results) <- c(a, results_title_glm, results_title_rf, results_title_xgb)
    assign('all_results', all_results, envir = .GlobalEnv)
  } else{
    assign('all_results', rbind(get(paste0('results', suffix))), envir = .GlobalEnv)
    rownames(all_results) <- c(results_title_glm, results_title_rf, results_title_xgb)
    assign('all_results', all_results, envir = .GlobalEnv)
  }
  
  # TO FIX - NOT SAVING PROPERLY...
  # Save Variables Importance plot
  # png(
  #   paste0('plots/fit_stack_glm_', ifelse(is.null(suffix), NULL, paste0(substr(suffix,2, nchar(suffix)), '_')), 'varImp.png'),
  #   width = 1500,
  #   height = 1000
  # )
  # p <- plot(varImp(get(paste0('fit_stack_glm', suffix))), top = 30)
  # p
  # dev.off()
  
  # Predicting against Test Set
  assign(paste0('pred_glm_stack_test', suffix), predict(get(paste0('fit_stack_glm', suffix)), test_set), envir = .GlobalEnv)
  assign(paste0('pred_rf_stack_test', suffix), predict(get(paste0('fit_stack_rf', suffix)), test_set), envir = .GlobalEnv)
  assign(paste0('pred_xgbTree_stack_test', suffix), predict(get(paste0('fit_stack_xgbTree', suffix)), test_set), envir = .GlobalEnv)
  
  submissions_test <- as.data.frame(cbind(
    get(paste0('pred_glm_stack_test', suffix)),
    get(paste0('pred_rf_stack_test', suffix)),
    get(paste0('pred_xgbTree_stack_test', suffix))# To adjust if target is transformed
  ))
  
  colnames(submissions_test) <- c('stack_glm', 'stack_rf', 'stack_xgbTree')
  submissions_test[,'stack_glm'] <- ifelse(submissions_test[,'stack_glm']==2,0,1)
  submissions_test[,'stack_rf'] <- ifelse(submissions_test[,'stack_rf']==2,0,1)
  submissions_test[,'stack_xgbTree'] <- ifelse(submissions_test[,'stack_xgbTree']==2,0,1)
  assign(paste0('submission_stack_test', suffix), submissions_test, envir = .GlobalEnv)
  
  # Generating submissions file
  write.csv(get(paste0('submission_stack_test', suffix)),
            paste0('submissions/submission_stack', suffix, '.csv'),
            row.names = FALSE)
  
  
  # Predicting against Valid Set with original target
  assign(paste0('pred_stack_glm_valid', suffix), predict(get(paste0('fit_stack_glm', suffix)), valid_set), envir = .GlobalEnv)
  assign(paste0('pred_stack_rf_valid', suffix), predict(get(paste0('fit_stack_rf', suffix)), valid_set), envir = .GlobalEnv)
  assign(paste0('pred_stack_xgbTree_valid', suffix), predict(get(paste0('fit_stack_xgbTree', suffix)), valid_set), envir = .GlobalEnv)
  
  # Generate real_results with original target
  submissions_valid <- as.data.frame(cbind(
    get(paste0('pred_stack_glm_valid', suffix)),
    get(paste0('pred_stack_rf_valid', suffix)),
    get(paste0('pred_stack_xgbTree_valid', suffix)) # To adjust if target is transformed
  ))
  colnames(submissions_valid) <- c('stack_glm', 'stack_rf', 'stack_xgbTree')
  submissions_valid[,'stack_glm'] <- ifelse(submissions_valid[,'stack_glm']=='1',0,1)
  submissions_valid[,'stack_rf'] <- ifelse(submissions_valid[,'stack_rf']=='1',0,1)
  submissions_valid[,'stack_xgbTree'] <- ifelse(submissions_valid[,'stack_xgbTree']=='1',0,1)
  assign(paste0('submission_stack_valid', suffix), submissions_valid, envir = .GlobalEnv)
  
  assign(paste0('real_results', suffix), as.data.frame(rbind(
    cbind(
    'Accuracy' = Accuracy(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_glm'], y_true = as.numeric(valid_set[, c(target)])),
    'Sensitivity' = Sensitivity(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_glm'], y_true = as.numeric(valid_set[, c(target)]),positive = '1'),
    'Precision' = Precision(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_glm'], y_true = as.numeric(valid_set[, c(target)]),positive = '1'),
    'Specificity' = Specificity(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_glm'], y_true = as.numeric(valid_set[, c(target)]),positive = '1'),
    'F1 Score' = F1_Score(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_glm'], y_true = as.numeric(valid_set[, c(target)]),positive = '1'),
    'AUC'      = AUC::auc(AUC::roc(as.numeric(valid_set[, c(target)]), as.factor(get(paste0('submission_stack_valid', suffix))[, 'stack_glm']))),
    'Coefficients' = length(get(paste0('fit_stack_glm', suffix))$ens_model$finalModel$coefficients),
    'Train Time (min)' = round(as.numeric(get(paste0('time_fit_stack', suffix)), units = 'mins'), 1)),

    cbind(
      'Accuracy' = Accuracy(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_rf'], y_true = as.numeric(valid_set[, c(target)])),
      'Sensitivity' = Sensitivity(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_rf'], y_true = as.numeric(valid_set[, c(target)]),positive = '1'),
      'Precision' = Precision(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_rf'], y_true = as.numeric(valid_set[, c(target)]),positive = '1'),
      'Specificity' = Specificity(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_rf'], y_true = as.numeric(valid_set[, c(target)]),positive = '1'),
      'F1 Score' = F1_Score(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_rf'], y_true = as.numeric(valid_set[, c(target)]),positive = '1'),
      'AUC'      = AUC::auc(AUC::roc(as.numeric(valid_set[, c(target)]), as.factor(get(paste0('submission_stack_valid', suffix))[, 'stack_rf']))),
      'Coefficients' = length(get(paste0('fit_stack_rf', suffix))$ens_model$finalModel$num.independent.variables),
      'Train Time (min)' = round(as.numeric(get(paste0('time_fit_stack', suffix)), units = 'mins'), 1)),
      
    cbind(
      'Accuracy' = Accuracy(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_xgbTree'], y_true = as.numeric(valid_set[, c(target)])),
      'Sensitivity' = Sensitivity(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_xgbTree'], y_true = as.numeric(valid_set[, c(target)]),positive = '1'),
      'Precision' = Precision(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_xgbTree'], y_true = as.numeric(valid_set[, c(target)]),positive = '1'),
      'Specificity' = Specificity(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_xgbTree'], y_true = as.numeric(valid_set[, c(target)]),positive = '1'),
      'F1 Score' = F1_Score(y_pred = get(paste0('submission_stack_valid', suffix))[, 'stack_xgbTree'], y_true = as.numeric(valid_set[, c(target)]),positive = '1'),
      'AUC'      = AUC::auc(AUC::roc(as.numeric(valid_set[, c(target)]), as.factor(get(paste0('submission_stack_valid', suffix))[, 'stack_xgbTree']))),
      'Coefficients' = length(get(paste0('fit_stack_xgbTree', suffix))$finalModel$nfeatures),
      'Train Time (min)' = round(as.numeric(get(paste0('time_fit_stack', suffix)), units = 'mins'), 1)
        )
      )
    )
  )
  
  # Generate all_real_results table with original target
  if (exists('all_real_results')){
    a <- rownames(all_real_results)
    assign('all_real_results', rbind(all_real_results, results_title = get(paste0('real_results', suffix))), envir = .GlobalEnv)
    rownames(all_real_results) <- c(a, results_title_glm, results_title_rf, results_title_xgb)
    assign('all_real_results', all_real_results, envir = .GlobalEnv)
  } else{
    assign('all_real_results', rbind(results_title = get(paste0('real_results', suffix))), envir = .GlobalEnv)
    rownames(all_real_results) <- c(results_title_glm, results_title_rf, results_title_xgb)
    assign('all_real_results', all_real_results, envir = .GlobalEnv)
  }
  
  # PLOT ROC
  roc_glm <- AUC::roc(as.factor(valid_set[, c(target)]), as.factor(get(paste0('submission_stack_valid', suffix))[, 'stack_glm']))
  assign(paste0('roc_object_stack_glm', suffix), roc_glm,  envir = .GlobalEnv)
  roc_glm <- AUC::roc(as.factor(valid_set[, c(target)]), as.factor(get(paste0('submission_stack_valid', suffix))[, 'stack_rf']))
  assign(paste0('roc_object_stack_rf', suffix), roc_glm,  envir = .GlobalEnv)
  roc_glm <- AUC::roc(as.factor(valid_set[, c(target)]), as.factor(get(paste0('submission_stack_valid', suffix))[, 'stack_xgbTree']))
  assign(paste0('roc_object_stack_xgbTree', suffix), roc_glm,  envir = .GlobalEnv)
  
  # plot(get(paste0('roc_object_stack_glm', suffix)), col=color4, lwd=4, main="ROC Curve Stack glm")
  # plot(get(paste0('roc_object_stack_rf', suffix)), col=color4, lwd=4, main="ROC Curve Stack rf")
  # plot(get(paste0('roc_object_stack_xgbTree', suffix)), col=color4, lwd=4, main="ROC Curve Stack xgbTree")
  
  # Density Plot
  prob_glm <- data.frame(No = get(paste0('pred_glm_stack_prob', suffix)), Yes = 1 - get(paste0('pred_glm_stack_prob', suffix)))
  prob_glm<- melt(prob_glm)
  assign(paste0('density_plot_stack_glm', suffix), ggplot(prob_glm,aes(x=value, fill=variable)) + geom_density(alpha=0.25)+
           theme_tufte(base_size = 5, ticks=F)+ 
           ggtitle(paste0('Density Plot Stack glm', suffix))+
           theme(plot.margin = unit(c(10,10,10,10),'pt'),
                 axis.title=element_blank(),
                 axis.text = element_text(colour = color2, size = 9, family = font2),
                 axis.text.x = element_text(hjust = 1, size = 9, family = font2),
                 plot.title = element_text(size = 15, face = "bold", hjust = 0.5), 
                 plot.background = element_rect(fill = color1)), envir = .GlobalEnv)
  
  prob_glm <- data.frame(No = get(paste0('pred_rf_stack_prob', suffix)), Yes = 1 - get(paste0('pred_rf_stack_prob', suffix)))
  prob_glm<- melt(prob_glm)
  assign(paste0('density_plot_stack_rf', suffix), ggplot(prob_glm,aes(x=value, fill=variable)) + geom_density(alpha=0.25)+
           theme_tufte(base_size = 5, ticks=F)+ 
           ggtitle(paste0('Density Plot Stack RF', suffix))+
           theme(plot.margin = unit(c(10,10,10,10),'pt'),
                 axis.title=element_blank(),
                 axis.text = element_text(colour = color2, size = 9, family = font2),
                 axis.text.x = element_text(hjust = 1, size = 9, family = font2),
                 plot.title = element_text(size = 15, face = "bold", hjust = 0.5), 
                 plot.background = element_rect(fill = color1)), envir = .GlobalEnv)
  
  prob_glm <- data.frame(No = get(paste0('pred_xgbTree_stack_prob', suffix)), Yes = 1 - get(paste0('pred_xgbTree_stack_prob', suffix)))
  prob_glm<- melt(prob_glm)
  assign(paste0('density_plot_stack_xgbTree', suffix), ggplot(prob_glm,aes(x=value, fill=variable)) + geom_density(alpha=0.25)+
           theme_tufte(base_size = 5, ticks=F)+ 
           ggtitle(paste0('Density Plot Stack xgbTree', suffix))+
           theme(plot.margin = unit(c(10,10,10,10),'pt'),
                 axis.title=element_blank(),
                 axis.text = element_text(colour = color2, size = 9, family = font2),
                 axis.text.x = element_text(hjust = 1, size = 9, family = font2),
                 plot.title = element_text(size = 15, face = "bold", hjust = 0.5), 
                 plot.background = element_rect(fill = color1)), envir = .GlobalEnv)

  
  # Confusion Matrix
  assign(paste0('cm_stack_glm', suffix), confusionMatrix(as.factor(get(paste0('submission_stack_valid', suffix))[, 'stack_glm']), as.factor(valid_set[, c(target)])), envir = .GlobalEnv)
  # cm_plot_glm <- fourfoldplot(cm_glm$table)
  # assign(paste0('cm_plot_stack_glm', suffix), cm_plot_glm, envir = .GlobalEnv)
  # get(paste0('cm_plot_stack_glm', suffix))
  
  assign(paste0('cm_stack_rf', suffix), confusionMatrix(as.factor(get(paste0('submission_stack_valid', suffix))[, 'stack_rf']), as.factor(valid_set[, c(target)])), envir = .GlobalEnv)
  # cm_plot_glm <- fourfoldplot(cm_glm$table)
  # assign(paste0('cm_plot_stack_rf', suffix), cm_plot_glm, envir = .GlobalEnv)
  # get(paste0('cm_plot_stack_rf', suffix))
  
  assign(paste0('cm_stack_xgbTree', suffix), confusionMatrix(as.factor(get(paste0('submission_stack_valid', suffix))[, 'stack_xgbTree']), as.factor(valid_set[, c(target)])), envir = .GlobalEnv)
  # cm_plot_glm <- fourfoldplot(cm_glm$table)
  # assign(paste0('cm_plot_stack_xgbTree', suffix), cm_plot_glm, envir = .GlobalEnv)
  # get(paste0('cm_plot_stack_xgbTree', suffix))

  # List of files for Dashboard
  assign(paste0('files_glm', suffix), as.data.frame(cbind(
    'model_file' = paste0('fit_stack_glm', suffix),
    'cm_file' = paste0('cm_stack_glm', suffix),
    'roc' = paste0('roc_object_stack_glm', suffix),
    'density' = paste0('density_plot_stack_glm', suffix)
  )), envir = .GlobalEnv)
  
  assign(paste0('files_rf', suffix), as.data.frame(cbind(
    'model_file' = paste0('fit_stack_rf', suffix),
    'cm_file' = paste0('cm_stack_rf', suffix),
    'roc' = paste0('roc_object_stack_rf', suffix),
    'density' = paste0('density_plot_stack_rf', suffix)
  )), envir = .GlobalEnv)
  
  assign(paste0('files_xgbTree', suffix), as.data.frame(cbind(
    'model_file' = paste0('fit_stack_xgbTree', suffix),
    'cm_file' = paste0('cm_stack_xgbTree', suffix),
    'roc' = paste0('roc_object_stack_xgbTree', suffix),
    'density' = paste0('density_plot_stack_xgbTree', suffix)
  )), envir = .GlobalEnv)
  
  if (exists('file_list')){
    a <- rownames(file_list)
    assign('file_list', rbind(file_list, 'model_file_1' = get(paste0('files_glm', suffix)), 'model_file_2' = get(paste0('files_rf', suffix)), 'model_file_3' = get(paste0('files_xgbTree', suffix))), envir = .GlobalEnv)
    rownames(file_list) <- c(a, results_title_glm, results_title_rf, results_title_xgb)
    assign('file_list', file_list, envir = .GlobalEnv)
  } else{
    a <- rownames(file_list)
    assign('file_list', rbind('model_file_1' = get(paste0('files_glm', suffix)), 'model_file_2' = get(paste0('files_rf', suffix)), 'model_file_3' = get(paste0('files_xgbTree', suffix))), envir = .GlobalEnv)
    rownames(file_list) <- c(a, results_title_glm, results_title_rf, results_title_xgb)
    assign('file_list', file_list, envir = .GlobalEnv)
  }
  
  
    
  print(paste0(
    ifelse(exists('start_time'), paste0('[', round(
      difftime(Sys.time(), start_time, units = 'mins'), 1
    ), 'm]: '), ''),
    'Stacking with done!', ifelse(is.null(suffix), NULL, paste0(' ', substr(suffix,2, nchar(suffix))))))
}
