####
#### THIS SCRIPT CALLS ALL SUB-SCRIPTS TO READ AND PREPARE THE DATASET,
#### RUN THE ANALYSIS AND OUTPUT RELEVANT DATA FILES
####

start_time <- Sys.time()
print(paste0('---START--- Starting at ', start_time))

options(warn = 0) # -1 to hide the warnings, 0 to show them
seed <- 2019
set.seed(seed)

# Install Necessary Packages ----
source('scripts/install_packages.R')

# Read and Format Dataset ----
source('scripts/read_format_data.R')

# Split and Preprocess Dataset ----
source('scripts/split_n_preproc.R')

# Model Pipelines ----
source('scripts/model_glm.R')
source('scripts/model_xgbTree.R')
source('scripts/model_ranger.R')
source('scripts/model_stacking.R')

# Parameters of Baseline ----
source('scripts/param_modeling.R')

# Baseline Logistic Regression ----
pipeline_glm(target = 'y', train_set = bank_train_A_proc_dum,
             valid_set = bank_train_B_proc_dum, test_set = bank_test_proc_dum,
             trControl = fitControl, tuneGrid = NULL,
             suffix = 'baseline', calculate = FALSE, seed = seed,
             n_cores = detectCores()-1)

# Baseline XGBoost ----
pipeline_xgbTree(target = 'y', train_set = bank_train_A_proc_dum,
                 valid_set = bank_train_B_proc_dum, test_set = bank_test_proc_dum,
                 trControl = fitControl, tuneGrid = NULL,
                 suffix = 'baseline', calculate = FALSE, seed = seed,
                 n_cores = detectCores()-1)

# Baseline Ranger ----
pipeline_ranger(target = 'y', train_set = bank_train_A_proc_dum,
                 valid_set = bank_train_B_proc_dum, test_set = bank_test_proc_dum,
                 trControl = fitControl, tuneGrid = NULL,
                 suffix = 'baseline', calculate = FALSE, seed = seed,
                 n_cores = detectCores()-1)

# Feature Engineering Clustering ----
calculate <- FALSE
source('scripts/fe_clusters.R')

# Logistic Regression Clustering ----
pipeline_glm(target = 'y', train_set = bank_train_A_FE1,
             valid_set = bank_train_B_FE1, test_set = bank_test_FE1,
             trControl = fitControl, tuneGrid = NULL,
             suffix = 'FE1 Clustering', calculate = FALSE, seed = seed,
             n_cores = detectCores()-1)

# XGBoost Clustering ----
pipeline_xgbTree(target = 'y', train_set = bank_train_A_FE1,
             valid_set = bank_train_B_FE1, test_set = bank_test_FE1,
             trControl = fitControl, tuneGrid = NULL,
             suffix = 'FE1 Clustering', calculate = FALSE, seed = seed,
             n_cores = detectCores()-1)

# Ranger Clustering ----
pipeline_ranger(target = 'y', train_set = bank_train_A_FE1,
                valid_set = bank_train_B_FE1, test_set = bank_test_FE1,
                trControl = fitControl, tuneGrid = NULL,
                suffix = 'FE1 Clustering', calculate = FALSE, seed = seed,
                n_cores = detectCores()-1)

# Feature Engineering Binning ----
source('scripts/fe_binning.R')

# Logistic Regression Binning ----
pipeline_glm(target = 'y', train_set = bank_train_A_FE2,
             valid_set = bank_train_B_FE2, test_set = bank_test_FE2,
             trControl = fitControl, tuneGrid = NULL,
             suffix = 'FE2 Binning', calculate = FALSE, seed = seed,
             n_cores = detectCores()-1)

# XGBoost Binning ----
pipeline_xgbTree(target = 'y', train_set = bank_train_A_FE2,
                 valid_set = bank_train_B_FE2, test_set = bank_test_FE2,
                 trControl = fitControl, tuneGrid = NULL,
                 suffix = 'FE2 Binning', calculate = FALSE, seed = seed,
                 n_cores = detectCores()-1)

# Ranger Binning ----
pipeline_ranger(target = 'y', train_set = bank_train_A_FE2,
                valid_set = bank_train_B_FE2, test_set = bank_test_FE2,
                trControl = fitControl, tuneGrid = NULL,
                suffix = 'FE2 Binning', calculate = FALSE, seed = seed,
                n_cores = detectCores()-1)

# Feature Selection Lasso ----
source('scripts/featsel_lasso.R')

# Feature Selection RFE ----
calculate <- FALSE
source('scripts/featsel_rfe.R')

# Logistic Regression Post RFE ----
pipeline_glm(target = 'y', train_set = bank_train_A_rfe,
                 valid_set = bank_train_B_rfe, test_set = bank_test_rfe,
                 trControl = fitControl, tuneGrid = NULL,
                 suffix = 'RFE', calculate = FALSE, seed = seed,
                 n_cores = detectCores()-1)

# XGBoost Post RFE ----
pipeline_xgbTree(target = 'y', train_set = bank_train_A_rfe,
                 valid_set = bank_train_B_rfe, test_set = bank_test_rfe,
                 trControl = fitControl, tuneGrid = NULL,
                 suffix = 'RFE', calculate = FALSE, seed = seed,
                 n_cores = detectCores()-1)

# XGBoost Tuning ----
pipeline_xgbTree(target = 'y', train_set = bank_train_A_rfe,
                 valid_set = bank_train_B_rfe, test_set = bank_test_rfe,
                 trControl = tuningControl, tuneGrid = xgb_grid,
                 suffix = 'Tuning', calculate = FALSE, seed = seed,
                 n_cores = detectCores()-1)

# Ranger Tuning ----
pipeline_ranger(target = 'y', train_set = bank_train_A_rfe,
                 valid_set = bank_train_B_rfe, test_set = bank_test_rfe,
                 trControl = tuningControl, tuneGrid = ranger_grid,
                 suffix = 'Tuning', calculate = FALSE, seed = seed,
                 n_cores = detectCores()-1)

# Create Predictions Correlation Matrix ----
source('scripts/pred_corr.R')

# Baseline Stacking Logistic Regression | Ranger | xgbTree ----
pipeline_stack(target = 'y', train_set = bank_train_A_proc_dum,
                valid_set = bank_train_B_proc_dum, test_set = bank_test_proc_dum,
                trControl = fitControl, tuneGrid = NULL,
                suffix = 'baseline', calculate = FALSE, seed = seed,
                n_cores = detectCores()-1)

# FE1 Stacking Logistic Regression | Ranger | xgbTree ----
pipeline_stack(target = 'y', train_set = bank_train_A_FE1,
               valid_set = bank_train_B_FE1, test_set = bank_test_FE1,
               trControl = fitControl, tuneGrid = NULL,
               suffix = 'clustering', calculate = FALSE, seed = seed,
               n_cores = detectCores()-1)

# FE2 Stacking Logistic Regression | Ranger | xgbTree ----
pipeline_stack(target = 'y', train_set = bank_train_A_FE2,
               valid_set = bank_train_B_FE2, test_set = bank_test_FE2,
               trControl = fitControl, tuneGrid = NULL,
               suffix = 'binning', calculate = FALSE, seed = seed,
               n_cores = detectCores()-1)

# RFE Stacking Logistic Regression | Ranger | xgbTree ----
pipeline_stack(target = 'y', train_set = bank_train_A_rfe,
               valid_set = bank_train_B_rfe, test_set = bank_test_rfe,
               trControl = fitControl, tuneGrid = NULL,
               suffix = 'Tuning', calculate = FALSE, seed = seed,
               n_cores = detectCores()-1)

# Creating the table with the sensitivities for different thresholds
source('scripts/sensitivity_thresholds.R')


# Save RData for RMarkdown ----
save(
  list = c(
    'raw_train',
    'raw_test',
    'bank_train',
    'bank_test',
    'bank_train_A',
    'bank_train_B',
    'bank_test',
    'bank_train_A_FE2',
    'all_results',
    'all_real_results',
    'opt_nb_clusters',
    'silhouette_9',
    'kmeans_9',
    'roc_object_glm_baseline',
    'roc_object_ranger_baseline',
    'roc_object_xgbTree_baseline',
    'cm_glm_baseline',
    'cm_glm_FE1 Clustering',
    'cm_glm_FE2 Binning',
    'cm_glm_RFE',
    'cm_ranger_baseline',
    'cm_ranger_FE1 Clustering',
    'cm_ranger_FE2 Binning',
    'cm_ranger_Tuning',
    'cm_xgbTree_baseline',
    'cm_xgbTree_FE1 Clustering',
    'cm_xgbTree_FE2 Binning',
    'cm_xgbTree_RFE',
    'cm_xgbTree_Tuning',
    'density_plot_glm_baseline',
    'density_plot_glm_FE1 Clustering',
    'density_plot_glm_FE2 Binning',
    'density_plot_glm_RFE',
    'density_plot_ranger_baseline',
    'density_plot_ranger_FE1 Clustering',
    'density_plot_ranger_FE2 Binning',
    'density_plot_ranger_Tuning',
    'density_plot_xgbTree_baseline',
    'density_plot_xgbTree_FE1 Clustering',
    'density_plot_xgbTree_FE2 Binning',
    'density_plot_xgbTree_RFE',
    'density_plot_xgbTree_Tuning',
    'density_plot_stack_glm_baseline',
    'density_plot_stack_glm_binning',
    'density_plot_stack_glm_clustering',
    'density_plot_stack_glm_Tuning',
    'density_plot_stack_rf_baseline',
    'density_plot_stack_rf_binning',
    'density_plot_stack_rf_clustering',
    'density_plot_stack_rf_Tuning',
    'density_plot_stack_xgbTree_baseline',
    'density_plot_stack_xgbTree_binning',
    'density_plot_stack_xgbTree_clustering',
    'density_plot_stack_xgbTree_Tuning',
    'roc_object_glm_baseline',
    'roc_object_glm_FE1 Clustering',
    'roc_object_glm_FE2 Binning',
    'roc_object_glm_RFE',
    'roc_object_ranger_baseline',
    'roc_object_ranger_FE1 Clustering',
    'roc_object_ranger_FE2 Binning',
    'roc_object_ranger_Tuning',
    'roc_object_stack_glm_baseline',
    'roc_object_stack_glm_binning',
    'roc_object_stack_glm_clustering',
    'roc_object_stack_glm_Tuning',
    'roc_object_stack_rf_baseline',
    'roc_object_stack_rf_binning',
    'roc_object_stack_rf_clustering',
    'roc_object_stack_rf_Tuning',
    'roc_object_stack_xgbTree_baseline',
    'roc_object_stack_xgbTree_binning',
    'roc_object_stack_xgbTree_clustering',
    'roc_object_stack_xgbTree_Tuning',
    'roc_object_xgbTree_baseline',
    'roc_object_xgbTree_FE1 Clustering',
    'roc_object_xgbTree_FE2 Binning',
    'roc_object_xgbTree_RFE',
    'roc_object_xgbTree_Tuning',
    'varsSelected',
    'varsNotSelected',
    'var_sel_rfe',
    'sensitivity_thresholds',
    'varImp_rfe',
    'fit_xgbTree_Tuning',
    'fit_ranger_Tuning',
    'pred_corr'
  ),
  file = 'data_output/RMarkdown_Objects.RData'
)

# save.image(file = 'data_output/ALL.RData')

# Save files for ShinyApps
saveRDS(bank_train, file = 'shinyapps/plot_eda/data/bank_train.rds')
saveRDS(all_real_results, file = 'shinyapps/model_dash/data/all_real_results.rds')
saveRDS(file_list, file = 'shinyapps/model_dash/data/file_list.rds')

m=1
for (m in seq(nrow(file_list))){
  # saveRDS(get(paste0(file_list[m, 'model_file'])), file = paste0('shinyapps/model_dash/data/', file_list[m, 'model_file'], '.rds'))
  saveRDS(get(paste0(file_list[m, 'cm_file'])), file = paste0('shinyapps/model_dash/data/', file_list[m, 'cm_file'], '.rds'))
  saveRDS(get(paste0(file_list[m, 'roc'])), file = paste0('shinyapps/model_dash/data/', file_list[m, 'roc'], '.rds'))
  saveRDS(get(paste0(file_list[m, 'density'])), file = paste0('shinyapps/model_dash/data/', file_list[m, 'density'], '.rds'))
}

print(paste0('[', round(
  difftime(Sys.time(), start_time, units = 'mins'), 1
), 'm]: ',
'All operations are over!'))


# Render RMarkdown Report ----
if (is.null(webshot:::find_phantom())) {
  webshot::install_phantomjs()
}
invisible(
  rmarkdown::render(
    'Bank-Marketing-Report.Rmd',
    'github_document',
    params = list(shiny = FALSE),
    runtime = 'static'
  )
)
invisible(
  rmarkdown::render(
    'Bank-Marketing-Report.Rmd',
    'html_document',
    params = list(shiny = TRUE),
    output_options = list(code_folding = 'hide')
  )
)
# # invisible(rmarkdown::run('Bank-Marketing-Report.Rmd'))

# beep(8)
#
# print(paste0('[', round(
#   difftime(Sys.time(), start_time, units = 'mins'), 1
# ), 'm]: ',
# 'Report generated! ---END---'))
