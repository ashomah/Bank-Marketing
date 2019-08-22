####
#### THIS SCRIPT IS TO COMPARE THE DIFFERENT MODELS FOR THE DIFFERENT THRESHOLDS
####
sensitivity_thresholds <- data.frame(rbind(
  glm_baseline = sens_temp_glm_baseline,
  glm_clustering = `sens_temp_glm_FE1 Clustering`,
  glm_binning = `sens_temp_glm_FE2 Binning`,
  glm_RFE = sens_temp_glm_RFE,
  ranger_baseline = sens_temp_ranger_baseline,
  ranger_clustering = `sens_temp_ranger_FE1 Clustering`,
  ranger_binning = `sens_temp_ranger_FE2 Binning`,
  ranger_tuned = sens_temp_ranger_Tuning,
  xgbTree_baseline = sens_temp_xgbTree_baseline,
  xgbTree_clustering = `sens_temp_xgbTree_FE1 Clustering`,
  xgbTree_binning = `sens_temp_xgbTree_FE2 Binning`,
  xgbTree_tuned = sens_temp_xgbTree_Tuning,
  xgbTree_RFE = sens_temp_xgbTree_RFE,
  glm_stack_baseline = sens_temp_glm_stack_baseline,
  glm_stack_clustering = sens_temp_glm_stack_clustering,
  glm_stack_binning = sens_temp_glm_stack_binning,
  glm_stack_tuned = sens_temp_glm_stack_Tuning,
  rf_stack_baseline = sens_temp_rf_stack_baseline,
  rf_stack_clustering = sens_temp_rf_stack_clustering,
  rf_stack_binning = sens_temp_rf_stack_binning,
  rf_stack_tuned = sens_temp_rf_stack_Tuning,
  xgbTree_stack_baseline = sens_temp_xgbTree_stack_baseline,
  xgbTree_stack_clustering = sens_temp_xgbTree_stack_clustering,
  xgbTree_stack_binning = sens_temp_xgbTree_stack_binning,
  xgbTree_stack_tuned = sens_temp_xgbTree_stack_Tuning)
)

# rownames(sensitivity_thresholds) <- c('glm_baseline','glm_clustering','glm_binning','glm_RFE',
#                          'ranger_baseline','ranger_clustering','ranger_binning','ranger_tuned',
#                          'xgbTree_baseline','xgbTree_clustering','xgbTree_binning','xgbTree_tuned','xgbTree_RFE',
#                          'glm_stack_baseline', 'glm_stack_clustering', 'glm_stack_binning','glm_stack_tuned',
#                          'rf_stack_baseline','rf_stack_clustering','rf_stack_binning','rf_stack_tuned',
#                          'xgbTree_stack_baseline','xgbTree_stack_clustering','xgbTree_stack_binning','xgbTree_stack_tuned')


assign('sensitivity_thresholds', sensitivity_thresholds, envir = .GlobalEnv)