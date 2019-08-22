####
#### THIS SCRIPT DETERMINES THE CORRELATION BETWEEN THE PREDICTIONS OF THE DIFFERENT MODELS TO DEFINE THE BEST ONES TO STACK
####
source('scripts/fct_plot_correlation.R')
pred_corr <- data.frame(
  glm_baseline = submission_glm_valid_baseline,
  glm_clustering = `submission_glm_valid_FE1 Clustering`,
  glm_binning = `submission_glm_valid_FE2 Binning`,
  glm_RFE = submission_glm_valid_RFE,
  ranger_baseline = `submission_ranger_valid_baseline`,
  ranger_clustering = `submission_ranger_valid_FE1 Clustering`,
  ranger_binning = `submission_ranger_valid_FE2 Binning`,
  ranger_tuned = submission_ranger_valid_Tuning,
  xgbTree_baseline = submission_xgbTree_valid_baseline,
  xgbTree_clustering = `submission_xgbTree_valid_FE1 Clustering`,
  xgbTree_binning = `submission_xgbTree_valid_FE2 Binning`,
  xgbTree_tuned = submission_xgbTree_valid_Tuning,
  xgbTree_RFE = submission_xgbTree_valid_RFE
)

colnames(pred_corr) <- c('glm_baseline','glm_clustering','glm_binning','glm_RFE',
                         'ranger_baseline','ranger_clustering','ranger_binning','ranger_tuned',
                         'xgbTree_baseline','xgbTree_clustering','xgbTree_binning','xgbTree_tuned','xgbTree_RFE')

png( 'plots/prediction_corrplot',
     width = 1500,
     height = 1000)
plot_correlation(pred_corr, size_p = 1, size_t = 0.8)
dev.off()

assign('pred_corr', pred_corr, envir = .GlobalEnv)