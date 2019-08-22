####
#### THIS SCRIPT CHECK IF NECESSARY PACKAGES ARE INSTALLED AND LOADED
####

packages_list <- c(
  'data.table',
  'pROC',
  'ggthemes',
  'dplyr',
  'tibble',
  'tidyr',
  'corrplot',
  'GGally',
  'ggmap',
  'ggplot2',
  'grid',
  'gridExtra',
  'caret',
  'glmnet',
  'MLmetrics',
  'ranger',
  'xgboost',
  'doParallel',
  'factoextra',
  'foreach',
  'parallel',
  'kableExtra',
  'knitr',
  'RColorBrewer',
  'shiny',
  'beepr',
  'tufte',
  'flexclust',
  'caretEnsemble',
  'AUC',
  'shinydashboard',
  'rsconnect'
)

for (i in packages_list) {
  if (!i %in% installed.packages()) {
    install.packages(i, dependencies = TRUE)
    library(i, character.only = TRUE)
    warnings(paste0(i, ' has been installed'))
  } else {
    warnings(paste0(i, ' is already installed'))
    library(i, character.only = TRUE)
  }
}

print(paste0(
  ifelse(exists('start_time'), paste0('[', round(
    difftime(Sys.time(), start_time, units = 'mins'), 1
  ),
  'm]: '), ''),
  'All necessary packages installed and loaded'
))

