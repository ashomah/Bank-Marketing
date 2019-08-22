####
#### THIS SCRIPT DEFINES PARAMETERS AND FUNCTIONS FOR MODELING
####

seed <- ifelse(exists('seed'), seed, 2019)
set.seed(seed)

time_fit_start <- 0
time_fit_end <- 0
all_results <- data.frame()
all_real_results <- data.frame()
file_list <- data.frame()


# Cross-Validation Settings ----
fitControl <-
  trainControl(
    method = 'repeatedcv',
    number = 10,
    repeats = 3,
    verboseIter = TRUE,
    allowParallel = TRUE,
    classProbs = TRUE,
    savePredictions = TRUE
  )

print(paste0(
  ifelse(exists('start_time'), paste0('[', round(
    difftime(Sys.time(), start_time, units = 'mins'), 1
  ),
  'm]: '), ''),
  'Fit Control will use ',
  fitControl$method,
  ' with ',
  fitControl$number,
  ' folds and ',
  fitControl$repeats,
  ' repeats.'
))


# Cross-Validation for Tuning Settings ----
tuningControl <-
  trainControl(
    method = 'cv',
    number = 10,
    verboseIter = TRUE,
    allowParallel = TRUE,
    classProbs = TRUE,
    savePredictions = TRUE
  )

print(paste0(
  ifelse(exists('start_time'), paste0('[', round(
    difftime(Sys.time(), start_time, units = 'mins'), 1
  ),
  'm]: '), ''),
  'Tuning Control will use ',
  tuningControl$method,
  ' with ',
  tuningControl$number,
  ' folds and ',
  tuningControl$repeats,
  ' repeats.'
))


# Default tuneGrid for Random Forest ----
ranger_grid = expand.grid(
  mtry = c(1, 2, 3, 4, 5, 6, 7, 10),
  splitrule = c('gini'),
  min.node.size = c(5, 6, 7, 8, 9, 10)
)


# Default tuneGrid for XGBoost ----
nrounds = 1000

xgb_grid = expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 100),
  max_depth = c(5, 6, 7, 8, 9),
  eta = c(0.025, 0.05, 0.1, 0.2, 0.3),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(2, 3, 5),
  subsample = 1
)


# Function to get the best results in caret ----
get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}


# Color Palette
color1 = 'white'
color2 = 'black'
color3 = 'black'
color4 = 'darkturquoise'
font1 = 'Impact'
font2 = 'Helvetica'
BarFillColor <- "#330066"
HBarFillColor <- "#000099"
BarLineColor <- "#FFFAFA"
MissingColor <- "#FF6666"


print(paste0(
  ifelse(exists('start_time'), paste0('[', round(
    difftime(Sys.time(), start_time, units = 'mins'), 1
  ),
  'm]: '), ''),
  'Modeling Parameters and Functions ready!'
))


