####
#### THIS SCRIPT SELECTS FEATURE USING A RECURSIVE FEATURE ELIMINATION
####


# Set seed
seed <- ifelse(exists('seed'), seed, 2019)
set.seed(seed)

# Feature Selection with Recursive Feature Elimination ----
subsets <- c(10, 20, 24, 26, 28, 30, 32, 34, 36, 40, 50, 60, 80, 100)

ctrl <- rfeControl(
  functions = rfFuncs,
  method = "cv",
  number = 5,
  verbose = TRUE,
  allowParallel = TRUE
)

if (calculate == TRUE) {
  library(doParallel)
  cl <- makePSOCKcluster(7)
  clusterEvalQ(cl, library(foreach))
  registerDoParallel(cl)
  print(paste0('[',
               round(
                 difftime(Sys.time(), start_time, units = 'mins'), 1
               ),
               'm]: ',
               'Starting RFE...'))
  time_fit_start <- Sys.time()
  
  results_rfe <-
    rfe(
      x = bank_train_A_lasso[,!names(bank_train_A_lasso) %in% c('y')],
      y = bank_train_A_lasso[, 'y'],
      sizes = subsets,
      rfeControl = ctrl,
      metric = 'Accuracy',
      maximize = TRUE
    )
  time_fit_end <- Sys.time()
  stopCluster(cl)
  registerDoSEQ()
  time_fit_rfe <- time_fit_end - time_fit_start
  saveRDS(results_rfe, 'models/results_rfe.rds')
  saveRDS(time_fit_rfe,
          'models/time_fit_rfe.rds')
}

results_rfe <- readRDS('models/results_rfe.rds')
time_fit_rfe <- readRDS('models/time_fit_rfe.rds')


varImp_rfe <-
  data.frame(
    'Variables' = attr(results_rfe$fit$importance[, 2], which = 'names'),
    'Importance' = as.vector(round(results_rfe$fit$importance[, 2], 4))
  )
varImp_rfe <- varImp_rfe[order(varImp_rfe$Importance), ]
varImp_rfe$perc <-
  round(varImp_rfe$Importance / sum(varImp_rfe$Importance) * 100, 4)
var_sel_rfe <- varImp_rfe[varImp_rfe$perc > 0.1, ]
var_rej_rfe <- varImp_rfe[varImp_rfe$perc <= 0.1, ]

ggplot(tail(varImp_rfe,50), aes(x = reorder(Variables, Importance), y = Importance)) +
  geom_bar(stat = 'identity') +
  coord_flip()

bank_train_A_rfe <-
  bank_train_A_FE2[, names(bank_train_A_FE2) %in% var_sel_rfe$Variables |
                       names(bank_train_A_FE2) == 'y']
bank_train_B_rfe <-
  bank_train_B_FE2[, names(bank_train_B_FE2) %in% var_sel_rfe$Variables |
                       names(bank_train_B_FE2) == 'y']
bank_test_rfe <-
  bank_test_FE2[, names(bank_test_FE2) %in% var_sel_rfe$Variables |
                    names(bank_test_FE2) == 'y']


print(
  paste0(
    '[',
    round(difftime(Sys.time(), start_time, units = 'mins'), 1),
    'm]: ',
    'Feature Selection with Recursive Feature Elimination is done!'
  )
)
