####
#### THIS SCRIPT SELECTS FEATURE USING A LASSO REGRESSION
####


# Set seed
seed <- ifelse(exists('seed'), seed, 2019)
set.seed(seed)

# Feature Selection with Lasso Regression ----
X <- model.matrix(y ~ ., bank_train_A_FE2)[,-1]
y <- ifelse(bank_train_A_FE2$y == 'Yes', 1, 0)
cv <- cv.glmnet(X, y, alpha = 1, family = 'binomial')
lasso_glmnet <- glmnet(X, y, alpha = 1, family = 'binomial', lambda = cv$lambda.min)
coef(lasso_glmnet)

lassoVarImp <-
  varImp(lasso_glmnet, scale = FALSE, lambda = cv$lambda.min)

varsSelected    <-
  rownames(lassoVarImp)[which(lassoVarImp$Overall != 0)]
varsNotSelected <-
  rownames(lassoVarImp)[which(lassoVarImp$Overall == 0)]

print(
  paste0(
    'The Lasso Regression selected ',
    length(varsSelected),
    ' variables, and rejected ',
    length(varsNotSelected),
    ' variables.'
  )
)

bank_train_A_lasso <-
  bank_train_A_FE2[,!names(bank_train_A_FE2) %in% varsNotSelected]
bank_train_B_lasso <-
  bank_train_B_FE2[,!names(bank_train_B_FE2) %in% varsNotSelected]
bank_test_lasso <-
  bank_test_FE2[,!names(bank_test_FE2) %in% varsNotSelected]

print(paste0(
  '[',
  round(difftime(Sys.time(), start_time, units = 'mins'), 1),
  'm]: ',
  'Feature Selection with Lasso Regression is done!'
))
