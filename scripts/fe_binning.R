####
#### THIS SCRIPT CREATES A NEW FEATURE BASED ON QUARTILE BINNING METHOD
####


# Set seed
seed <- ifelse(exists('seed'), seed, 2019)
set.seed(seed)


# Custom Function to bin featues:
bin_features <- function(df, lst, cut_rate) {
  for (i in lst) {
    df[, paste0(i, '_bin_', cut_rate)] <- .bincode(df[, i],
                                                   breaks = quantile(df[, i], seq(0, 1, by = 1 / cut_rate)),
                                                   include.lowest = TRUE)
    df[, paste0(i, '_bin_', cut_rate)] <-
      factor(df[, paste0(i, '_bin_', cut_rate)], levels = seq(1, cut_rate))
    # df[, i] <- NULL
  }
  return(df)
}

features_list <- c('age', 'balance', 'duration', 'campaign')

# First: binning numerical variables in 3 bins:
bank_train_A_FE2 <- copy(bank_train_A_FE1)
bank_train_B_FE2 <- copy(bank_train_B_FE1)
bank_test_FE2 <- copy(bank_test_FE1)
bank_train_A_FE2 <- bin_features(bank_train_A_FE2, features_list, 3)
bank_train_B_FE2 <- bin_features(bank_train_B_FE2, features_list, 3)
bank_test_FE2 <- bin_features(bank_test_FE2, features_list, 3)

# Second: binning numerical variables in 4 bins:
bank_train_A_FE2 <- bin_features(bank_train_A_FE2, features_list, 4)
bank_train_B_FE2 <- bin_features(bank_train_B_FE2, features_list, 4)
bank_test_FE2 <- bin_features(bank_test_FE2, features_list, 4)

# Third: binning numerical variables in 5 bins:
bank_train_A_FE2 <- bin_features(bank_train_A_FE2, features_list, 5)
bank_train_B_FE2 <- bin_features(bank_train_B_FE2, features_list, 5)
bank_test_FE2 <- bin_features(bank_test_FE2, features_list, 5)

# Fourth: binning numerical variables in 10 bins:
bank_train_A_FE2 <-
  bin_features(bank_train_A_FE2, features_list, 10)
bank_train_B_FE2 <-
  bin_features(bank_train_B_FE2, features_list, 10)
bank_test_FE2 <- bin_features(bank_test_FE2, features_list, 10)


# Dummies
dummy_bins <-
  dummyVars(formula = '~.', data = bank_train_A_FE2[, (substr(
    x = colnames(bank_train_A_FE2),
    start = nchar(colnames(bank_train_A_FE2)) - 5,
    stop = nchar(colnames(bank_train_A_FE2)) - 2
  ) == '_bin') |
    (substr(
      x = colnames(bank_train_A_FE2),
      start = nchar(colnames(bank_train_A_FE2)) - 5,
      stop = nchar(colnames(bank_train_A_FE2)) - 2
    ) == 'bin_')])
bank_train_A_FE2 <-
  cbind(bank_train_A_FE2[, (substr(
    x = colnames(bank_train_A_FE2),
    start = nchar(colnames(bank_train_A_FE2)) - 5,
    stop = nchar(colnames(bank_train_A_FE2)) - 2
  ) != '_bin') &
    (substr(
      x = colnames(bank_train_A_FE2),
      start = nchar(colnames(bank_train_A_FE2)) - 5,
      stop = nchar(colnames(bank_train_A_FE2)) - 2
    ) != 'bin_')],
  predict(dummy_bins, bank_train_A_FE2[, (substr(
    x = colnames(bank_train_A_FE2),
    start = nchar(colnames(bank_train_A_FE2)) - 5,
    stop = nchar(colnames(bank_train_A_FE2)) - 2
  ) == '_bin') |
    (substr(
      x = colnames(bank_train_A_FE2),
      start = nchar(colnames(bank_train_A_FE2)) - 5,
      stop = nchar(colnames(bank_train_A_FE2)) - 2
    ) == 'bin_')]))

bank_train_B_FE2 <-
  cbind(bank_train_B_FE2[, (substr(
    x = colnames(bank_train_B_FE2),
    start = nchar(colnames(bank_train_B_FE2)) - 5,
    stop = nchar(colnames(bank_train_B_FE2)) - 2
  ) != '_bin') &
    (substr(
      x = colnames(bank_train_B_FE2),
      start = nchar(colnames(bank_train_B_FE2)) - 5,
      stop = nchar(colnames(bank_train_B_FE2)) - 2
    ) != 'bin_')],
  predict(dummy_bins, bank_train_B_FE2[, (substr(
    x = colnames(bank_train_B_FE2),
    start = nchar(colnames(bank_train_B_FE2)) - 5,
    stop = nchar(colnames(bank_train_B_FE2)) - 2
  ) == '_bin') |
    (substr(
      x = colnames(bank_train_B_FE2),
      start = nchar(colnames(bank_train_B_FE2)) - 5,
      stop = nchar(colnames(bank_train_B_FE2)) - 2
    ) == 'bin_')]))

bank_test_FE2 <-
  cbind(bank_test_FE2[, (substr(
    x = colnames(bank_test_FE2),
    start = nchar(colnames(bank_test_FE2)) - 5,
    stop = nchar(colnames(bank_test_FE2)) - 2
  ) != '_bin') &
    (substr(
      x = colnames(bank_test_FE2),
      start = nchar(colnames(bank_test_FE2)) - 5,
      stop = nchar(colnames(bank_test_FE2)) - 2
    ) != 'bin_')],
  predict(dummy_bins, bank_test_FE2[, (substr(
    x = colnames(bank_test_FE2),
    start = nchar(colnames(bank_test_FE2)) - 5,
    stop = nchar(colnames(bank_test_FE2)) - 2
  ) == '_bin') |
    (substr(
      x = colnames(bank_test_FE2),
      start = nchar(colnames(bank_test_FE2)) - 5,
      stop = nchar(colnames(bank_test_FE2)) - 2
    ) == 'bin_')]))

