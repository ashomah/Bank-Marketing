####
#### THIS SCRIPT READS AND PREPARE THE DATASET
####


# Read datasets ----
raw_train <- read.csv("data_input/BankCamp_train.csv")
raw_test <- read.csv("data_input/BankCamp_test.csv")

saveRDS(raw_train, "data_output/raw_train.rds")
saveRDS(raw_test, "data_output/raw_test.rds")


# Check if NAs ----
print(paste0(
  # '[',
  # round(difftime(Sys.time(), start_time, units = 'mins'), 1),
  # 'm]: ',
  'There are ',
  sum(is.na(raw_train)),
  ' NAs in Train, and ',
  sum(is.na(raw_test)),
  ' NAs in Test.'
))


# Prepare Datasets ----
bank_train <- raw_train
bank_test <- raw_test

bank_train$y       = ifelse(bank_train$y == "yes", 1, 0) #yes = 1, no = 0
bank_train$default = ifelse(bank_train$default == "yes", 1, 0) #yes = 1, no = 0
bank_train$housing = ifelse(bank_train$housing == "yes", 1, 0) #yes = 1, no = 0
bank_train$loan    = ifelse(bank_train$loan == "yes", 1, 0) #yes = 1, no = 0

bank_test$y <- NA
bank_test$default = ifelse(bank_test$default == "yes", 1, 0) #yes = 1, no = 0
bank_test$housing = ifelse(bank_test$housing == "yes", 1, 0) #yes = 1, no = 0
bank_test$loan    = ifelse(bank_test$loan == "yes", 1, 0) #yes = 1, no = 0

bank_train$y <- as.factor(bank_train$y)
levels(bank_train$y) <- c('No', 'Yes')
bank_test$y <- as.factor(bank_test$y)
levels(bank_test$y) <- levels(bank_train$y)

saveRDS(bank_train, "data_output/bank_train.rds")
saveRDS(bank_test, "data_output/bank_test.rds")


print(paste0(
  ifelse(exists('start_time'), paste0('[', round(
    difftime(Sys.time(), start_time, units = 'mins'), 1
  ), 'm]: '), ''),
  'Datasets Loaded'))
