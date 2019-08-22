####
#### THIS SCRIPT CREATES A NEW FEATURE BASED ON CLUSTERING METHOD
####

# Set seed
seed <- ifelse(exists('seed'), seed, 2019)
set.seed(seed)


# Select the features describing the clients
to_cluster <- colnames(bank_train_A_proc_dum)[substr(x = colnames(bank_train_A_proc_dum), start = 1, stop = 3) %in% c('age', 'job', 'mar', 'edu', 'def', 'bal', 'hou', 'loa')]


# Define the optimal number of clusters for K-Means
if (calculate == TRUE){
  opt_nb_clusters <- fviz_nbclust(bank_train_A_proc_dum[, to_cluster], kmeans, method = c("silhouette", "wss", "gap_stat"))
  saveRDS(opt_nb_clusters, 'data_output/opt_nb_clusters.rds')
}
opt_nb_clusters <- readRDS('data_output/opt_nb_clusters.rds')
k <- as.integer(opt_nb_clusters$data[opt_nb_clusters$data$y == max(opt_nb_clusters$data$y), 'clusters'])

# Training Clusters ----
if (calculate == TRUE){
  assign(paste0('kmeans_', k), kcca(bank_train_A_proc_dum[, to_cluster], k, kccaFamily('kmeans'), save.data = TRUE))
  assign(paste0('silhouette_', k), Silhouette(get(paste0('kmeans_', k))))
  
  saveRDS(get(paste0('kmeans_', k)), paste0('data_output/kmeans_', k, '.rds'))
  saveRDS(get(paste0('silhouette_', k)), paste0('data_output/silhouette_', k, '.rds'))
}

assign(paste0('kmeans_', k), readRDS(paste0('data_output/kmeans_', k, '.rds')))
assign(paste0('silhouette_', k), readRDS(paste0('data_output/silhouette_', k, '.rds')))


# Predicting Clusters
dummy_cluster <- dummyVars(formula = '~.', data = data.frame('cluster' = as.factor(predict(get(paste0('kmeans_', k))))))
bank_train_A_FE1 <- cbind(bank_train_A_proc_dum, predict(dummy_cluster, data.frame('cluster' = as.factor(predict(get(paste0('kmeans_', k)))))))
bank_train_B_FE1 <- cbind(bank_train_B_proc_dum, predict(dummy_cluster, data.frame('cluster' = as.factor(predict(get(paste0('kmeans_', k)), bank_train_B_proc_dum[, to_cluster])))))
bank_test_FE1 <- cbind(bank_test_proc_dum, predict(dummy_cluster, data.frame('cluster' = as.factor(predict(get(paste0('kmeans_', k)), bank_test_proc_dum[, to_cluster])))))


