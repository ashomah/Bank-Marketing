##################################################################################################
################################ PLOT CORRELATION ################################################
# Takes df with only numerical data as input. Which basically means first subset the numerical data and the use the function
plot_correlation <- function (df, size_p = 0.5, size_t = 0.8){
  
  cor_numVar <- cor(df, use="pairwise.complete.obs") #correlations of all numeric variables
  cor_sorted <- as.matrix(sort(cor_numVar[,'glm_baseline'], decreasing = TRUE))
  CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.01)))
  cor_numVar <- cor_numVar[CorHigh, CorHigh]
  corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt", number.cex = size_p, tl.cex = size_t)
  
}

