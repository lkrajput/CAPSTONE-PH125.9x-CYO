# title: "CAPSTONE-PH125.9x-Indian Patient Liver Diseases"
# author: "Laxmansingh Rajput"
# date: "6/3/2019"

##
##   Libraries used
##
library(tidyverse)  # for data manipulation
library(knitr)      # for creating readable information
library(kableExtra)
library(caret)      # for model-building
library(DMwR)       # for smote implementation
library(pROC)      # for AUC calculations

# OBJECTIVES

## Creating a recommendation system using the Indian Liver Patient dataset. 
## Train a machine learning algorithm using the inputs in one subset to predict liver patients in the validation set.

# The overall accuracy is used as predictions are compared to the value in the validation.

##
##  Read the csv file in as a dataframe
##
indian_liver_patient <- data.frame(read_csv("Data/indian_liver_patient.csv"))
##
##   Convert the Gender and Dataset (prediction column) to factors
##
indian_liver_patient$Dataset <- factor(indian_liver_patient$Dataset)
indian_liver_patient$Gender <- factor(indian_liver_patient$Gender)
##
##  Compute the column means for all but the Gender and Dataset columns
##    As there are missing values in the dataset remove them from the calculation
##
int_column_means <- data.frame(colMeans(select(indian_liver_patient, -2, -11), 
                                        na.rm = TRUE)) 
names(int_column_means) <- "mean_values"

column_means <- rbind(int_column_means$mean_values[1], 
                      'NA',
                      int_column_means$mean_values[2],
                      int_column_means$mean_values[3],
                      int_column_means$mean_values[4],
                      int_column_means$mean_values[5],
                      int_column_means$mean_values[6],
                      int_column_means$mean_values[7],
                      int_column_means$mean_values[8],
                      int_column_means$mean_values[9],
                      'NA')

int_column_means <- data.frame(colMeans(select(indian_liver_patient,-2, -11)))
##
## Explore the Indian Liver Patient dataset prior to splitting (train and test)
##
# 1. First 10 rows of the dataset

##  Prevent the Scientifc Notation by using scipen
##
options(scipen=999)
##
##  List the first 10 records in the dataset 
##    Use Kable to make the information more readable.
##
n_records <- 10

first_n_records <- data.frame(t(head(indian_liver_patient, n_records)))
names(first_n_records) <- seq(1:n_records)

kable(first_n_records, 
      align = c("l", "c"),
      caption = "Indian Liver Patient dataset information") %>%  
  kable_styling(bootstrap_options = c("striped", "hover"), 
                full_width = F, 
                fixed_thead = T,
                font_size = 7,
                latex_options = "hold_position") %>%
  footnote(general = "   First 10 rows displayed ")

# 2. The list of the variables.

##
##  Kable together the variables and corresponding means
##     note the above we had computed the column means and removed the NAs earlier
##
variable_information <- data.frame(variable.names(indian_liver_patient), 
                                   column_means)

names(variable_information) <- c("Variable Names",
                                 "Mean")
options(scipen=999)

kable(variable_information, 
      align = c("l", "c"),
      caption = "Variable Information with mean") %>%  
  kable_styling(bootstrap_options = c("striped", "hover"), 
                full_width = F, 
                latex_options = "hold_position") %>%
  footnote(general = "   Mean computed with na.rm = TRUE. ",
           number = c("     Albumin_and_Globulin_Ratio has 4 missing values. "))

# 3. Describe the data

##
##   Use the skimr library/package to get better insight into the data
##     eg: missing values, mean, sd, etc.
##
library(skimr)  # provide summary statistics about variables

ilp_list <- skimr::skim_to_list(indian_liver_patient)

ilp_list %>% .$factor %>% 
  skimr::kable(caption = 'Factor Variables', 
               format = 'latex', 
               booktabs = T ) %>% 
  kable_styling(latex_options = 'hold_position', 
                font_size = 9)

ilp_list  %>% .$numeric %>% select(-12) %>%
  skimr::kable(caption = 'Numeric Variables', 
               format = 'latex', 
               booktabs = T ) %>% 
  kable_styling(latex_options = 'hold_position', 
                font_size = 9)
## Missing values and Imputation

# Find the missing value(s) and use various plots to explore the dataset

##
# lets quickly check the ‘missings’ pattern using mice::md.pattern.
#           Multivariate Imputation by Chained Equations
##
##   use capture.output to suppress extraneous data
##
library(mice)

garbage <- capture.output(md.pattern(indian_liver_patient, 
           rotate.names = TRUE))

##
##  Explore other libraries to display the same informattion in a more
##     helpful fashion.
##  Please note that the plotting is done according to column order in the dataset
##    hence even though there are labels missing on the X-axis, one can extrapolate 
##    the column with the missing value.
##  
library(VIM)

aggr_plot <- aggr(indian_liver_patient, 
                  col=c('navyblue',
                        'red'), 
                  numbers=TRUE, 
                  sortVars=FALSE, 
                  labels=names(indian_liver_patient), 
                  cex.axis=.7, 
                  gap=2, 
                  main="Aggregations For Missing/Imputed Values",
                  ylab=c("Histogram of missing data",
                         "Pattern"))
##
##  Use the marginplot to get additional information, note that the missing values are listed in the
##         margins.
## columns
##   10 - Albumin_and_Globulin_Ratio
##    9 - Albumin

marginplot(indian_liver_patient[c(10, 9)],  
            col = c("navyblue", "red", "red4", "orange", "orange4"),
           main = "Indian Patient Liver Dataset",
           sub = "Scatterplot With Additional Information (missing Values) In The Margins")


### Implementing the imputation in 2-steps, using mice() to build the model and complete() to generate the completed data. 
##
##  Identify the index of the missing values in the "Albumin_and_Globulin_Ratio" and its corresponding value
##     Which should be NA
##
index_missing_values = which(is.na(indian_liver_patient$Albumin_and_Globulin_Ratio))
actual_and_predicted_values = data.frame(index_missing_values, 
                                         indian_liver_patient$Albumin_and_Globulin_Ratio[index_missing_values])
#
# Mean of the "Albumin_and_Globulin_Ratio" before imputation (NAs to be taken into consideration)
#
mean_pre_imputation <- mean(indian_liver_patient$Albumin_and_Globulin_Ratio, na.rm = TRUE)
paste("Mean Pre Imputation of Albumin_and_Globulin_Ratio", mean_pre_imputation)
##
# Step 1 - perform mice imputation, based on random forests.
##
garbage <- capture.output(miceMod <- mice(indian_liver_patient[, !names(indian_liver_patient) %in% "Dataset"], 
                method="rf"))  

# Step 2 - generate the completed data.

miceOutput <- mice::complete(miceMod)
#
# Lets compute the prediction of Albumin_and_Globulin_Ratio.
#
#actuals <- indian_liver_patient$Albumin_and_Globulin_Ratio[is.na(indian_liver_patient$Albumin_and_Globulin_Ratio)]
predicteds <- miceOutput[is.na(indian_liver_patient$Albumin_and_Globulin_Ratio), 
                         "Albumin_and_Globulin_Ratio"]

indian_liver_patient$Albumin_and_Globulin_Ratio[which(is.na(indian_liver_patient$Albumin_and_Globulin_Ratio))] <- predicteds
#
# Compute the mean post-imputation
#
mean_post_imputation <- mean(indian_liver_patient$Albumin_and_Globulin_Ratio)
paste("Mean Post Imputation of Albumin_and_Globulin_Ratio", mean_post_imputation)
##
##  Create the dataset for the missing and predicted values for output
##
actual_and_predicted_values = cbind(actual_and_predicted_values, 
                                    indian_liver_patient$Albumin_and_Globulin_Ratio[index_missing_values])

names(actual_and_predicted_values) <- c("Missing Index",
                                        "Pre Imputation",
                                        "Post Imputation")

kable(actual_and_predicted_values, 
      align = c("c", "c", "c"),
      caption = "Predicted values for Albumin_and_Globulin_Ratio") %>%  
  kable_styling(bootstrap_options = c("striped", "hover"), 
                full_width = F, 
                fixed_thead = T,
                font_size = 12,
                latex_options = "hold_position")
#
# Split the data into train and test sets
#
set.seed(2969)  # for easy replication

test_index <- createDataPartition(y = indian_liver_patient$Dataset, 
                                  times = 1, 
                                  p = 0.1, 
                                  list = FALSE)

train <- indian_liver_patient[-test_index,]
test <- indian_liver_patient[test_index,]
##
##  Convert the values factors from
##    1 - YES
##    2 - NO
## 
train$Dataset <- ifelse(train$Dataset == 1, "YES", "NO")
train$Dataset <- as.factor(train$Dataset)

test$Dataset <- ifelse(test$Dataset == 1, "YES", "NO")
test$Dataset <- as.factor(test$Dataset)


#check table

# table uses the cross-classifying factors to build a contingency table of the counts at each combination of factor levels.

tt_count <- data.frame(table(train$Dataset))

# check classes distribution (Express Table Entries As Fraction Of Marginal Table)

tp_train <- data.frame(prop.table(table(train$Dataset)))

combo_table <- cbind(tt_count, tp_train[2])
names(combo_table) <- c("Value", 
                        "Count", 
                        "Percentage")

kable(combo_table, 
      align = c("c"),
      caption = "Cross Tabulation counts/Marginal Table") %>%  
  kable_styling(bootstrap_options = c("striped", "hover"), 
                full_width = F, 
                fixed_thead = T,
                font_size = 12,
                latex_options = "hold_position")


### More information - Accuracy measurements/Area under the Curve
##
##  Use rpart to get the area under the curve
##
library(rpart)
treeimb <- rpart(Dataset ~ ., 
                 data = train)

pred.treeimb <- predict(treeimb, Dataset = c("YES", "NO"), # 1 - YES, 2 - NO
                        newdata = test)

library(ROSE)  # Random Over-Sampling Examples
# ROSE package has a function names accuracy.meas, 
# it computes important metrics such as precision, recall & F measure.

ac_ms <- accuracy.meas(test$Dataset, pred.treeimb[,2])
roc_c <- roc.curve(test$Dataset, pred.treeimb[,2], 
                   plotit = FALSE)

imbal_table <- data.frame(
   "Description" = c(names(ac_ms[3]), 
                     names(ac_ms[4]), 
                     names(ac_ms[5]), 
                     "Area under the curve (AUC)"),
   "Value" = c(ac_ms$precision, 
               ac_ms$recall, 
               ac_ms$F, 
               roc_c$auc)
)

kable(imbal_table,
      align = c("l", "c"),
      caption = "Important metrics based on rpart ") %>%  
  kable_styling(bootstrap_options = c("striped", "hover"), 
                full_width = F, 
                fixed_thead = T,
                font_size = 12,
                latex_options = "hold_position")
##
## Visualize the data (by the predictors and gender)
##
library(ggplot2)
library(reshape2)
melt_train <- data.frame(melt(train))

ggplot(melt_train,
       aes(x=variable, 
           y = value, 
           color = Gender)) + 
  geom_point() + 
  theme(axis.text.x = element_text(angle = 45, 
                                   hjust = 1))+
  facet_wrap(~Dataset)
##
## Correlation 
##
##
##  Check for correlation and plot and display values
##
X <- select(train, -2, -11)
plot(X)  #  identify the various trends

ilp_cor <- round(cor(X),4)

library(xtable)

kable(xtable(ilp_cor),
      align = c("l", "c"),
      format = "latex",
      booktabs = T,
      caption = "The correlation matrix") %>%  
  kable_styling(bootstrap_options = c("striped", "hover"), 
                full_width = F, 
                fixed_thead = T,
                font_size = 12,
                latex_options = c("hold_position", "scale_down"))
##
## Principal Component Analysis (PCA) 
##
X <- select(train, -2, -11)   # remove Gender and dataset

pca <- prcomp(X, 
              center = T,
              scale = T)

pca_summary <- summary(pca)$importance %>%
  as.data.frame()

kable(pca_summary, caption = "PCA Summary") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                latex_options = c("scale_down", "hold_position"),
                full_width = F)
##
##  Screen plot with eigen-value 
##
screeplot(pca, 
          type = "l", 
          npcs = 9, 
          main = "Screenplot of the all 9 PCs")
abline(h = 1, 
       col="red", 
       lty=5)
legend("topright", legend=c("Eigenvalue = 1"),
       col=c("red"), lty=5, cex=0.6)
##
##  Cumulative variance Plot
##
cumpro <- cumsum(pca$sdev^2 / sum(pca$sdev^2))
plot(cumpro[0:9], 
     xlab = "PC #", 
     ylab = "Amount of explained variance", 
     main = "Cumulative variance plot")
abline(v = 3, col="blue", lty=5)
abline(h = 0.9437, col="blue", lty=5)
legend("topleft", legend=c("Cut-off @ PC3"),
       col=c("blue"), lty=5, cex=0.6)
##
## Random Forest to identify the variable that are most important
##

library(randomForest)

#Let's use random forest to see which variables are most important
##
##  Create the formula Prediction ~ features + ...
##
allVars <- colnames(train)
predictorVars <- allVars[!allVars%in% c('Dataset')]

predictorVars <- paste(predictorVars, collapse = "+")
f <- as.formula(paste("Dataset~", predictorVars, collapse = "+"))

rf_model <- randomForest(formula = f,
                         data = train)

importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

varImportance <-varImportance[order(-varImportance$Importance),]   # sort in reverse order 

top_3_important_var_int <- head(varImportance, 3)   # use head to get the top 3
top_3_var <- data.frame(top_3_important_var_int$Variables)
names(top_3_var) <- "Top 3 variables"

kable(top_3_var, caption = "PCA Summary") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                latex_options = c("hold_position"),
                full_width = F)

# RESULTS

## Train the following models:

##  adaboost
##  avNNet
##  LogitBoost
##  lda
##  loclda
##  naive_bayes
##  wsrf
##  gamLoess
##  kknn
##  knn
##  monmlp
##  mlp
##  mlpML
##  nnet
##  svmLinear3
##  svmLinear
##  svmRadial
##  svmRadialCost
##  svmRadialSigma
##  rf
##  Rborist
##  nodeHarvest

library(caret)
library(doParallel)
library(Rborist)

# Restrict the models

models <- c("avNNet", 
            "lda",
            "loclda",
            "naive_bayes",  
            "wsrf",
            "gamLoess", 
            "kknn", 
            "monmlp",
            "mlp", 
            "mlpML",
            "nnet",
            "svmLinear3",
            "svmLinear", 
            "svmRadial", 
            "svmRadialCost", 
            "svmRadialSigma",
            "rf")


garbage <- capture.output(
  fits <- lapply(models, function(model) { 
    print(model)
    train(Dataset ~ ., 
          method = model, 
          data = train, 
          silent = FALSE)
  }) )


names(fits) <- models

##
##  create a matrix of predictions for the test set. 
##
##

fits_predict_map <- map(fits, function(fit){
  predict(fit, 
          test)
})

##
##  Unlist the prediction map and calculate column means
##
mt_map <- matrix(unlist(fits_predict_map), 
                 nrow=dim(test)[1])

colnames(mt_map) <- models

acc1 <- colMeans(mt_map == test$Dataset)  
##
## Compute the accuracy for each models by creating a confusion matrix
##
df_map <- data.frame(matrix(unlist(fits_predict_map), 
                            nrow=dim(test)[1]))

names(df_map) <- models

garbage <- capture.output(confusion_matrix_map <- sapply(seq(1, length(models)), function(col_no){
    print(col_no)
    confusionMatrix(factor(df_map[,col_no]), 
                    test$Dataset)$overall["Accuracy"]
  }))

test_df_map <- cbind(test$Dataset, df_map)  # debugging purposes ONLY
##
## Display information for models and corresponding accuracy.
##    Sort in reverse order
##
accuracy_results <- data.frame(cbind(models = as.character(models), 
                                     accuracy = as.numeric(confusion_matrix_map))) %>%
            arrange(desc(accuracy))
##
## Prediction for Various models & overall accuracy
##

kable(df_map, 
      align = c("l", "c"),
      caption = "Prediction for Various models") %>%  
  kable_styling(bootstrap_options = c("striped", "hover"), 
                full_width = F, 
                fixed_thead = T,
                font_size = 7.5,
                latex_options = c("hold_position", "scale_down"))

kable(accuracy_results, 
      align = c("l", "c"),
      caption = "Overall Accuracy") %>%  
  kable_styling(bootstrap_options = c("striped", "hover"), 
                full_width = F, 
                fixed_thead = T,
                font_size = 7.5,
                latex_options = "hold_position")

accuracy_results_plot <- data.frame(cbind(models = as.character(models), 
                                     accuracy = as.numeric(confusion_matrix_map)))

kable(accuracy_results_plot, 
      align = c("l", "c"),
      caption = "Overall Accuracy") %>%  
  kable_styling(bootstrap_options = c("striped", "hover"), 
                full_width = F, 
                fixed_thead = T,
                font_size = 7.5,
                latex_options = "hold_position")
##
##   Plot the accuracy against the models
##
ggplot(accuracy_results_plot, 
       aes(models, 
           confusion_matrix_map)) +
  geom_point() + 
  ggtitle("Accuracy for the  models") +
  ylab("Accuracy") +
  theme(axis.text.x = element_text(angle = 45, 
                                   hjust = 1))

# CONCLUSION

## 
##  Display top model and the corresponding accuracy
##
print(paste("Model   : ", accuracy_results$models[1]))
print(paste("Accuracy: ", accuracy_results$accuracy[1]))


## Session Information for the execution

sessionInfo()
