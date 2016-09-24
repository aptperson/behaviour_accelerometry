##### run script for testing the cv-by aninmal
rm(list = ls())
setwd("~/Documents/repos/behaviour_accelerometry/")

library(caret)
library(randomForest)
library(glmnet)
library(e1071)
require(devtools)
library(xgboost)

source("parallelCVxgb.r")
source("ml_run.r")
source("seals_data_prep_for_ml.r")

##### load the data
load("processed_data/featData4_13.RData")

# head(featData7_13)
# unique(featData7_13$SealName)
# [1] abbey   bella   groucho malie   mav     maxine  miri    nelson  rocky   ronnie  sly     teiko 

table(featData4_13$EventIds)

##### defin class maxes for down sampling
# these class maxs should give roughly ballanced overall class sizes
calss_max_table_train = data.frame(behaviours = c("Foraging", "Grooming", "Resting", "Travelling"),
                             classMax = c(550, Inf, 550, 450))
calss_max_table_test = data.frame(behaviours = c("Foraging", "Grooming", "Resting", "Travelling"),
                             classMax = c(550, Inf, 300, 300))
##### prep the data for ml
# save this data for uploading as minimal data
processed_feat_data <- seals_data_prep_for_ml(featureData = featData4_13,
                                              test_animal_names = c("maxine", "groucho"),
                                              codeTest = FALSE, # when this is TRUE the class max is forced to be 20
                                              classMaxTrain = calss_max_table_train,
                                              classMaxTest = calss_max_table_test)

save(processed_feat_data, file = "processed_data/processed_feat_data.RData")

##### run the ml model
xgb_test_run <- ml_run(trainData = processed_feat_data$trainDataSplit,
                       testData = processed_feat_data$testDataSplit,
                       folds_list = processed_feat_data$folds_list,
                       Model = "XGB")

save(xgb_test_run, file = "model_outputs/xgb_test_run.RData")

