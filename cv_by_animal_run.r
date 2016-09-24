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
load("processed_data/featData7_13v2.RData")
load("processed_data/featData4_13.RData")

# head(featData7_13)
# unique(featData7_13$SealName)
# [1] abbey   bella   groucho malie   mav     maxine  miri    nelson  rocky   ronnie  sly     teiko 

##### prep the data for ml
processed_feat_data <- seals_data_prep_for_ml(featureData = featData4_13,
                                              codeTest = TRUE, # when this is TRUE the class max is forced to be 20
                                              classMax = 100)



##### run the ml model
xgb_test_run <- ml_run(trainData = processed_feat_data$trainDataSplit,
                       testData = processed_feat_data$testDataSplit,
                       folds_list = processed_feat_data$folds_list,
                       Model = "XGB")

save(xgb_test_run, file = "model_outputs/xgb_test_run.RData")

