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
# load("processed_data/featData4_13.RData")
load("processed_data/outputData_13.RData")
# head(featData7_13)
# unique(featData7_13$SealName)
# [1] abbey   bella   groucho malie   mav     maxine  miri    nelson  rocky   ronnie  sly     teiko 
# SL: abbey, miri, maxine, rocky, malie, teiko
# FS: bella, groucho, mav, ronnie, sly, nelson

print(table(outputData$EventIds, outputData$SealName))
print(table(outputData$EventIds, outputData$Place))

##### remove land Foraging

# names(outputData)
# 
outputData$Place <- as.numeric(as.factor(outputData$Place))
# table(outputData$Place)
# 
# feat_data_less_feat <- outputData[, !grepl(pattern = "[0-9]", x = names(outputData))]
# feat_data_less_feat <- feat_data_less_feat[, !grepl(pattern = "LQ", x = names(feat_data_less_feat))]
# feat_data_less_feat <- feat_data_less_feat[, !grepl(pattern = "UQ", x = names(feat_data_less_feat))]
# names(feat_data_less_feat)


##### defin class maxes for down sampling
# these class maxs should give roughly ballanced overall class sizes
calss_max_table_train = data.frame(behaviours = c("Foraging", "Grooming", "Resting", "Travelling"),
                                   classMax = c(550, Inf, 550, 450))
calss_max_table_test = data.frame(behaviours = c("Foraging", "Grooming", "Resting", "Travelling"),
                                  classMax = c(300, Inf, 300, 300))
##### prep the data for ml
# save this data for uploading as minimal data
processed_feat_data <- seals_data_prep_for_ml(featureData = outputData,
                                              test_animal_names = c("maxine", "groucho"),
                                              codeTest = FALSE, # when this is TRUE the class max is forced to be 20
                                              classMaxTrain = calss_max_table_train,
                                              classMaxTest = calss_max_table_test)

print(table(processed_feat_data$trainDataSplit$EventIds))
print(table(processed_feat_data$testDataSplit$EventIds))

save(processed_feat_data, file = "processed_data/processed_feat_data.RData")
load(file = "processed_data/processed_feat_data.RData")

##### look at the classes by fold
trainEventIds <- as.factor(processed_feat_data$trainDataSplit[, "EventIds"])
testEventIds <- as.factor(processed_feat_data$testDataSplit[, "EventIds"])

trainFoldTable <- lapply(1:10, function(i){
table(trainEventIds[-processed_feat_data$folds_list[[i]]])
})
testFoldTable <- lapply(1:10, function(i){
  table(testEventIds[-processed_feat_data$folds_list[[i]]])
})
print(do.call(rbind, trainFoldTable))
print(do.call(rbind, testFoldTable))


##### run the ml model
xgb_test_run <- ml_run(trainData = processed_feat_data$trainDataSplit,
                       # codeTest = F,
                       testData = processed_feat_data$testDataSplit,
                       folds_list = processed_feat_data$folds_list,
                       Model = "XGB")

save(xgb_test_run, file = "model_outputs/xgb_test_run.RData")

