# ==Library
library(ggplot2)
library(corrplot)
library(plyr)
library(tidyverse)
library(MLmetrics)


setwd("F:/ZINDI/Mobile_money_Tanzanie")
data <-read.csv("Data/training.csv")
data <- data[ ,-which(names(data)%in%c("ID","mobile_money","savings","borrowing","insurance"))]
data$mobile_money_classification <- as.factor(as.character(data$mobile_money_classification))


#==extract train and test data
set.seed(123)
Xtrain<- sample_frac(tbl = data, replace = FALSE, size = 0.60)
Xtest <- anti_join(data, Xtrain)
pos <- which(names(Xtrain)=="mobile_money_classification")


#== Trees
library(rpart)
fit <- rpart(mobile_money_classification~., data=Xtrain)
summary(fit)
predictions <- predict(fit, Xtest[,-pos], type = "prob")
MultiLogLoss(y_true = Xtest[,pos], y_pred = predictions)
#0.8502316


#== C4.5
library(RWeka)
fit <- J48(mobile_money_classification~., data=Xtrain)
summary(fit)
predictions <- predict(fit, Xtest[,-pos], type = "prob")
MultiLogLoss(y_true = Xtest[,pos], y_pred = predictions)
#5.983423



#== PART
library(RWeka)
fit <- PART(mobile_money_classification~., data=Xtrain)
summary(fit)
predictions <- predict(fit, Xtest[,-pos], type = "prob")
MultiLogLoss(y_true = Xtest[,pos], y_pred = predictions)
#8.132859



#== Bagging CART
library(ipred)
fit <- bagging(mobile_money_classification~., data=Xtrain)
summary(fit)
predictions <- predict(fit, Xtest[,-pos], type = "prob")
MultiLogLoss(y_true = Xtest[,pos], y_pred = predictions)
#1.513668


#== Random Forest
library(randomForest)
fit <- randomForest(mobile_money_classification~., data=Xtrain)
summary(fit)
predictions <- predict(fit, Xtest[,-pos], type = "prob")
MultiLogLoss(y_true = Xtest[,pos], y_pred = predictions)
#0.8074661



#==Gradient Boosted Machine
library(gbm)
fit <- gbm(mobile_money_classification~., data=Xtrain, distribution="multinomial")
print(fit)
predictions <- predict(fit, Xtest[,-pos])
MultiLogLoss(y_true = Xtest[,pos], y_pred = predictions)


#== Boosted C5.0
library(C50)
fit <- C5.0(mobile_money_classification~., data=Xtrain, trials=10)
print(fit)
predictions <- predict(fit,Xtest[,-pos], type = "prob")
MultiLogLoss(y_true = Xtest[,pos], y_pred = predictions)


#==tree
library(party)
tree = ctree(mobile_money_classification~., data=Xtrain)
predictions <- predict(tree,Xtest[,-pos])
MultiLogLoss(y_true = Xtest[,pos], y_pred = unlist(predictions))





