
#Library
library(ggplot2)
library(corrplot)
library(plyr)
library(tidyverse)




########################################################
########################################################
######                                            ######
######           Data Import                      ######


setwd("F:/ZINDI/Mobile_money_Tanzanie")
datatrain <-read.csv("Data/training.csv")




########################################################
########################################################
######                                            ######
######  Tiaitement des données                    ######


#traitement des NA
sum(is.na(datatrain))

#data extraction
datatrain <- datatrain[ ,-which(names(datatrain)%in%c("mobile_money","savings","borrowing","insurance"))]


#Normalisation des données
normalize <-function(x){
  return((x-min(x,na.rm = TRUE))/(max(x,na.rm = TRUE)-min(x,na.rm = TRUE)))
}

for(l in colnames(datatrain[,!(names(datatrain)%in%c("ID","mobile_money_classification"))])){
  datatrain[,l] <-normalize(datatrain[,l])
}



########################################################
########################################################
######                                            ######
######  Analyse exploratiore                      ######


#corelation
library(corrplot)
res <- cor(datatrain, use = "complete.obs")
corrplot(res, tl.col="black", tl.cex=0.8, tl.srt=70)
col<- colorRampPalette(c("blue", "white", "red"))(20)
heatmap(x = res, col = col, symm = TRUE)


#ACP
library("FactoMineR")
library("factoextra")
res.pca <- PCA(datatrain, graph = FALSE)

#contribution
eig.val <- get_eigenvalue(res.pca)
eig.val
fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 50))

#var contr
fviz_pca_var(res.pca, col.var = "black")

#matrix de corelation
var <- get_pca_var(res.pca)
corrplot(var$cos2, is.corr=FALSE)

# Colorer en fonction du cos2
fviz_pca_var(res.pca, col.var = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE )



########################################################
########################################################
######                                            ######
######    Analyse explicative                     ######

df <- datatrain
pos.ID <- which(names(df)=="ID")
df <- df[,-pos.ID]

#== extract train and test data
set.seed(12345)
Xtrain<- sample_frac(tbl = df, replace = FALSE, size = 0.60)
Xtest <- anti_join(df, Xtrain)

x_train=model.matrix(mobile_money_classification~.-1,data=Xtrain) 
y_train=Xtrain$mobile_money_classification

x_tests=model.matrix(mobile_money_classification~.-1,data=Xtest)
y_test=Xtest$mobile_money_classification

#== Lasso
library(glmnet)
lasso_model <- glmnet(x_train,y_train,alpha=1)

#Meilleur model
pred <- predict(lasso_model,x_test)
rmse <- sqrt(apply((y_test-pred)^2,2,mean))
plot(log(lasso_model$lambda),rmse,type='b',xlab='Log(lambda)',
     main="meilleur lambda")

#Extrai var
best_lasso_model=lasso_model$lambda[order(rmse)[1]]
print (log(best_lasso_model))
coef(lasso_model,s=best_lasso_model)


########################################################
########################################################
######                                            ######
######    Analyse prédictive                      ######

dt <- datatrain
var <- c("ID")
dt <- dt[ ,-which(names(dt)%in%var)]
dt$mobile_money_classification <- as.factor(as.character(dt$mobile_money_classification))

##extract train and test data
set.seed(12345)
Xtrain<- sample_frac(tbl = dt, replace = FALSE, size = 0.60)
Xtest <- anti_join(dt, Xtrain)
pos <- which(names(Xtrain)=="mobile_money_classification")
f <- as.formula(paste("mobile_money_classification ~",paste(names(Xtrain[,-pos]), collapse = " + ")))




###################
####  Random Forest 
###
library("randomForest")
library("caret")
rf_model <- randomForest(f,data=Xtrain,type = "prob")
rf_model
#prediction
pred_rf <- predict(rf_model, newdata = Xtest[,-pos], type = "prob")
#Accuracy
confusionMatrix(pred_rf, Xtest[,pos])
accuracy_rf <- confusionMatrix(pred_rf, Xtest[,pos])$overall[1]
##loss
library(MLmetrics)
MultiLogLoss(y_true = Xtest[,pos], y_pred = pred_rf)






###################
####  SVM
###
library(e1071)
svm_model <- svm(f , data = Xtrain,probability = TRUE)
svm_model
#prediction
pred_svm <- predict(svm_model, newdata = Xtest[,-pos], probability = TRUE)
#Accuracy
confusionMatrix(pred_svm, Xtest[,pos])
accuracy_svm <- confusionMatrix(pred_svm, Xtest[,pos])$overall[1]
##loss
library(MLmetrics)
MultiLogLoss(y_true = Xtest[,pos], y_pred = attr(pred_svm, "probabilities"))


###################
####  NN
###
library(neuralnet)
m_Xtrain <- Xtrain
m_Xtrain$mobile_money_classification <- as.numeric(m_Xtrain$mobile_money_classification)
ff <- as.formula(paste("~mobile_money_classification+",paste(names(Xtrain[,-pos]), collapse = " + ")))
m_Xtrain <- model.matrix(ff,data =m_Xtrain)
nn_model<-neuralnet(f,data = m_Xtrain)
plot(nn_model,rep="best")
#prediction
model_nn<-neuralnet::compute(nn_model, Xtest[,-pos])
pred_nn<-round(model_nn$net.resul)
#Accuracy
Result <- ifelse(pred_nn[,1] %in% 1, 0,
                 ifelse(pred_nn[,1] %in% 2, 1,
                        ifelse(pred_nn[,1] %in% 3, 2,3)))
confusionMatrix(as.factor(Result),Xtest[,pos])
accuracy_nn <- confusionMatrix(as.factor(Result),Xtest[,pos])$overall[1]



###################
####  comparaison
###
accuracy <- tibble(Model_trainning =c("RF model","SVM model","NN model "),
                               MSE = c(accuracy_rf,accuracy_svm,accuracy_nn))

accuracy %>% ggplot(aes(Model_trainning,MSE))+ 
  geom_col(position = "dodge") + 
  ggtitle("Comparaison modéles")






###################
####  Hyperparamettre
###

#cost
res_cost=c()
cost = c(1e-3, 1e-2, 1e-1, 1,1e+1, 1e+2)
for(val in cost){
  svm_model <- svm(f , data = Xtrain)
  pred_svm <- predict(svm_model, newdata = Xtest[,-pos])
  accuracy_svm <- confusionMatrix(pred_svm, Xtest[,pos])$overall[1]
  res_cost <- c(res_cost,accuracy_svm)
}

lab_cost <- as.character(cost)
acc_cost <- tibble(Model_trainning =lab_cost,
                                 MSE =res_cost)
acc_cost %>% ggplot(aes(Model_trainning,MSE))+ 
  geom_col(position = "dodge") + 
  ggtitle("the best cost")


#kernel
res_kernel=c()
kernel = c("linear","polynomial","radial")
for(val in kernel){
    svm_model <- svm(f , data = Xtrain)
    pred_svm <- predict(svm_model, newdata = Xtest[,-pos])
    accuracy_svm <- confusionMatrix(pred_svm, Xtest[,pos])$overall[1]
    res_kernel <- c(res_kernel,accuracy_svm)
}

acc_kernel <- tibble(Model_trainning =kernel,
                                   MSE =res_kernel)
acc_kernel %>% ggplot(aes(Model_trainning,MSE))+ 
  geom_col(position = "dodge") + 
  ggtitle("the best kernel")



########################################################
########################################################
######                                            ######
######    Application sur base test   svm         ######
test <-read.csv("Data/test.csv")
datatest <- test[,-1]
predict_svm <- predict(svm_model, newdata = datatest,
                       decision.values = TRUE, probability = TRUE)
proba_pred <- attr(predict_svm, "probabilities")
#
proba_pred <- cbind.data.frame(test[,"ID"],proba_pred)
colnames(proba_pred) <- c("ID","no_financial_services","other_only","mm_only","mm_plus")
proba_pred$ID <- as.character(proba_pred$ID)
write.table(proba_pred,"Predict/prediction_svm.csv",sep=",",row.names =FALSE)




########################################################
########################################################
######                                            ######
######    Application sur base test   rf         ######

test <-read.csv("Data/test.csv")
datatest <- test[,-1]
predict_rf <- predict(rf_model, newdata = datatest,type = "prob")
proba_pred <- predict_rf
#
proba_pred <- cbind.data.frame(test[,"ID"],proba_pred)
colnames(proba_pred) <- c("ID","no_financial_services","other_only","mm_only","mm_plus")
proba_pred$ID <- as.character(proba_pred$ID)
write.table(proba_pred,"Predict/prediction_rf.csv",sep=",",row.names =FALSE)

#wite data kone yakouba
#write.table(Trying,"Predict/prediction_kone.csv",sep=",",row.names =FALSE)



 