setwd('F:/ZINDI/Mobile_money_Tanzanie')

random_pred <- cbind.data.frame(random_pred_ID,random_pred)
colnames(random_pred) <- c("ID","no_financial_services","other_only","mm_only","mm_plus")
random_pred$ID <- as.character(random_pred$ID)
write.table(random_pred,"Predict/random_pred_rf.csv",sep=",",row.names =FALSE)


svm_pred <- cbind.data.frame(svm_pred_ID,svm_pred)
colnames(svm_pred) <- c("ID","no_financial_services","other_only","mm_only","mm_plus")
svm_pred$ID <- as.character(svm_pred$ID)
write.table(svm_pred,"Predict/svm_pred_svm.csv",sep=",",row.names =FALSE)
