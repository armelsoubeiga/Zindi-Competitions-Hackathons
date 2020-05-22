# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 20:49:35 2018

@author: Soubeiga Armel
"""


#Importation des données
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# work director
os.chdir(r'F:/ZINDI/Mobile_money_Tanzanie')
data=pd.read_csv(r'Data\training.csv', sep=",", engine='python')



# Data check
print(data.shape)
print(data.columns)
data.describe()



# Data splite
data=data.drop(['mobile_money', 'savings', 'borrowing','insurance'], axis=1)
ytrain = data['mobile_money_classification']
xtrain=data.drop(['mobile_money_classification', 'ID'], axis=1)



# Features transformation
np.isnan(xtrain.copy()).sum().sum()
xtrain.isnull().T.any().T.sum()
len(xtrain.columns[xtrain.isnull().any()].tolist()) 




# Visualisation de la variable d'interrêt mobile_money_classification
counts = pd.value_counts(ytrain)
plt.bar(range(4), counts)
plt.ylabel('number of instances')
plt.title('dataset constitution')
plt.xlabel('mobil money categories')
plt.xticks(range(4), np.unique(ytrain),rotation='vertical')
plt.show()




# ACP Exploration
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# normalize dataset
scaler = StandardScaler()  
Xnorm = scaler.fit_transform(xtrain.copy())
#Transforme ytrain
encoder=LabelEncoder()
y=encoder.fit_transform(ytrain.copy())
#Realisation du acp
pca = PCA(n_components = 5)
pca.fit(Xnorm)
Xpca = pca.transform(Xnorm) 
#axes contribution
plt.bar(np.arange(len(pca.explained_variance_ratio_))+ 0.5, pca.explained_variance_ratio_)
plt.title("Variance expliquée")
#Representation
plt.scatter(Xpca[:,0],Xpca[:,1], c = y)
plt.colorbar()
plt.show()



# train et validation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain, stratify = ytrain, test_size = 0.3)
x_train.shape
x_test.shape

# transformation des données
factor = pd.factorize(y_train)
y_train = factor[0]
definitions = factor[1]









####################################
##          KNN                   #
###################################

# importing necessary libraries 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
  
# fitting a KNN classifier 
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(x_train, y_train) 
# prediction sur xtest
knn_pred = knn.predict_proba(x_test)  
# Conversion des classes numériques des valeurs prédites
reversefactor = dict(zip(range(4),definitions))
knn_pred = np.vectorize(reversefactor.get)(knn_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, knn_pred, rownames=['Actual Species'], colnames=['Predicted Species']))
#L'accuracy
knn_acc=accuracy_score(y_test, knn_pred)
print(knn_acc)

#0.3903240958196336








####################################
##          Random Forest         #
###################################

#Importing Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

# Feature Scaling
#scaler = StandardScaler()
#x_train_norm = scaler.fit_transform(x_train.copy())

# Fitting Random Forest
rf_classifier = RandomForestClassifier(n_estimators = 500, criterion = 'gini')
rf_classifier.fit(x_train, y_train)

# Prediction sur les données de validation 
y_pred = rf_classifier.predict(x_test)
#y_pred = rf_classifier.predict_proba(x_test)

# Conversion des classes numériques des valeurs prédites
reversefactor = dict(zip(range(4),definitions))
y_pred = np.vectorize(reversefactor.get)(y_pred)

# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))

# L'accuracy
rf_acc=accuracy_score(y_test, y_pred)
print(rf_acc)
#0.6416157820573039








####################################
##          SVM                   #
###################################

# importation des librairy 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 

# fiting a linear SVM classifier 
svm_classifier = SVC(kernel = 'linear', C = 100, probability=True)
svm_classifier.fit(x_train, y_train) 
# Prediction sur les données de validation
svm_pred = svm_classifier.predict(x_test) 
#y_pred = svm_classifier.predict(x_test)




####################################
# save the model to disk
from sklearn.externals import joblib
joblib.dump(svm_classifier, r'Model\svm_classifier_2.pkl') 

# Importation des données de test
datatest=pd.read_csv(r'Data\test.csv', sep=",", engine='python')
ID = datatest['ID']
test=datatest.drop(['ID'], axis=1)

# importation du model du disk et paplication
loaded_model = joblib.load(r'Model\svm_classifier_2.pkl')

#Prediction sur les données test
svm_pred = loaded_model.predict_proba(test) 

#Enregsitrement des prediction est test
np.savetxt(r'Predict\svm_pred.csv', svm_pred, delimiter=",")
np.savetxt(r'Predict\svm_pred_ID.csv', ID, delimiter=",")






##################################

#Conversion des classes numériques des valeurs prédites
reversefactor = dict(zip(range(4),definitions))
svm_pred = np.vectorize(reversefactor.get)(svm_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, svm_pred, rownames=['Actual Species'], colnames=['Predicted Species']))
# L'accuracy 
svm_acc=accuracy_score(y_test, svm_pred)
print(svm_acc)


#0.6589948332550494
#linear c= 1e1 0.6552372005636449
#linear c=1e-1 0.6547674964772193
#linear c=1e-2 0.6613433536871771

# grid search with svm
from sklearn.grid_search import GridSearchCV
k=['rbf', 'linear','sigmoid']
k=['rbf', 'linear','poly','sigmoid']
c= [0.001, 0.10, 0.1,1, 10, 20,100]
g=['auto',1e-2, 1e-3]

param_grid=dict(kernel=k, C=c, gamma=g)
print param_grid
svr=svm.SVC()
grid = GridSearchCV(svr, param_grid, cv=5,scoring='accuracy')
grid.fit(X, y)  
print()
print("Grid scores on development set:")
print()  
print grid.grid_scores_  
print("Best parameters set found on development set:")
print()
print(grid.best_params_)
print("Grid best score:")
print()
print (grid.best_score_)
# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print grid_mean_scores


#Selection du modele
acc = pd.Series([knn_acc,rf_acc,svm_acc], index=['knn', 'rf', 'svm'])
knn, rf, svm = plt.bar(range(3), acc)
knn.set_facecolor('r')
rf.set_facecolor('g')
svm.set_facecolor('b')
plt.ylabel('accuracy')
plt.title('comparaison des performances')
plt.xlabel('modéles')
plt.xticks(range(3), ['knn', 'rf', 'svm'],rotation='vertical')
plt.show()





###### SVM et hyperparamatre  #######


#Performance par noyau
def check_kernel():
    svm_acc = []
    for kernel in ['linear','poly','rbf']:
        svm_classifier = SVC(kernel = kernel, C = 1, gamma='auto').fit(x_train, y_train)
        svm_pred = svm_classifier.predict(x_test) 
        reversefactor = dict(zip(range(4),definitions))
        svm_pred = np.vectorize(reversefactor.get)(svm_pred)
        svm_ac=accuracy_score(y_test, svm_pred)
        svm_acc.append(svm_ac)
    return(svm_acc)
    
svm_accuracy_kernel=check_kernel()

#visualisation
acc = pd.Series(svm_accuracy_kernel, index=['linear','poly','rbf'])
plt.bar(range(3), acc)
plt.ylabel('accuracy')
plt.title('comparaison des noyau du svm')
plt.xlabel('modéles')
plt.xticks(range(3), ['linear','poly','rbf'],rotation='vertical')
plt.show()



#Performance par hyper-parametres
def check_hyperparametre():
    svm_acc = []
    for c in [1e-2, 1e-1, 1, 1e1, 1e2]:
        svm_classifier = SVC(kernel = 'linear', C = c).fit(x_train, y_train)
        svm_pred = svm_classifier.predict(x_test) 
        reversefactor = dict(zip(range(4),definitions))
        svm_pred = np.vectorize(reversefactor.get)(svm_pred)
        svm_ac=accuracy_score(y_test, svm_pred)
        svm_acc.append(svm_ac)
    return(svm_acc)
    
svm_accuracy_c=check_hyperparametre()

#visualisation
acc = pd.Series(svm_accuracy_c, index=[1e-2, 1e-1, 1, 1e1, 1e2])
plt.bar(range(5), acc)
plt.ylabel('accuracy')
plt.title('comparaison des noyau du svm')
plt.xlabel('modéles')
plt.xticks(range(5), [1e-2, 1e-1, 1, 1e1, 1e2],rotation='vertical')
plt.show()





###### Enregistrement du modéle selectionne #######
######       et application sur test        #######

#Stocker le modèle formé.
#Nous allos utilisé le package joblib

svm_classifier = SVC(kernel = 'linear', C = 1)
svm_classifier.fit(x_train, y_train)

# save the model to disk
from sklearn.externals import joblib
joblib.dump(svm_classifier, r'Model\svm_model.pkl') 

# Importation des données de test
#datatest=pd.read_csv(r'Data\test-data.txt', delim_whitespace=True, skipinitialspace=True)
#print(datatest.columns)
#xtest=datatest

# importation du model du disk et paplication
loaded_model = joblib.load(r'Model\svm_model.pkl')

#Prediction sur les données test
svm_pred = loaded_model.predict(x_test) 
reversefactor = dict(zip(range(4),definitions))
svm_pred = np.vectorize(reversefactor.get)(svm_pred)

#Enregsitrement des prediction est test
i=1
with open(r'Prediction\armel_sahal_exo_2.txt', 'w') as file:
    for espece in svm_pred:
        file.write("%s\n" %espece)
        i=i+1
print(i,'prediction faite')

