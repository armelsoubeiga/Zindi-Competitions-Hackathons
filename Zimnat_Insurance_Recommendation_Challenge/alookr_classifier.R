#https://github.com/choonghyunryu/alookr

# ==== Library
library(alookr)
library(dlookr)
library(dplyr)


# === load data set 
train_cl <- read.csv("C:/Users/aso/Downloads/ZimnatInsurance/data/clean_data/train_cl.csv", 
                     sep=";")

train_cl <- train_cl[, -c(1)]

# === Veification des données
names(train_cl)
summary(train_cl)
sapply(train_cl, function(x) class(x)[1])

#
train_cl$Label <- as.factor(train_cl$Label)



# === Traitement des données manquante
diagnose(train_cl) %>%
  filter(missing_count >0)





# === Splite train and test set 
tmp <- train_cl %>%
  split_by(Label, ratio = 0.7)
summary(tmp)


# list of categorical variables in the train set 
# that contain missing levels
nolevel_in_train <- tmp %>%
  compare_target_category() %>% 
  filter(train == 0) %>% 
  select(variable) %>% 
  unique() %>% 
  pull
nolevel_in_train




# === imbalanced traget
#Surechantilonnage la classe minoritaire
train_over <- tmp %>%
  sampling_target(method = "ubOver")
table(train_over$Label)


#Generation de donne supplementaire
train_smote <- tmp %>%
  sampling_target(seed = 1234L, method = "ubSMOTE")
table(train_smote$Label)


## ======== use train_over
# Clean data set 
train <- train_over %>%
  cleanse

test <- tmp %>%
  extract_set(set = "test")

# fit model
result <- train %>% 
  run_models(target = "Label", positive = 1)
result

