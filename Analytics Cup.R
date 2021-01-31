
# Import data and relevant libraries #
#######################################

setwd("~/Desktop/Mastermodule/BA/training_data_v2_L8PXAxZ")
library(tidyverse)
library(lubridate)
library(tidymodels)
library(vip)
options(dplyr.widh=Inf)
set.seed(2021)

companies<-read_csv('companies.csv')
payments<-read_csv('payments.csv')
physicians<-read_csv('physicians.csv')

####################################################################################################################

# Clean and Enrich Data #
##########################

## Change data type ##

payments<-payments %>% mutate(Date=mdy(Date),
                              Form_of_Payment_or_Transfer_of_Value=factor(Form_of_Payment_or_Transfer_of_Value),
                              Nature_of_Payment_or_Transfer_of_Value=factor(Nature_of_Payment_or_Transfer_of_Value),
                              Ownership_Indicator=factor(Ownership_Indicator),
                              Third_Party_Recipient=factor(Third_Party_Recipient),
                              Charity=factor(Charity),
                              Third_Party_Covered=factor(Third_Party_Covered)
                              )

physicians<-physicians %>% mutate(Primary_Specialty=factor(Primary_Specialty))

## Add a column to the physicians data frame indicating Ownership Indicator ##

physiciansid_WithOI<-payments %>% 
  filter(Ownership_Indicator=='Yes') %>% select(Physician_ID) %>% unique()
physicians <- physicians %>% 
  mutate(Ownership_Indicator=ifelse(id %in% physiciansid_WithOI$Physician_ID,'Yes',''))
physicians$Ownership_Indicator<-ifelse(physicians$Ownership_Indicator=='' & physicians$set=='train','No',physicians$Ownership_Indicator)
#physicians$Ownership_Indicator<-factor(physicians$Ownership_Indicator)

## Drop some columns that I will definetly not need. A Tableau plot also showed that the physicians with and without an Ownership Indicator
## are scattered homogeanously accross the US, so the location variable won't be used ##

physicians<-physicians %>% select(-First_Name,-Middle_Name,-Last_Name,
                                  -Name_Suffix,-City,-State,-Zipcode,
                                  -Country,-Province,-License_State_1,
                                  -License_State_2,-License_State_3,-License_State_4,-License_State_5)
physicians<-physicians[!(physicians$id==5886),]
payments<-payments[!(payments$Physician_ID==5886),]

## Add a colum to the physicians data frame indicating ratio of physicians with 
## Ownership_Indicator=Yes who work in the same speciality. 
## Tableau Plot showed that some specialities tend to have a bigger proportion of physicians with Ownership-Interest ##

Ratio_yes<- physicians %>% filter(Ownership_Indicator!='') %>% 
  group_by(Primary_Specialty,Ownership_Indicator) %>% 
  summarize(n=n()) %>% ungroup() %>% 
  group_by(Primary_Specialty) %>% 
  summarise(RatioOI_Speciality=n/sum(n), Ownership_Indicator=Ownership_Indicator) %>% 
  filter(Ownership_Indicator=='Yes') %>% select(Primary_Specialty,RatioOI_Speciality)
physicians<-left_join(physicians,Ratio_yes,by="Primary_Specialty")
physicians$RatioOI_Speciality[is.na(physicians$RatioOI_Speciality)]<-0
physicians<-physicians %>% select(-Primary_Specialty)

## Enriching the physicians dataset with attributes derived from the payments dataset. Payments 
## with Ownership Indicator=Yes are filtered out, as they are not included for physicians from the test set ##

payments_with_No_OI<-payments %>% filter(Ownership_Indicator=='No')

t1<-payments_with_No_OI %>% group_by(Physician_ID,year(Date)) %>% 
  summarise(Total_Amount_received=sum(Total_Amount_of_Payment_USDollars), 
            Total_Number_of_transactions=n(), 
            Number_of_transferring_Companies=length(unique(Company_ID))) %>% ungroup() %>% group_by(Physician_ID) %>%
            summarise(Total_Amount_received_by_Year=mean(Total_Amount_received),
                      Total_Number_of_transactions_by_Year=mean(Total_Number_of_transactions),Number_of_transferring_Companies_by_Year=mean(Number_of_transferring_Companies))
physicians<-left_join(physicians,t1,by=c("id"="Physician_ID"))

t2<-payments_with_No_OI %>% group_by(Physician_ID,Nature_of_Payment_or_Transfer_of_Value,year(Date)) %>% 
  summarize(x=sum(Total_Amount_of_Payment_USDollars)) %>% 
  pivot_wider(names_from = "Nature_of_Payment_or_Transfer_of_Value",values_from="x") 
t2[is.na(t2)] <- 0
t2<-t2 %>% group_by(Physician_ID) %>% summarise(across(c(2:14),mean))
physicians<-left_join(physicians,t2,by=c("id"="Physician_ID"))

t3<-payments_with_No_OI %>% group_by(Physician_ID,Form_of_Payment_or_Transfer_of_Value,year(Date)) %>% 
  summarize(x=sum(Total_Amount_of_Payment_USDollars)) %>% 
  pivot_wider(names_from = "Form_of_Payment_or_Transfer_of_Value", values_from="x") 
t3[is.na(t3)] <- 0
t3<-t3 %>% group_by(Physician_ID) %>% summarise(across(c(2:3),mean))
t3<-t3 %>% mutate(./rowSums(t3[,2:3]),Physician_ID=c(1:5999))
physicians<-left_join(physicians,t3,by=c("id"="Physician_ID"))

t4<-payments_with_No_OI %>% group_by(Physician_ID,Related_Product_Indicator,year(Date)) %>% 
  summarize(x=sum(Total_Amount_of_Payment_USDollars)) %>% 
  pivot_wider(names_from = "Related_Product_Indicator", values_from="x") 
t4[is.na(t4)] <- 0
t4<-t4 %>% group_by(Physician_ID) %>% summarise(across(c(2:7),mean))
t4<-t4 %>% mutate(./rowSums(t4[,2:7]),Physician_ID=c(1:5999))
physicians<-left_join(physicians,t4,by=c("id"="Physician_ID"))

## seperate the physicians column into train and test data##

train<-physicians %>% filter(set=='train') %>% select(-set)
test<-physicians%>% filter(set=='test') %>% select(-set)

#####################################################################################################

# Build the model #
################

# 1) Penalized logistic regression model:

## The glmnet R package fits a generalized linear model via penalized maximum likelihood. 
## This method of estimating the logistic regression slope parameters uses a penalty on 
## the process so that less relevant predictors are driven towards a value of zero. 

## We’ll set the penalty argument to tune() as a placeholder for now. 
## This is a model hyperparameter that we will tune to find the best value for 
## making predictions with our data. Setting mixture to a value of one means that 
## the glmnet model will potentially remove irrelevant predictors and choose a simpler model.

lr_mod <- 
  logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")

# Before we fit this model, we need to set up a grid of penalty values to tune.
lr_reg_grid <- tibble(penalty = 10^seq(-6, -3, length.out = 30))

# 2)  Random Forest Model
## The ranger package offers a built-in way to compute individual random forest models in parallel. 
## To do this, we need to know the the number of cores we have to work with. 

cores <- parallel::detectCores()
rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger", num.threads = cores) %>% 
  set_mode("classification")

#########################################################################################################

# Preprocess Data and create workflow #
#######################################

## preprocessing recipe ##

rec <- recipe(Ownership_Indicator ~ ., data = train) %>% 
  update_role(id, new_role = "ID") %>%
  step_meanimpute(all_numeric()) %>%
  step_string2factor(all_nominal(), -all_outcomes(), -has_role("ID")) %>% 
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

## training workflow ##

training_workflow <- 
  workflow() %>% 
  add_recipe(rec) %>% 
  add_model(rf_mod_final)

## resampling ##
val_set <- validation_split(train, 
                            strata = Ownership_Indicator, 
                            prop = 0.80)

######################################################################################################

# Train and Tune model #
#########################

# 1) Penalized logistic regression model 

## Train 30 penalized logistic regression models and save the validation set predictions ##
lr_res <- 
  training_workflow %>% 
  tune_grid(val_set,
            grid = lr_reg_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(bal_accuracy))

lr_plot <- 
  lr_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Balanced Accuracy") +
  scale_x_log10(labels = scales::label_number())

top_models <-
  lr_res %>% 
  show_best("bal_accuracy", n = 30) %>% 
  arrange(penalty)

lr_best <- 
  lr_res %>% 
  collect_metrics() %>% 
  arrange(penalty) %>% 
  slice(20)

# 2) Random Forest model

rf_res <- 
  training_workflow %>% 
  tune_grid(val_set,
            grid = 25,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(bal_accuracy))

rf_res %>% 
  show_best(metric = "bal_accuracy")

rf_best <- 
  rf_res %>% 
  select_best(metric = "bal_accuracy")
################################################################################

# Train the final model #
##########################
rf_mod_final <- 
  rand_forest(mtry = 16, min_n = 25, trees = 1000) %>% 
  set_engine("ranger", num.threads = cores) %>% 
  set_mode("classification")


trained_model <- training_workflow %>% 
  fit(data=train)

trained_model

train_set_with_predictions <-
  bind_cols(
    train,
    trained_model %>% predict(train)
  )
train_set_with_predictions

trained_model %>% pull_workflow_fit() %>% vip::vip()
##########################################################################3

#Predict on Test set#
#####################

submission <-bind_cols(
     test %>% select(id),
     trained_model %>% predict(test)
   ) %>% 
     # column names must be the same as in submission template!
    rename(prediction = .pred_class) %>% 
    # order by id
    arrange(id)
submission$prediction<-ifelse(submission$prediction=='No',0,1)

write_csv(submission, "Solo_Hot_Cocoa_SUBMISSION_NUMBER_1.csv")
