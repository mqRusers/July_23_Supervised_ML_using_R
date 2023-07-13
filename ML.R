# reading the dataset
#read.csv("C:/Users/amara/OneDrive - Macquarie University/CRF/R-workshop_july2023/small_molecules_dataset.csv")
small_moecules <- read.csv("C:/Users/amara/OneDrive - Macquarie University/CRF/R-workshop_july2023/small_molecules_dataset.csv")
# Viewing the data dimensions
dim(small_moecules)
#viewing the dataset
View(small_moecules)

# Step 1: Missing Value Filter
#identifying positions for the missing values
which(is.na(small_moecules))
#total number of missing values
sum(is.na(small_moecules))
#removing the missing values
small_moecules_new <- na.omit(small_moecules) #it omits all rows with missing data
dim(small_moecules_new) # all rows deleted
#keeping all rows and deleting columns (features) with missing data
small_moecules_MVF <- small_moecules[ , colSums(is.na(small_moecules))==0]
# checking the dimensions of data after missing value filter
dim(small_moecules_MVF)

#Step 2: Removing the low variance data
library(caret)
#checking which columns have low variance (near to zero) and storing into a vector
nzv <- nearZeroVar(small_moecules_MVF)
nzv
# keeping columns (features) with variance
small_moecules_MVF_LVF <- small_moecules_MVF[, -nzv]
# checking dimensions
View(small_moecules_MVF_LVF)
dim(small_moecules_MVF_LVF)
str(small_moecules_MVF_LVF)
#Ste 3: removing the highly correalted data
#coverting data to numeric values
small_moecules_MVF_LVF_num <- small_moecules_MVF_LVF[sapply(small_moecules_MVF_LVF, is.numeric)]
View(small_moecules_MVF_LVF_num)
#computing the correlation
small_moecules_MVF_LVF_num_cor <- cor(small_moecules_MVF_LVF_num)
small_moecules_MVF_LVF_num_cor
#visualizing the upper correlation matrix for first 5 features of first five data points
library(corrplot)
corrplot(small_moecules_MVF_LVF_num_cor[1:5,1:5], method="number")
#removing the upper-triangle and diagonals to retain unique values in the matrix
cor_matrix_rm <- small_moecules_MVF_LVF_num_cor 
cor_matrix_rm[upper.tri(cor_matrix_rm)] <- 0
diag(cor_matrix_rm) <- 0
cor_matrix_rm
#visulizing the updated matrix
corrplot(cor_matrix_rm[1:5,1:5], method="number")
#removing the highly correlated features
small_moecules_MVF_LVF_num_remcor <- small_moecules_MVF_LVF_num[ , !apply(cor_matrix_rm, 2,function(x) any(x > 0.70))]   
# Remove highly correlated variables
dim(small_moecules_MVF_LVF_num_remcor)
View(small_moecules_MVF_LVF_num_remcor)
#integrating features with classification data i.e. class
class <- small_moecules[, c(2)]
class
filtered_data <- cbind(class,small_moecules_MVF_LVF_num_remcor)
View(filtered_data)
#Step 4: splitting into training and test data
library(caTools)
set.seed(100)
split_data <- sample.split(Y= filtered_data, SplitRatio = 0.7)
traindata <-filtered_data[split_data,]
testdata <-filtered_data[!split_data,]
View(testdata)
dim(traindata)
dim(testdata)
#removing missing values in traindata
traindata_filter <- na.omit(traindata)
testdata_filter <- na.omit(testdata)
View(traindata_filter)
#visualizing the data to see the class imbalance
table(traindata_filter$class)
#visualizing the class imbalance
barplot(table(traindata_filter$class), col = rainbow(2))
#Step 5: dealing with imbalanced dataset through oversampling
library(ROSE)
oversampled_data <- ovun.sample(class ~., data=traindata_filter, method = "over", N=374)$data
table(oversampled_data$class)
View(oversampled_data)
#for next step (RFE), the response variable should be numeric or factor
str(oversampled_data$class)
oversampled_data$class<-as.factor(oversampled_data$class)
str(oversampled_data$class)
#building the random forest model after preprocessing (feature selection through filter methods)
library(randomForest)
set.seed(123) 
rf_all <- randomForest(class~., data = oversampled_data) 
rf_all
#apply model on test data
test_predict <- predict(rf_all, testdata_filter)
#generating the confusion matrix
testdata_filter$class <- as.factor(testdata_filter$class)
confusionMatrix(test_predict, testdata_filter$class)


#Step 6(a): feature selection through automatic recursive feature elimination
library(mlbench)
library(caret)
library(randomForest)
#setting controls
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
#executing rfe
results <- rfe(oversampled_data[,2:102], oversampled_data[,1], sizes=c(2:102), rfeControl=control)
#printing top 10 variables, by default five are printed
print(results, top = 5)
#plotting the varibale and impact on accuracy
plot(results, type= "o")
#displaying the variables (most to least important)
predictors(results)

#building the random forest model on RFE selected features
#top five features
rf_rfe <- randomForest(class~ JGI6 + FilterItLogS + ZMIC5 + mZagreb2 + GATS3are, data = oversampled_data) 
rf_rfe
#apply model on test data
test_predict <- predict(rf_rfe, testdata_filter)
#generating the confusion matrix
confusionMatrix(test_predict, testdata_filter$class)
# Try taking 10 features


# Step 6(b) Selecting the features through Boruta package
library(Boruta)
set.seed(1)
#gradually increase the value of maxruns, doTrace =2 is to print the progress
boruta <- Boruta(class ~ ., data = oversampled_data, doTrace = 2, maxRuns = 50)
print(boruta)
plot(boruta, las = 2, cex.axis = 0.5)
plotImpHistory(boruta) # to see if there are any important variables that were part of the blue line (shadow variable) and might not be important
attStats(boruta) #faeture-wise detail
#visulaize important feature
#Step 7 : building the random forest model with default parameters

#building the random forest model with Boruta suggested features
# getting the Boruta suggested fetures
boruta_features <- getConfirmedFormula(boruta)
boruta_features
rf_boruta <- randomForest(class~ ATSC4c + ATSC8c + ATSC4d + ATSC7d + 
                            ATSC4are + ATSC5are + ATSC6are + ATSC7are + ATSC8are + ATSC4i + 
                            ATSC5i + ATSC6i + ATSC7i + ATSC8i + MATS1c + MATS2d + MATS3d + 
                            MATS1s + MATS1se + MATS1are + MATS2are + MATS3are + MATS1i + 
                            MATS3i + GATS1c + GATS3d + GATS2m + GATS3m + GATS1are + GATS2are + 
                            GATS3are + GATS2i + GATS3i + BCUTc.1l + BCUTdv.1l + BCUTd.1l + 
                            BCUTs.1l + BCUTare.1h + BCUTi.1h + BCUTi.1l + BalabanJ + 
                            SM1_Dzv + nBondsD + RPCG + C1SP2 + Mi + NdssC + SdsCH + SdssC + 
                            SaasC + SssO + AETA_alpha + AETA_eta_L + ETA_psi_1 + BIC5 + 
                            CIC5 + MIC5 + ZMIC5 + FilterItLogS + PEOE_VSA2 + PEOE_VSA7 + 
                            PEOE_VSA8 + PEOE_VSA9 + PEOE_VSA10 + SMR_VSA6 + SlogP_VSA2 + 
                            SlogP_VSA3 + SlogP_VSA4 + EState_VSA2 + EState_VSA3 + EState_VSA4 + 
                            EState_VSA6 + EState_VSA7 + EState_VSA8 + EState_VSA9 + AMID_C + 
                            piPC10 + n6aRing + RotRatio + TopoPSA.NO. + TopoPSA + GGI9 + 
                            JGI3 + JGI4 + JGI5 + JGI6 + JGI7 + JGI8 + JGT10 + PetitjeanIndex + 
                            AMW + mZagreb2, data = oversampled_data) 
rf_boruta
p <- predict(rf_boruta , testdata_filter)
confusionMatrix(p, testdata_filter$class)
# Step 6(c) Selecting the features Variable importance by RF
control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(class~., data=oversampled_data, method= "rf", trControl=control)
importance <- varImp(model, scale=TRUE)
print(importance, top = 20)
plot(importance, top = 20)
# Building the RF model using variable importance suggested variables
rf_vi <- randomForest(class ~ FilterItLogS + mZagreb2 + JGI6 +
                        ATSC7d + ATSC7are +AMID_C + JGI7 + JGI5 +
                        JGI5 + ATSC6are, data = oversampled_data)
rf_vi
p <- predict(rf_vi , testdata_filter)
confusionMatrix(p, testdata_filter$class)
# Hyperparameterizing  RF
#getting to know how many features are required
t <- tuneRF(oversampled_data[,2:102], oversampled_data[,1],
            stepFactor = 0.5,
            plot = TRUE,
            ntreeTry = 500,
            trace =TRUE,
            improve = 0.05)
rf2 <- randomForest(class~., data = oversampled_data, ntree=500, method= "cv", mtry=10, number=10, importance= TRUE)
plot(rf2)

rf_final <- randomForest(class~ FilterItLogS + mZagreb2 + JGI6 +
                     ATSC7d + ATSC7are +AMID_C + JGI7 + JGI5 +
                     JGI5 + ATSC6are  , data = oversampled_data, method= "cv", number=10, ntree=140, mtry=10, importance= TRUE) 
rf_final
p <- predict(rf_final , testdata_filter)
confusionMatrix(p, testdata_filter$class)

If you need further help please Contact me @
amara.jabeen@mq.edu.au


