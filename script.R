library(readr)
library(party)
library(caret)
library(ggplot2) # Data visualization
library(pROC)
options(warn=-1)
Indian_Liver_Patient_Dataset_ILPD_ <- read_csv("../input/Indian Liver Patient Dataset (ILPD).csv")
df=data.frame(Indian_Liver_Patient_Dataset_ILPD_)
df[,"gender"]=as.factor(df[,"gender"])
meanOfAlk = mean(df$alkphos,na.rm = T)
df[,"alkphos"][is.na(df[,"alkphos"])] <- meanOfAlk
smp_size <- floor(0.70 * nrow(df))
set.seed(123)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

train <- df[train_ind, ]
test <- df[-train_ind, ]

library(rpart)
library(rpart.plot)
library(caret)

#----------------------------------------------------------
# decision Tree Model
# Accuracy-.72
#Here the tree is pruned before construction, we have set a limit on the height of the tree and
#information gain is based on the gini index
dt <- rpart(is_patient~.,data=train,method = "class", control = rpart.control(cp=0.02,maxdepth = 5),parms = list(split="gini"))
dt
rpart.plot(dt,cex=0.6)

printcp(dt)
plotcp(dt,minline = T)

# Pruning the tree
dt1<-prune(dt,cp=0.023)

rpart.plot(dt1,main = "Classification Tree for Liver Disease", cex=0.7)

pred<- predict(dt1,newdata = test,type="class")
print(confusionMatrix(pred,test$is_patient))

#Plotting RoC Curve
library(ROCR)
pred1 <- prediction(as.numeric(pred),as.numeric(test$is_patient))
perf<-performance(pred1,"tpr","fpr") #tpr=TP/P fpr=FP/N
plot(perf,col="blue")
abline(0,1, lty = 8, col = "grey")

auc<-performance(pred1,"auc")
auc<-unlist(slot(auc,"y.values"))
auc

df1=df[complete.cases(df),]
smp_size <- floor(0.70 * nrow(df1))
set.seed(123)
train_ind <- sample(seq_len(nrow(df1)), size = smp_size)

train1 <- df1[train_ind, ]
test1 <- df1[-train_ind, ]
train1=train1[complete.cases(train1),]
test1=test1[complete.cases(test1),]

#----------------------------------------------------------
# logistic Model
# Accuracy-.73
for(i in range(1:nrow(test))){
  if(test[i,"is_patient"]==1){
    test1[i,"is_patient"]=0
  }
  else{
    test1[i,"is_patient"]=1
  }
}

for(i in range(1:nrow(train))){
  if(train[i,"is_patient"]==1){
    train1[i,"is_patient"]=0
  }
  else{
    train1[i,"is_patient"]=1
  }
}
train1[,"is_patient"]=as.factor(train1[,"is_patient"])
test1[,"is_patient"]=as.factor(test1[,"is_patient"])

model=glm(is_patient~.,family= binomial(link="logit"),data=train1)
summary(model)


fitted.results <- predict(model,newdata=subset(test1,select=c(seq(1:11))),type='response')
fitted.results1 <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results1 != test1$is_patient)
print(paste('Accuracy',1-misClasificError))

plot(roc(as.numeric(test1$is_patient),as.numeric(fitted.results1)),
     main = "ROC Curve For Logistic")
     
     
liver <-  read_csv("../input/Indian Liver Patient Dataset (ILPD).csv")
df=data.frame(Indian_Liver_Patient_Dataset_ILPD_)
library(e1071)
#gives column name which have NA value
a = colnames(liver)[ apply(liver, 2, anyNA) ] 

#processing the data assigning mean value to all the NA values of Alkphos
meanOfAlk = mean(liver$alkphos,na.rm = T)
liver[,"alkphos"][is.na(liver[,"alkphos"])] <- meanOfAlk

#dividing data set into trainig and testing dataset
intrain <- createDataPartition(liver$is_patient, p= 0.7, list = FALSE)
training <- liver[intrain,]
testing <- liver[-intrain,]

#convering is_patitient into levels so that they can be classified
training$is_patient <- as.factor(training$is_patient)

#----------------------------------------------------------
# SVM Model
# Accuracy-.68
# This model is not good as it can't distict the classses accuratly with its
# plane.

#svm and knn model on the training set
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
svm_Linear <- train(is_patient~., data = training, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"))
svm_pred <- predict(svm_Linear, newdata = testing)
svm_pred_parameter <- confusionMatrix(svm_pred,testing$is_patient)
print(svm_pred_parameter)

#Use the predictions on the data



#Plot the predictions and the plot to see our model fit


#----------------------------------------------------------
# knn Model
# Accuracy-.706
knn <- train(is_patient~., data = training, method = "knn",
             trControl=trctrl,
             preProcess = c("center", "scale"))
knn_pred <- predict(knn, newdata = testing)
knn_pred_parameter <- confusionMatrix(knn_pred,testing$is_patient)
print(knn_pred_parameter)
plot(roc(as.numeric(testing$is_patient),as.numeric(svm_pred)),
     main = "ROC Curve For SVM")

plot(roc(as.numeric(testing$is_patient),as.numeric(knn_pred)),
     main = "ROC Curve For KNN")
     
     
# Based on the ROC curves we can decide that decision trees is the best classifier     
     
