# Classification template


# Importing the dataset

dataset = read.csv('liver.csv')
for (i in 1:300){
  if(dataset[i,11] == 2)
  {
    dataset[i,11] = 0
  }
}
dataset = dataset[2:11]
dataset$alkphos = ifelse(is.na(dataset$alkphos),
                     ave(dataset$alkphos, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$alkphos)
# Encoding the target feature as factor for classification of levels
dataset$is_patient = factor(dataset$is_patient, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$is_patient, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-10] = scale(training_set[-10])
test_set[-10] = scale(test_set[-10])


#dimension reduction
# Applying kernal PCA cuz its non linear problem
# Applying Kernel PCA
#kernal pca is unsupervized so no depened variable.
# install.packages('kernlab')
library(kernlab)
kpca = kpca(~., data = training_set[-10], kernel = 'rbfdot', features = 2)
training_set_pca = as.data.frame(predict(kpca, training_set))
training_set_pca$is_patient = training_set$is_patient
test_set_pca = as.data.frame(predict(kpca, test_set))
test_set_pca$is_patient = test_set$is_patient


ggplot(training_set_pca, aes(x = V1, y = V2, colour = is_patient)) + geom_point()
ggplot(training_set_pca, aes(x = V1, y = V2)) + geom_point() + facet_grid(~is_patient)
#after dimension reduction

# Fitting classifier to the Training set
# Create your classifier here
# Fitting Logistic Regression to the Training set
classifier = glm(formula = is_patient ~ .,
                 family = binomial,
                 data = training_set_pca)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set_pca[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set_pca[, 3], y_pred > 0.5)
cm
library(caret) 
confusionMatrix(cm)

library(ggplot2)
ggplot(data=test_set,aes(x=y_pred,fill=factor(test_set_pca[,3])))+geom_bar(position = "fill")
# Fitting logistic regression to the Training set

library(ElemStatLearn)
set = training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Fitting K-NN to the Training set and Predicting the Test set results
#less acuuracy then logistic
library(class)
y_pred = knn(train = training_set_pca[, -3],
             test = test_set_pca[, -3],
             cl = training_set_pca[, 3],
             k = 5,
             prob = TRUE)
# Making the Confusion Matrix
cm = table(test_set_pca[, 3], y_pred)
cm
library(caret) 
confusionMatrix(cm)
ggplot(data=test_set,aes(x=y_pred,fill=factor(test_set_pca[,3])))+geom_bar(position = "fill")


# Visualising the Training set results for knn
library(ElemStatLearn)
set = training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
y_grid = knn(
  train = training_set_pca[, -3], test = grid_set, cl = training_set_pca[, 3], k = 5)
plot(set[, -3],
     main = 'K-NN (Training set)',
     xlab = 'parms', ylab = 'is_patient',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Fitting Kernel SVM to the Training set
library(e1071)
classifier = svm(formula = is_patient ~ .,
                 data = training_set_pca,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set_pca[-3])

# Making the Confusion Matrix
cm = table(test_set_pca[, 3], y_pred)
cm
confusionMatrix(cm)
ggplot(data=test_set,aes(x=y_pred,fill=factor(test_set_pca[,3])))+geom_bar(position = "fill")

#worst
# install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set_pca[-3],
                        y = training_set_pca$is_patient)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set_pca[-3])

# Making the Confusion Matrix
cm = table(test_set_pca[, 3], y_pred)
cm
library(caret)
confusionMatrix(cm)
ggplot(data=test_set,aes(x=y_pred,fill=factor(test_set_pca[,3])))+geom_bar(position = "fill")


# Fitting Decision Tree Classification to the Training set
# install.packages('rpart')
library(rpart)
classifier = rpart(formula = is_patient ~ .,
                   data = training_set_pca)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set_pca[-3], type = 'class')

# Making the Confusion Matrix
cm = table(test_set_pca[, 3], y_pred)
cm
library(caret)
confusionMatrix(cm)
ggplot(data=test_set,aes(x=y_pred,fill=factor(test_set_pca[,3])))+geom_bar(position = "fill")
# Visualising the Training set results for decision tree

#plotting tree
plot(classifier)
text(classifier)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
#accuracy is 0.733 ~ 73%
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set_pca[-3],
                          y = training_set_pca$is_patient,
                          ntree = 1000)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set_pca[-3])

# Making the Confusion Matrix
cm = table(test_set_pca[, 3], y_pred)
cm
library(caret)
confusionMatrix(cm)
plot(classifier)
text(classifier)
ggplot(data=test_set,aes(x=y_pred,fill=factor(test_set_pca[,3])))+geom_bar(position = "fill")

# Visualising the Training set results
library(ElemStatLearn)
set = training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)

grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1','V2')
y_grid = predict(classifier, grid_set)
plot(set[, -3],
     main = 'Random Forest Classification (Training set)',
     xlab = 'parameters', ylab = 'Estimated patiant',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, grid_set)
plot(set[, -3], main = 'Random Forest Classification (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
