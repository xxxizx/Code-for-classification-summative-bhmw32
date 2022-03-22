#install all the packages we need
install.packages("dplyr")
install.packages("skimr")
install.packages("psych")
install.packages("rpart")
install.packages("maptree")
install.packages("randomForest")
install.packages("pROC")


#read data from given website
banks <- read.csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
#View(banks)


#move Personal.Loan to the first column
library(dplyr)
banks <- banks %>% 
  relocate(Personal.Loan)
#view simple data summary through skimr package
library(skimr)
skim(banks)


#pairwise scatter plot
library(psych)
pairs.panels(banks[,2:13])


#partitioning into training (50%), validation (25%), and test (25%) sets
#set seed and randomly sample 50% of the row IDs for training
set.seed(1234)
train.rows <- sample(1:nrow(banks), dim(banks)[1]*0.5)
df.train <- banks[train.rows,]
remain <- banks[-train.rows,]
#assign the 25% of the remain to validation
set.seed(1234)
valid.rows <- sample(1:nrow(remain), dim(banks)[1]*0.25)
df.valid <- remain[valid.rows,]
#assign the last 25% to test
df.test <- remain[-valid.rows,]
#View(df.train)
#View(df.valid)
#View(df.test)
#model fitting using logistic regression
fit.logit <- glm(Personal.Loan~., data = df.train, family = binomial())
summary(fit.logit)
#apply this model to test set
prob <- predict(fit.logit, df.test, type = "response")
logit.pred <- factor(prob > .5, levels=c(FALSE, TRUE), labels = c("Not Upsold", "Upsold"))
logit.perf <- table(df.test$Personal.Loan, logit.pred, dnn = c("Actual", "Predicted"))
logit.perf


#CART model implementation
library(rpart)
library(maptree)
set.seed(1234)
#generate the tree
dtree <- rpart(Personal.Loan ~ ., data = df.train, method = "class")
dtree$cptable
draw.tree(dtree)
plotcp(dtree)
#prune
#?rpart
dtree.pruned <- prune(dtree, cp = dtree$cptable[which.min(dtree$cptable[,"xerror"]),"CP"])
dtree.pruned$cptable
draw.tree(dtree.pruned)
plotcp(dtree.pruned)
dtree.pred <- predict(dtree, df.test, type = "class")
dtree.perf <- table(df.test$Personal.Loan, dtree.pred, dnn = c("Actual", "Predicted"))
dtree.perf
prune.pred <- predict(dtree.pruned, df.test, type = "class")
prune.perf <- table(df.test$Personal.Loan, prune.pred, dnn = c("Actual", "Predicted"))
prune.perf


#random forest model
library(randomForest)
set.seed(1234)
df.train$Personal.Loan <- factor(df.train$Personal.Loan)
fit.forest <- randomForest(Personal.Loan ~ ., data = df.train, importance = TRUE)
fit.forest
#given the importance of variables
importance(fit.forest, type=2)
forest.pred <- predict(fit.forest, df.test)
forest.perf <- table(df.test$Personal.Loan, forest.pred, dnn = c("Actual", "Predicted"))
forest.perf


#perfomance evaluation
performance <- function(table, n=3){
  if(!all(dim(table) == c(2 ,2)))
    stop("Should be 2*2 table")
  tn = table[1,1]
  fp = table[1,2]
  fn = table[2,1]
  tp = table[2,2]
  sensitivity = tp/(tp+fn)
  specificity = tn/(tn+fp)
  ppp = tp/(tp+fp)
  npp = tn/(tn+fn)
  hitrate = (tp+tn)/(tn+fp+fn+tp)
  result <- paste("Sensitivity = ", round(sensitivity, n),
                  "\nSpecificity = ", round(specificity, n),
                  "\nPositive Predictive Value = ", round(ppp, n),
                  "\nNegative Predictive Value = ", round(npp, n),
                  "\nAccuracy = ", round(hitrate, n), "\n", sep = "")
  cat(result)
}
performance(prune.perf)
#plot ROC
library(pROC)
tree.preds <- predict(dtree.pruned, df.test, type = "prob")[, 2]
tree.roc <- roc(df.test$Personal.Loan, tree.preds)
plot(tree.roc,print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, grid=c(0.1, 0.2), 
     grid.col=c("green", "red"), max.auc.polygon=TRUE, 
     auc.polygon.col="skyblue", print.thres=TRUE)