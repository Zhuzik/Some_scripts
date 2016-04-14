
#load data
train<-read.csv("train.csv")
test<-read.csv("test.csv")

#new features
train[, "max"] <- apply(train[, 2:94], 1, max)
train[, "mean"] <- apply(train[, 2:94], 1, mean)
train[, "length"] <- sqrt(rowSums((train[, 2:94])^2))

train<-train[sample(nrow(train)),]

X<-train[,-c(1,95)]
y<-train[,95]

#validation set
library(caret)
set.seed(212)
trainIndex <- createDataPartition(y, p = .6,
                                  list = FALSE,
                                  times = 1)
xTrain <- X[-trainIndex,]
xTest  <- X[trainIndex,]

yTrain<-y[-trainIndex]
yTest<-y[trainIndex]

#RandomForest

library(randomForest)
library(caTools)
library(foreach)
library(doMC)


registerDoMC(cores=16)

buildRFModel <- function(X,y, mtry) {
  
  gc(reset=TRUE)
  set.seed(23)
  RF <- foreach(ntree=rep(200,16), .combine=combine,
                .multicombine=TRUE,
                .packages="randomForest") %dopar% {
                  randomForest(X,
                               factor(y),
                               ntree=ntree,
                               mtry=mtry,
                               strata=factor(y),
                               do.trace=TRUE, importance=TRUE, forest=TRUE,
                               replace=TRUE)
                }
  RF
}


RFmodel<-buildRFModel(xTrain,yTrain,30)
predRF<-predict(RFmodel, newdata=xTest, type="prob")


#XGBoost
require(xgboost)
require(methods)
library(devtools)

ynew = as.integer(yTrain)-1 #xgboost take features in [0,numOfClass)

x = rbind(xTrain,xTest)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(ynew)
teind = (nrow(xTrain)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "num_class" = 9,
              "nthread" = 16,
              "min_child_weight" = 1,
              "max_depth" = 5,
              "subsample" = 0.9,
              "colsample_bytree" = 0.8,
              "gamma" = 2)

# Run Cross Validation
cv.nround = 250
bst.cv = xgb.cv(param=param, data = x[trind,], label = ynew, 
                nfold = 3, nrounds=cv.nround)

# Train the model
nround = 250
bst = xgboost(param=param, data = x[trind,], label = ynew, nrounds=nround)

# Make prediction
predXGB = predict(bst,x[teind,])
predXGB = matrix(predXGB,9,length(predXGB)/9)
predXGB = t(predXGB)
predXGB = data.frame(predXGB)
names(predXGB) = paste0('Class_',1:9)

predTotal<-1/2*predRF+1/2*predXGB
