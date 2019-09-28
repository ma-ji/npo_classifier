##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
#                                                                                                                        #
# Replication file for "Machine Learning for Public Administration Research with Application to Organizational Reputation#
# Authors: L Jason Anastasopoulos (ljanastas@uga.edu) and Andy Whitford (aw@uga.edu)                                     #
# Please contact Jason with any questions or concerns that you might have regarding this code                            #
#                                                                                                                        #
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################


#########################################################################################
#########################################################################################
################### INTERCODER RELIABILITY PLOTS ########################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################


library(foreign)
library(ggplot2)


datapath = "C:\\Users\\LJA75704\\Dropbox\\Research\\Papers\\Promise_and_perils_ML\\JPART-Final-Code-Data"

setwd(datapath)


data = read.csv("coded-tweet-data.csv")

attach(data)

# Recode the variables
#1= Performative
##2 = Moral
#3 = Procedural
##4 = Technical
#0 = None of the above

names(data)[2] = c("Text")

Coder1 = as.numeric(Answer1)
Coder2 = as.numeric(Answer2)

# Recode to match expert coding

for(i in 1:length(Coder1)){
  if(Coder1[i] == 1){Coder1[i] <- 2}
  else if(Coder1[i] == 2){Coder1[i] <- 0}
  else if(Coder1[i] == 3){Coder1[i] <- 1} 
  else if(Coder1[i] == 4){Coder1[i] <- 3} 
  else if(Coder1[i] == 5){Coder1[i] <- 4} 

  if(Coder2[i] == 1){Coder2[i] <- 2}
  else if(Coder2[i] == 2){Coder2[i] <- 0} 
  else if(Coder2[i] == 3){Coder2[i] <- 1} 
  else if(Coder2[i] == 4){Coder2[i] <- 3}
  else if(Coder2[i] == 5){Coder2[i] <- 4}
}

# Let's make a table of agreements overall and by category

Coder12 = sum(Coder1 == Coder2)/length(Coder2)
Coder1Jason = sum(Coder1 == JasonCode)/length(Coder2)
Coder2Jason = sum(Coder2 == JasonCode)/length(Coder2)


Coder12.Moral = sum(ifelse(Coder1 == 2 & Coder2 == 2,1,0),na.rm = TRUE)/sum(ifelse(Coder1 == 2 | Coder2 == 2,1,0),na.rm = TRUE)
Coder1Jason.Moral =sum(ifelse(Coder1 == 2 & JasonCode == 2,1,0),na.rm = TRUE)/sum(ifelse(Coder1 == 2 | JasonCode == 2,1,0),na.rm = TRUE)
Coder2Jason.Moral = sum(ifelse(JasonCode == 2 & Coder2 == 2,1,0),na.rm = TRUE)/sum(ifelse(JasonCode == 2 | Coder2 == 2,1,0),na.rm = TRUE)

Coder12.Technical = sum(ifelse(Coder1 == 4 & Coder2 == 4,1,0),na.rm = TRUE)/sum(ifelse(Coder1 == 4 | Coder2 == 4,1,0),na.rm = TRUE)
Coder1Jason.Technical =sum(ifelse(Coder1 == 4 & JasonCode == 4,1,0),na.rm = TRUE)/sum(ifelse(Coder1 == 4 | JasonCode == 4,1,0),na.rm = TRUE)
Coder2Jason.Technical = sum(ifelse(JasonCode == 4 & Coder2 == 4,1,0),na.rm = TRUE)/sum(ifelse(JasonCode == 4 | Coder2 == 4,1,0),na.rm = TRUE)

Coder12.Procedural = sum(ifelse(Coder1 == 3 & Coder2 == 3,1,0),na.rm = TRUE)/sum(ifelse(Coder1 == 3 | Coder2 == 3,1,0),na.rm = TRUE)
Coder1Jason.Procedural =sum(ifelse(Coder1 == 3 & JasonCode == 3,1,0),na.rm = TRUE)/sum(ifelse(Coder1 == 3 | JasonCode == 3,1,0),na.rm = TRUE)
Coder2Jason.Procedural = sum(ifelse(JasonCode == 3 & Coder2 == 3,1,0),na.rm = TRUE)/sum(ifelse(JasonCode == 3 | Coder2 == 3,1,0),na.rm = TRUE)

Coder12.Performative = sum(ifelse(Coder1 == 1 & Coder2 == 1,1,0),na.rm = TRUE)/sum(ifelse(Coder1 == 1 | Coder2 == 1,1,0),na.rm = TRUE)
Coder1Jason.Performative=sum(ifelse(Coder1 == 1 & JasonCode == 1,1,0),na.rm = TRUE)/sum(ifelse(Coder1 == 1 | JasonCode == 1,1,0),na.rm = TRUE)
Coder2Jason.Performative= sum(ifelse(JasonCode == 1 & Coder2 == 1,1,0),na.rm = TRUE)/sum(ifelse(JasonCode == 1 | Coder2 == 1,1,0),na.rm = TRUE)


Coder12.None = sum(ifelse(Coder1 == 0 & Coder2 == 0,1,0),na.rm = TRUE)/sum(ifelse(Coder1 == 0 | Coder2 == 0,1,0),na.rm = TRUE)
Coder1Jason.None=sum(ifelse(Coder1 == 0 & JasonCode == 0,1,0),na.rm = TRUE)/sum(ifelse(Coder1 == 0 | JasonCode == 0,1,0),na.rm = TRUE)
Coder2Jason.None= sum(ifelse(JasonCode == 0 & Coder2 == 0,1,0),na.rm = TRUE)/sum(ifelse(JasonCode == 0 | Coder2 == 0,1,0),na.rm = TRUE)


All = c(Coder12,Coder1Jason,Coder2Jason)
Moral = c(Coder12.Moral,Coder1Jason.Moral,Coder2Jason.Moral)
Technical = c(Coder12.Technical,Coder1Jason.Technical,Coder2Jason.Technical)
Procedural = c(Coder12.Procedural,Coder1Jason.Procedural,Coder2Jason.Procedural)
Performative = c(Coder12.Performative,Coder1Jason.Performative,Coder2Jason.Performative)
None = c(Coder12.None,Coder1Jason.None,Coder2Jason.None)

df2 = data.frame(
  Agreement = c(All, Moral, Technical, Procedural, Performative, None),
  Label = rep(c("Coder 1 and 2", "Coder 1 and Expert", "Coder 2 and Expert"),3),
  Reputation = c(
    rep("All",3),
    rep("Moral",3),
    rep("Technical",3),
    rep("Procedural",3),
    rep("Technical",3),
    rep("None",3)
  )
)

setwd("C:\\Users\\LJA75704\\Dropbox\\Research\\Papers\\Promise_and_perils_ML\\Draft\\Overleaf")

# Create a barplot
ggplot(data=df2, aes(x=Reputation, y=Agreement, fill=Label)) +
  geom_bar(stat="identity", color="black", position=position_dodge())+
  theme_minimal() +  scale_fill_brewer(palette="Blues") +
  ggtitle("Intercoder Reliability By Organizational Reputation Type") + xlab("Reputation Type") + 
  ylab("Agreement (% of Tweets)")

ggsave("intercoder.png")

#########################################################################################
#########################################################################################
################### ALGORITHM TRAINING ##################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

setwd(datapath)
# Train the algorithm using labels from all 3 coders
library(pacman)

# This loads and installs the packages you need at once
pacman::p_load(tm,SnowballC,foreign,plyr,
               twitteR,slam,foreign,
               caret,ranger,rpart,rpart.plot, xgboost,e1071)

source("textcleaner.R") # loads the textcleaning function

cleantweets<-text_cleaner(data$Text)


# Create TF-IDF
dtm<-DocumentTermMatrix(cleantweets)
dtm<-removeSparseTerms(dtm,sparse=0.98)
dtm_mat<-as.matrix(dtm)



moral1<-ifelse(Coder1 == 2 , 1, 0)
moral2<-ifelse(Coder2 == 2 , 1, 0)
moralJason<-ifelse(JasonCode == 2 , 1, 0)



############## CODER 1
moral = moral1

set.seed(41616)
mllabel = data.frame(moral,dtm_mat)

train=sample(1:dim(mllabel)[1],
             dim(mllabel)[1]*0.7)
dtm_mat<-as.matrix(dtm)
trainX = dtm_mat[train,]
testX = dtm_mat[-train,]
trainY = moral[train]
testY = moral[-train]

traindata<-data.frame(trainY,trainX)
testdata<-data.frame(testY,testX)

traindata.b <- xgb.DMatrix(data = trainX,label = trainY) 
testdata.b  <- xgb.DMatrix(data = testX,label=testY)

pospredweight = as.vector(table(trainY)[1])/as.vector(table(trainY)[2])

set.seed(100)

# Parameter tuning
# these are default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
               subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = traindata.b,
                 nrounds = 100, nfold = 5, showsd = T, 
                 stratified = T, early.stopping.rounds = 20, print.every_n = 10,
                 maximize = F,
                 scale_pos_weight = pospredweight)

# Which number of iterations has the lowest training error?
best.iter = which(xgbcv$evaluation_log$test_error_mean ==  min(xgbcv$evaluation_log$test_error_mean))
best.iter = best.iter[1]

#first default - model training
xgb1 <- xgb.train(params = params, data = traindata.b, 
                  nrounds = best.iter, 
                  watchlist = list(val=testdata.b,train=traindata.b),
                  print.every_n = 10, early_stopping_rounds = 10, 
                  maximize = F , eval_metric = "error",
                  scale_pos_weight = pospredweight)

#model prediction
xgbpred <- predict(xgb1,testdata.b)
xgbpred <- ifelse(xgbpred > 0.5,1,0)

perf1 = confusionMatrix(as.factor(xgbpred), as.factor(testY),positive="1")




############## CODER 2
moral = moral2

set.seed(41616)
mllabel = data.frame(moral,dtm_mat)

train=sample(1:dim(mllabel)[1],
             dim(mllabel)[1]*0.7)
dtm_mat<-as.matrix(dtm)
trainX = dtm_mat[train,]
testX = dtm_mat[-train,]
trainY = moral[train]
testY = moral[-train]

traindata<-data.frame(trainY,trainX)
testdata<-data.frame(testY,testX)

traindata.b <- xgb.DMatrix(data = trainX,label = trainY) 
testdata.b  <- xgb.DMatrix(data = testX,label=testY)

pospredweight = as.vector(table(trainY)[1])/as.vector(table(trainY)[2])

set.seed(100)

# Parameter tuning
# these are default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
               subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = traindata.b,
                 nrounds = 100, nfold = 5, showsd = T, 
                 stratified = T, early.stopping.rounds = 20, print.every_n = 10,
                 maximize = F,
                 scale_pos_weight = pospredweight)

# Which number of iterations has the lowest training error?
best.iter = which(xgbcv$evaluation_log$test_error_mean ==  min(xgbcv$evaluation_log$test_error_mean))
best.iter = best.iter[1]

#first default - model training
xgb1 <- xgb.train(params = params, data = traindata.b, 
                  nrounds = best.iter, 
                  watchlist = list(val=testdata.b,train=traindata.b),
                  print.every_n = 10, early_stopping_rounds = 10, 
                  maximize = F , eval_metric = "error",
                  scale_pos_weight = pospredweight)

#model prediction
xgbpred <- predict(xgb1,testdata.b)
xgbpred <- ifelse(xgbpred > 0.5,1,0)

perf2 = confusionMatrix(as.factor(xgbpred), as.factor(testY),positive="1")



############## Expert coder
moral = moralJason

set.seed(41616)
mllabel = data.frame(moral,dtm_mat)

train=sample(1:dim(mllabel)[1],
             dim(mllabel)[1]*0.7)
dtm_mat<-as.matrix(dtm)
trainX = dtm_mat[train,]
testX = dtm_mat[-train,]
trainY = moral[train]
testY = moral[-train]

traindata<-data.frame(trainY,trainX)
testdata<-data.frame(testY,testX)

traindata.b <- xgb.DMatrix(data = trainX,label = trainY) 
testdata.b  <- xgb.DMatrix(data = testX,label=testY)

pospredweight = as.vector(table(trainY)[1])/as.vector(table(trainY)[2])

set.seed(100)

# Parameter tuning
# these are default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
               subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = traindata.b,
                 nrounds = 100, nfold = 5, showsd = T, 
                 stratified = T, early.stopping.rounds = 20, print.every_n = 10,
                 maximize = F,
                 scale_pos_weight = pospredweight)

# Which number of iterations has the lowest training error?
best.iter = which(xgbcv$evaluation_log$test_error_mean ==  min(xgbcv$evaluation_log$test_error_mean))
best.iter = best.iter[1]

#first default - model training
xgb1 <- xgb.train(params = params, data = traindata.b, 
                  nrounds = best.iter, 
                  watchlist = list(val=testdata.b,train=traindata.b),
                  print.every_n = 10, early_stopping_rounds = 10, 
                  maximize = F , eval_metric = "error",
                  scale_pos_weight = pospredweight)

#model prediction
xgbpred <- predict(xgb1,testdata.b)
xgbpred <- ifelse(xgbpred > 0.5,1,0)

perfJason = confusionMatrix(as.factor(xgbpred), as.factor(testY),positive="1")

## Variable importance plot
mat <- xgb.importance(feature_names = colnames(trainX),model = xgb1)

#setwd("C:\\Users\\LJA75704\\Dropbox\\Research\\Papers\\Promise_and_perils_ML\\Draft\\Overleaf\\figs")
png("term-importance.png")
xgb.plot.importance(importance_matrix = mat[1:10],
                    xlab = "Information Gain",
                    ylab = "Term")
dev.off()


### Let's put together all of the relevant performance statistics

perf1$byClass[c(11,1,2,5,6,7)]
perf2$byClass[c(11,1,2,5,6,7)]
perfJason$byClass[c(11,1,2,5,6,7)]


################################################################################################
################################################################################################
################################################################################################
################################################################################################
# Apply trained expert classifier to data
################################################################################################
################################################################################################
################################################################################################
################################################################################################

twitterdata<-read.csv("agency_tweets_database.csv")
tweetstext = twitterdata$tweet_text

# Clean the text
#tweetstext = sapply(tweetstext,toString)

cleanads<-text_cleaner(tweetstext)


# Create TF-IDF
dtm_all_ads<-DocumentTermMatrix(cleanads)
dtm_all_mat<-as.matrix(dtm_all_ads)

# First step is to match the columns to subset 
# Note: "trainX" will be the training data for whichever algorithm you trained last. 
# Thus, if you trained on the "expert" coder data, this will be the training set here.

colnames<-colnames(trainX)
#fullnames<-dtm_all_ads$dimnames$Terms
fullnames<-colnames(dtm_all_mat)
indexno<-c()

for(i in 1:length(colnames)){
  tempname = colnames[i]
  indexnum = which(tempname == fullnames)
  indexno = c(indexno, indexnum)
}

dtm_mat_class<-dtm_all_mat[,indexno]

# Do the reverse because of matching problems
colnames<-colnames(dtm_mat_class)
#fullnames<-dtm_all_ads$dimnames$Terms
fullnames<-colnames(trainX)
indexno<-c()

for(i in 1:length(colnames)){
  tempname = colnames[i]
  indexnum = which(tempname == fullnames)
  indexno = c(indexno, indexnum)
}

trainX = trainX[,indexno]



# Now we can predict using the model
xgbpred <- predict(xgb1,dtm_mat_class)

# Create a plot of the predicted probabilities for 1978 and 1982

xgbpred.class <- ifelse(xgbpred > 0.5,1,0)

####################################################################################
####################################################################################
####################################################################################
# Now let's do summaries by agency for % of tweets related to moral reputation
agencies<-levels(factor(twitterdata$agency_id))
full.agencies = twitterdata$agency_id

pctmoral<-c()
lowerbound<-c()
upperbound<-c()

for(agency in agencies){
  moraltemp = xgbpred.class[full.agencies == agency]
  p = mean(moraltemp,na.rm = TRUE)
  cis = p + c(-qnorm(0.975),qnorm(0.975))*sqrt((1/length(moraltemp))*p*(1-p))
  lower = cis[1]
  upper = cis[2]
  
  pctmoral<-c(pctmoral,p)
  lowerbound<-c(lowerbound, lower)
  upperbound<-c(upperbound, upper)
}

# Create a ggplot with the 95% confidence intervals
setwd("~/Dropbox/Research/Papers/Promise_and_perils_ML/Draft/figs")

df = data.frame(
  Percent.Moral = pctmoral,
  Lower = lowerbound,
  Upper = upperbound,
  Agency = agencies
)


# Show the between-S CI's in red, and the within-S CI's in black
ggplot(df, aes(x=reorder(Agency,Percent.Moral), y=Percent.Moral, group=1)) +
  geom_errorbar(width=.2, aes(ymin=Lower, ymax=Upper), colour="red") +
  geom_point(shape=21, size=3, fill="black") + xlab("Agency") + 
  ylab("% Moral Reputation Tweets") + coord_flip() +
  theme_bw() + theme(axis.text=element_text(size=15),
                     axis.title=element_text(size=14,face="bold")) 

ggsave("pctmoralall.png")

# Remove veteran affairs
df2<-df[-c(3),]

ggplot(df2, aes(x=reorder(Agency,Percent.Moral), y=Percent.Moral, group=1)) +
  geom_errorbar(width=.2, aes(ymin=Lower, ymax=Upper), colour="red") +
  geom_point(shape=21, size=3, fill="black") + xlab("Agency") + 
  ylab("% Moral Reputation Tweets") + coord_flip() +
  theme_bw() + theme(axis.text=element_text(size=15),
                     axis.title=element_text(size=14,face="bold")) 

ggsave("pctmoraltrunc.png")






















