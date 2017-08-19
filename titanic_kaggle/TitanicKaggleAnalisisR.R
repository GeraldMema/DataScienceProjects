train <- read.csv("dataset/train.csv",header = TRUE)
test <- read.csv("dataset/test.csv",header = TRUE)

test.survived <- data.frame(Survived = rep("None" , nrow(test)),test[,])

data.combined <- rbind(train,test.survived)

str(data.combined)
str(train)

data.combined$Survived <- as.factor(data.combined$Survived)
data.combined$Pclass <- as.factor(data.combined$Pclass)

library(ggplot2)
train$Pclass<-as.factor(train$Pclass)
ggplot(train, aes(x=Pclass , fill = factor(Survived))) + 
  stat_count(width = 0.5) +
  xlab("Pclass") +
  ylab("Total counts")+
  labs(fill="Survived")

ggplot(train, aes(x=Sex=="female" & Pclass==3,fill = factor(Survived))) + 
  stat_count(width = 0.5) +
  xlab("Sex") +
  ylab("Total counts")+
  labs(fill="Survived")
