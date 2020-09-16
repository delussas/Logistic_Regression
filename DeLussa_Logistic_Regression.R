### This code builds a Logistic Regression function from scratch and tests its accuracy compared to R's built in Logistic Regression functions. Finally, it graphs a Reciever Operating Characteristic Curve using the data from 

setwd("~/Documents/Stockton Graduate/Machine_Learning/Logistic Regression")

#Load readr to open the dataset
library(readr)
#library(ROCR)
# Load ggplot2 for clean graphs
library(ggplot2)
# Load caTools to split the data
library(caTools)

# Using the read.csv function to import the dataset
data <- read.csv("LogisticData_1.csv", header = TRUE)
# Check for NA values in the data
sum(is.na(data))

# Uses seed 101 to recreate the randomized data
set.seed(101)
# Breaks up the data into a training set consiting of 80% training and 20% testing data
split = sample.split(data$Y, SplitRatio = 0.8)
train = subset(data, split == TRUE)
test = subset(data, split == FALSE)
rm(split)

# Establishes the x and y training values
x_train <- train[,-ncol(train)]
y_train <- train[3]

# Defines the Sigmoid function, which is used to keep the output value between a range of 0 and 1
sigmoid <- function(s){
  1/(1 +exp(-s))
}

# Defines the gradient function, which will be used in the logistic regression function to calculate and update the weights
gradient <- function(x_train, y_train, weights){
  m <- length(y_train)   # Total amount of observations
  x_train <- as.matrix(x_train) # Converts the x, y, and weights values into a matracies in order for them to be properly multiplied together using matrix multiplication
  y_train <- as.matrix(y_train) 
  weights <- as.matrix(weights) 
  weights <- sigmoid(x_train %*% weights) # Implements the sigmoid function on to the product of the x values and the weights, so that we can later predict the classification of the y values
  gradient <- (t(x_train)%*%(weights - y_train))/m
  return(gradient)
}
#===================================================
  # Start of my functions
  
# Defines the Logistic Regression function
Logistic_Regression <- function(x_train, y_train, iterations, learning_rate) {
 # First the x and y values are placed into a matrix then checked for "NA" values
  x_train <- as.matrix(x_train)
  y_train <- as.vector(y_train)
  x_train <- na.omit(x_train)
  y_train <- na.omit(y_train)
  # Creates a column of intercepts with the value of one that is equal to the length of the y training variable, then combines those intercepts with the x matrix.
  intercepts <- rep(1, length(y_train))
  x_train <- cbind(intercepts, x_train)
  # Creates a column of zeros that serves as the matrix for the x and y weight calculations 
  weights <- matrix(rep(0,ncol(x_train)), nrow = ncol(x_train))
  
# i establishes the starting number for the iterations
  i <- 0
# For loop to calculate, update, and recalculate the weights in the logistic regression 
  for (i in 1:iterations) {
    grad <- gradient(x_train, y_train, weights)
    weights <- weights - learning_rate * grad
    i = i + 1
  }
  return(weights)
}

# Uses the weights calculated from the Logistic Regression function to return the predicted y values
Logistic_Prediction <- function(weights, test_data) {
  # Cleans the test data by filtering out the NA values, selecting the Y column in the test data, placing it into a matrix, and finally multilpying that by the predicted weights
  test <- na.omit(test)
  test <- test[,-ncol(test_data)]
  test <- as.matrix(test)
  test <- cbind(rep(1,nrow(test)),test)
  return(sigmoid(test %*% weights))
}

# Establishes a confusion matrix which compares the Y predicted values, to the Y observed values
Confusion_Matrix <- function(test,My_Predictions){
  confusion_matrix <- table(t(test[3]),My_Predictions)
  # Quick reference to the confusion matrix subsets
  #Confusion_Matrix[1] = True  Negative
  #Confusion_Matrix[2] = False Negative
  #Confusion_Matrix[3] = False Positives
  #Confusion_Matrix[4] = True  Positives
  
  # Calculates the overall accuracy
  accuracy <- (confusion_matrix[1] + confusion_matrix[4]) / sum(confusion_matrix)
  # Calculates the sensitivity (True Positive / All Positive)
  sensitivity <- confusion_matrix[4] / (confusion_matrix[3] + confusion_matrix[4])
  # Calculates specificity (True Negative / All Negative)
  specificity <- confusion_matrix[1] / (confusion_matrix[1] + confusion_matrix[2]) 
  # Creates column headers
  Measures <- c("Accuracy", "Sensitivity", "Specificity")
  # Combines the above calculated metrics and headers
  my_values <- c(accuracy,sensitivity,specificity)
  my_metrics <- cbind(Measures, my_values)
  
  print("Confusion Matrix: ")
  print(confusion_matrix)
  
  print("My Metrics: ")
  print(my_metrics)
}

#=======================================================
  # R's built in functions 

# Creates a functin that leverages R's logistic regression functions
R_Functions <- function(train, test, threshold) {
  # Usues the generalized linear model to return the probability of an event occuring (ie: A or B, True or False, etc...) It first takes a formula argument, then a data frame, followed by the family argument. The family argument is used to describe the error distribution of the model. We are using binomial which leverages the logit function to find the logarithm of the odds
  GenLinMod <- glm(Y~., data = train, family = binomial)
  # Stores the coeffecients derived from the glm() function for later comparison to the coefficients derived from my scratch function
  R_Coeffecients <- GenLinMod$coefficients
  # Leverages the predict function to calculate the predicted probabilities. Type="response" returns R's predictions.
  R_Prediction <- predict(GenLinMod, newdata=test, type="response")
  # Creates a confusion matrix of predicted vs observed values
  R_CM <- table(t(test[3]),R_Prediction > threshold) # Quick Reference guide to the confusion matrix
  #R_CM[1] = True  Negative
  #R_CM[2] = False Negative
  #R_CM[3] = False Positives
  #R_CM[4] = True  Positives
  
  # Calculates the overall accuracy 
  accuracy <- (R_CM[1] + R_CM[4]) / sum(R_CM)
  # Calculates the sensitivity (True Positive / All Positive)
  sensitivity <- R_CM[4] / (R_CM[3] + R_CM[4])
  # Calculates specificity (True Negative / All Negative)
  specificity <- R_CM[1] / (R_CM[1] + R_CM[2])    
  # Creates column headers
  R_Measures <- c("Accuracy", "Sensitivity", "Specificity")
  # Combines the above calculated metrics and headers
  R_Value <- c(accuracy,sensitivity,specificity)
  metrics <- cbind(R_Measures, R_Value)
  print(metrics)
  
  # Creates a list, to return multiple varibles from the function
  results <- list(R_Prediction, R_Coeffecients)
  return(results)
}

# Defines a function to graph the Reciever Operating Characteristic Curve (ROC Curve). This plots the true postitive rate vs the false positive rate.
ROC <- function(train, test) {
  glm1 = glm(Y~.,data=train,family=binomial)
  TestPrediction = predict(glm1, newdata=test, type="response")
  # Creates vectors for the varibales that will store the confusion matrix metrics
  sensitivity = vector(mode = "numeric", length = 101)
  falsepositives = vector(mode = "numeric", length = 101)
  thresholds = seq(from = 0, to = 1, by = 0.01)
  for(i in seq_along(thresholds)) {
    # Establishes a table comprised of the Y predicted, Y observed, that are greater than the threshold for each iteration
    vals <- table(test$Y, TestPrediction > thresholds[i])
    # Calculates the sensitivity (True Positive / All Positive)
    sensitivity[i] = vals[4]/(vals[2]+vals[4])
    # false positives, or 1 - specificity
    falsepositives[i] = vals[3]/(vals[1]+vals[3])
  }
  # Plots the curve using the false positives vs the sensitivity in a line graph
  ggplot() + 
    geom_line(aes(falsepositives, sensitivity), colour="red") +
    geom_abline(slope = 1, intercept = 0, colour="blue") +
    labs(title="ROC Curve", x= "1 - Specificity (FP)", y="Sensitivity (TP)") +
    geom_text(aes(falsepositives, sensitivity), label=ifelse(((thresholds * 100) %% 10 == 0),thresholds,''),nudge_x=0,nudge_y=0)
}



#=======================================================

# Number of times the logistic regression will loop through the data
iterations <- 150000
# Controls the rate at which the function will learn
learning_rate <- .001
# Number set to determine whether a result will be True or False
threshold <- .5

# Runs the Logistic Regression Function created from scratch and returns the coefficients
betas <- Logistic_Regression(x_train, y_train, iterations, learning_rate)
betas

# Tests my predictions using the betas dirived from the Logistic Regression Function above, then rounds the results to either 1 or 0
My_Predictions <- round(Logistic_Prediction(betas, test))

# Creates a confusion matrix from the observed values vs the prediction my model calculated
CM <- Confusion_Matrix(test,My_Predictions)

# Retrurns R's results after using R's built in logistic regression functions
R_Caluclations <- R_Functions(train, test, threshold)

# Runs the ROC function to produce the Reciever Operating Characteristic (ROC) Curve given the training and testing datasets
ROC(train,test)
