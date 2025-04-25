# Required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(nnet)
library(ggplot2)

# Load data
data <- read.csv("application_train.csv")

# Select only most important features for simplicity
selected_features <- c("TARGET", "CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT", 
                       "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED",
                       "CNT_FAM_MEMBERS", "CNT_CHILDREN")
data <- data %>% select(all_of(selected_features))

# check and handle NA values in CODE_GENDER
data <- data %>% filter(!is.na(CODE_GENDER))

# Convert CODE_GENDER to proper numeric values
data$CODE_GENDER <- ifelse(data$CODE_GENDER == "M", 1, 0)

# Fill NA with column mean
data <- data %>% mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Normalize numeric data
num_cols <- sapply(data, is.numeric)
num_cols["CODE_GENDER"] <- FALSE  # Don't scale the gender column
num_cols["TARGET"] <- FALSE       # Don't scale the target
data[num_cols] <- scale(data[num_cols])

# Split into train (75%) and test (25%)
set.seed(42)
train_idx <- createDataPartition(data$TARGET, p = 0.75, list = FALSE)
train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]

# Separate labels
train_x <- train_data %>% select(-TARGET)
train_y <- train_data$TARGET
test_x <- test_data %>% select(-TARGET)
test_y <- test_data$TARGET

# Convert train_y to factor for both models
train_y_factor <- as.factor(train_y)

# ----------------------------
# Model 1: Random Forest
# ----------------------------
rf_model <- randomForest(x = train_x, y = train_y_factor, ntree = 100)
rf_pred <- predict(rf_model, test_x)
rf_acc <- mean(as.numeric(as.character(rf_pred)) == test_y)

# ----------------------------
# Model 2: Neural Network - Use a simple logistic regression approach
# ----------------------------
# Create a matrix for neural network
train_x_matrix <- as.matrix(train_x)
test_x_matrix <- as.matrix(test_x)

# Use a single hidden layer with 10 nodes
nn_model <- nnet(x = train_x_matrix, y = train_y, size = 10, decay = 0.1, 
                 maxit = 1000, trace = FALSE, linout = FALSE)

nn_pred_prob <- predict(nn_model, test_x_matrix)
nn_pred <- ifelse(nn_pred_prob > 0.5, 1, 0)
nn_acc <- mean(nn_pred == test_y)

# ----------------------------
# Bias Analysis by Gender
# ----------------------------
# Create results data frame with proper type conversion
results <- data.frame(
  CODE_GENDER = test_data$CODE_GENDER,
  TARGET = test_data$TARGET,
  rf_pred = as.numeric(as.character(rf_pred)),
  nn_pred = nn_pred
)

# Convert CODE_GENDER to factor with proper labels
results$CODE_GENDER <- factor(results$CODE_GENDER, 
                              levels = c(0, 1), 
                              labels = c("Female", "Male"))

# Print first few rows to debug
print(head(results))

# Calculate accuracies by gender
rf_acc_gender <- results %>%
  group_by(CODE_GENDER) %>%
  summarize(accuracy = mean(rf_pred == TARGET), .groups = 'drop') %>%
  mutate(model = "Random Forest")

nn_acc_gender <- results %>%
  group_by(CODE_GENDER) %>%
  summarize(accuracy = mean(nn_pred == TARGET), .groups = 'drop') %>%
  mutate(model = "Neural Network")

acc_gender <- bind_rows(rf_acc_gender, nn_acc_gender)

# ----------------------------
# Bar Plot
# ----------------------------
ggplot(acc_gender, aes(x = model, y = accuracy, fill = CODE_GENDER)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Model Accuracy by Gender",
       x = "Model",
       y = "Accuracy",
       fill = "Gender") +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1))

# ----------------------------
# Print Overall Accuracies
# ----------------------------
cat("Random Forest Accuracy:", round(rf_acc, 4), "\n")
cat("Neural Network Accuracy:", round(nn_acc, 4), "\n")

# Print gender-specific accuracies
cat("\nGender-specific Accuracies:\n")
print(acc_gender)
