# Bias Detection and Mitigation in Loan Prediction Models

This project analyzes gender-based bias in machine learning models trained to predict loan default risk using the Home Credit Default Risk dataset. Two different models—Random Forest and Neural Network—were implemented and evaluated to compare performance across gender groups.

## Dataset

- **Source**: [Home Credit Default Risk - Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)  
- **Size**: 30,757 samples  
  - 75% training set (23,068 samples)  
  - 25% test set (7,689 samples)  

## Preprocessing

- Ordinal encoding of categorical variables  
- Normalization using `StandardScaler` (zero mean, unit variance)  
- Missing value imputation using column mean  
- Gender column (`CODE_GENDER`) used as the protected variable for bias analysis  

## Models

### Random Forest
- 100 estimators (trees)
- Default hyperparameters
- Ensemble-based approach using multiple decision trees

### Neural Network
- Simple feedforward architecture with 1 hidden layer (10 nodes)
- Regularization applied (`decay = 0.1`) to prevent overfitting
- Trained using binary classification (0: no default, 1: default)

## Results

### Accuracy on Full Test Set
- **Random Forest**: 91.84%  
- **Neural Network**: 91.83%  

### Accuracy by Gender

| Gender | Random Forest | Neural Network |
|--------|---------------|----------------|
| Female | 92.9%         | 92.9%          |
| Male   | 89.6%         | 89.7%          |

The models demonstrate higher accuracy for female clients (~3% gap), indicating potential algorithmic bias.

## 📉 Visualization

Bar plots were generated to show the accuracy differences between gender groups for each model. These visualizations support the detection of bias in model performance.

## Discussion

The observed performance gap suggests the presence of bias, which may stem from:
- Sample imbalance or feature interactions with gender
- Differences in default behavior patterns
- Missing or unobserved predictive variables

### Possible Mitigation Strategies
- Fairness-aware training (e.g., reweighting, threshold adjustment)
- Feature engineering to include more predictive factors for underperforming groups
- More balanced or representative training data

## Project Structure

