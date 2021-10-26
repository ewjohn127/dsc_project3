# Understanding Seasonal Flu Vaccine Likelyhood in the United States

<img src="https://elitelv.com/wp-content/uploads/2020/01/When-The-Flu-Shot-Fails-1024x536.jpg" width="500">\

## Business Problem
In light of their new vaccination initiative, the CDC has conducted surveys on random individuals throughout the country. **The Goal:** Deliver an inferential binary classifier model to stakeholder (CDC) that determines if someone will take the Seasonal Flu vaccine based on responses to a phone survey. Predictions on future surveys can help assess public health risk by determining the percent of the population likely to get vaccinated.

## Data Understanding
The dataset was from the U.S. Department of Health and Human Services (DHHS) and consisted of phone surveys conducted in 2009. The survey contained demographic questions (age,sex,education etc.) along with opinion questions (vaccine effectiveness, flu risk, etc.) The target variable was wether or not the vaccine was taken. 47% of respondents to the survey had taken the 2009 seasonal flu vaccine.

## Data Cleaning
The dataset also included questions related to the 2009 H1N1 flu and vaccine. These input variables have been dropped from the dataset due to relevance to the business problem (seasonal flu vaccine). In addition, four input variables had high amounts of missing data (>10% missing):

|  Dropped Column        | % Nan |
|------------------------|-------|
| health_insurance       |  46%  |
| income_poverty         |  17%  |
| employment_industry    |  50%  |
| employment_occupation  |  50%  |

These columns were dropped from the dataset. Finally, the remaining columns were preprocessed by the following steps after a train/test split:

1. Missing Values Imputed Using Median
2. One Hot Encoded Categorical Variables
3. Scaled Input Variables

## Feature Selection
Recursive Feature Elimination was used for selecting the best features for each model. 

## Results
The performance metric used for optimization was ROC curve AUC for the following reasons:

* the fpr and fnr have equal importance for business problem
* the observations are balanced for each class
* binary classification problem

### Logistic Regression
The initial model chosen was logistic regression. This was due to the target variable being binary and the goal being an inferential model. The initial logistic regression model used one input feature selected by RFE which produced an ROC AUC score of .67. A chi squared test between each input variables showed significant collinearity and therefore a decision tree was chosen as the next model for adding more features to improve the ROC AUC score.

### Decision Tree
The initial decision tree model consisted of 5 input variables chosen by RFE. The model showed the most important features for someone to get the seasonal flu vaccine:

1. Scoring a 5/5 for vaccine efficacy opinion
2. Being 65+ or older
3. Doctor recommending the vaccine

### Random Forest

## Conclusion

1. Shorten survey to important opinion features to improve survey turnover
2. Education on vaccinations
3. Education for doctors on recommending vaccinations

## Future Work
