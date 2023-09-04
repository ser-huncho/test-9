# Home Credit Loan Default Risk Prediction

## Project Overview
As a lending company, it is important to know the applicants' repayment abilities so that the company can decide whether to provide a loan to those applicants or not based on their data.
- Predict how capable each applicant is of repaying a loan based on their data such as applicant's income, credit amount of loan, loan annuity, price of the goods for which the loan is given, applicant's age, type of organization where the applicant works, etc.
- The dataset was from Project-based Internship of Home Credit Indonesia x Rakamin, but also accessible on Kaggle [here](https://www.kaggle.com/competitions/home-credit-default-risk/data). I only used the 'application_train.csv' data for this project.

## Objectives
* The objective is to make a loan default risk prediction based on several applicant variables. As a result, the model can help the company to know the ability of the applicant so they can decide whether to give a loan or not.

## Methodology  
- **Preprocessing Data**

  Checked the duplicate data, changed the data types, handled missing values (deleted columns that have >50% missing values, removed columns that only have 1 class, imputed using data mean for the numerical variables and mode for the categorical variables), did feature selection (used ANOVA and Chi-Square for selecting the correlated variables), handled categorical data (used frequency encoding to not enlarge the data dimensions), handled imbalanced data (used undersampling for majority class), and did the feature scaling for the numerical variables (used standardization).

- **Modeling and Evaluation**

  The methods used were Logistic Regression and Random Forest Classifier using the F1-score of 5-fold cross-validation as an evaluation model. Also only used Random Forest features importance for the final variables.

## Application
End-users can predict the applicants' repayment abilities based on these variables:
- **Income of the applicant**
- **Credit amount of the loan**
- **Loan annuity**
- **Price of the goods**
- **Region population relative:** Normalized population of region where the applicant lives
- **Applicant's age**
- **Day of applying for the loan:** Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
- **Hour of applying for the loan:** 1 up to 24
- **Organization type where the applicant works:** Transport, Medicine, Construction, Housing, Services, Bank, Insurance, School, Telecom, and many more
- **Number of enquiries to credit bureau**
- **Days of employed:** How many days before the application did the applicant start his current employment
- **Days of registration** How many days before the application did the applicant change his registration
- **Days of ID publish:** How many days before the application did the applicant change the identity document with which he applied for the loan
- **Days of last phone change:** How many days before the application did the applicant change his phone
- **Years of begin expluatation mode:** Normalized information about the building where the applicant lives
- **Total area mode:** Normalized information about the building where the applicant lives
- **External Source 2:** Normalized score from external data source
- **External Source 3:** Normalized score from external data source

Those variables were the result of the Random Forest features importance using 'SelectFromModel' method.

## Conclusions
The F1-score of the Logistic Regression was **65.88%** and the Random Forest Classifier was **66.48%**.
