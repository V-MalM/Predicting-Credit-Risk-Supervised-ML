# Predicting Credit Risk
#### Supervised Machine Learning

## Objective 
* To build a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not.

## Background
LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.
You will be using this data to create machine learning models to classify the risk level of given loans. Specifically, you will be comparing the Logistic Regression model and Random Forest Classifier.

## Retrieve and prepare the data
In the Generator folder in Resources, there is a GenerateData.ipynb notebook that does the following
*   downloads the csv files from LendingClub. 
    
    https://resources.lendingclub.com/LoanStats_2019Q1.csv.zip
    https://resources.lendingclub.com/LoanStats_2019Q2.csv.zip
    https://resources.lendingclub.com/LoanStats_2019Q3.csv.zip
    https://resources.lendingclub.com/LoanStats_2019Q4.csv.zip
    https://resources.lendingclub.com/LoanStats_2020Q1.csv.zip

*   Cleanes the data and creates two CSVs that have been undersampled to give an evennumber of high risk and low risk loans.

    2019loans.csv
    2020Q1loans.csv

    In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques.

    
* Used an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).

