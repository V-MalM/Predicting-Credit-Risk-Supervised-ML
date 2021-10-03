# Predicting Credit Risk
**Supervised Machine Learning**

## Objective 
* To build a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not.

## Background
LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.
Task was to use this data to create machine learning models to classify the risk level of given loans. Specifically, would be comparing the Logistic Regression model and Random Forest Classifier.

## Retrieve and prepare the data
In the Generator folder, in Resources, there is a [GenerateData.ipynb](Resources/Generator/GenerateData.ipynb) notebook that does the following
*   downloads the csv files from LendingClub.  
    https://resources.lendingclub.com/LoanStats_2019Q1.csv.zip
    https://resources.lendingclub.com/LoanStats_2019Q2.csv.zip
    https://resources.lendingclub.com/LoanStats_2019Q3.csv.zip
    https://resources.lendingclub.com/LoanStats_2019Q4.csv.zip
    https://resources.lendingclub.com/LoanStats_2020Q1.csv.zip

*   Cleans the data and creates two CSVs that have been undersampled to give an evennumber of high risk and low risk loans.
   
    * 2019loans.csv  
    * 2020Q1loans.csv

    In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques.

    ```python
    low_risk_rows.sample(n=len(high_risk_rows), random_state=42)
    ```
    
    
* Used an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).

## Logistic regression Vs. Random Forests Classifier

Both the models are popular in machine learning. They are both efficient in generating reliable models for predictive modelling.

* Logistic regression is less complex, and less prone to over-fitting. 
* It does not require any parameters to tune.
* It performs best with scaled data.  
<br />

* Random Forest uses Decision trees that can be scaled to be complex, and hence more liable to over-fit. Pruning is applied to avoid this.
* Although default parameters may work fine, Random Forests work best when they are tuned by applying parametes.
* Random Forsests perfoms well with unscaled data. 

When creating a predictive model, both the techniques should be tried and the best performing model should be used.

## A prediction as to which model will perform better before I created, fit, and scored the models. 
I predict Logistic regression will perform better because it works best for binany classification problems. The data in question has binary out put belonging to one class or the other (High Risk, Low Risk)

**LET's FIND OUT**

Steps :
1. Convert categorical data to numeric and separate target feature for training data and testing data
2. Encode target values using class sklearn.preprocessing.LabelEncoder
3. Add missing dummy variables to testing set
4. Train the Logistic Regression model on the unscaled data and print the model score
    * Adjust hyperparameters to see if the score improves .
        * Adjusting the Hyperparameters on LR model on unscaled data 
            * Tried a few combinations to tune
            * It did improve the Testing Data Score by 5%
            * Takes longer execution time each time a parameter/parameters is/are changed
6. Train a Random Forest Classifier model on unscaled data and print the model score  
    * Adjusting hyperparameters to see if the score improves .  
    To choose which hyperparameters to adjust, we could visualize with validation_curve, or conduct Exhaustive Grid Search. 
     * Used validation_curve and test the parameters 'n_estimators', 'max_depth', 'min_samples_split' by giving them a range of values.       
     * Adjusting the Hyperparameters on RF Classifiers  on unscaled data 
        * Tried a few combinations to tune
        * It did NOT improve the Testing Data Score 
        * Takes longer execution time each time a parameter/parameters is/are changed
## Results
Unlike my prediction, The Random Forest Classifier performed far better then the logistic regression model.   
2020 First Quarter score was : **0.4913317572892041** for Logistic regression model.  
Random forest 2020 First Quarter score was : **0.67625426845285**
## Revisit the Preprocessing: Scale the data
 * I predict that scaling data will considerably improve logistic regression model and it will outperform Random Forest Model. 
 StandardScaler makes field values compareable by removing the mean and by scaling each feature/variable to unit variance.
 * We know that Random Forst Model is built on decision trees and ensemble methods that do not require feature scaling as they are not sensitive to the the variance in the data. Hence, we might not see significant improvement in Random Forest model.
## Scaled Training and Testging sets using StandardScaler().fit_transform()
## Train the Logistic Regression model on the scaled data and print the model score
* Logistic regression model improved considerably after scaling the data.
## Train a Random Forest Classifier model on the scaled data and print the model score
* Random Forest has no positive effect with scaled data. It did Not improve.
* Logistice Regression outperformed random Forest Classifier.
# Conclusions
* After scaling the data, Logistic Regression outperformed Random Forest Classifier.
## LogisticRegression Model Performed well on this Data and we can conclude that it is the right fit.
