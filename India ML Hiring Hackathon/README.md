### India ML Hiring Hackathon 2019

- Public Leaderboard score: 0.33175355450237
- Private Leaderboard score: 0.565217391304348

### Problem URL

https://datahack.analyticsvidhya.com/contest/india-ml-hiring-hackathon-2019/

### Problem Statement

Predict the loan delinquency of a customer on his 13th month given the customer transaction and credit profile data.

### Approach

I have used basic feature engineering to help model understand the customer behavior better and a simple approach by using 10 fold cross validated CatBoost model and used the results for my submission. The only thing I have taken care is to use the class_weights parameter as the classes in the dataset are highly unbalanced. There is no need to preprocess the data as the data looks clean without any missing values. I trusted the CV score because the public LB score is not representative of the local CV score and the private LB score is close to the local CV score.

***
> 1. ### Understanding the Dataset
>> - The loan delinquency dataset was provided in a csv file. The dataset consisted of 116,058 rows with 29 columns. The columns included a combination of date, boolean, int, float and string datatypes.
>> - As the dataset has no missing values there is no need to use any data imputation techniques.
>> - Columns with date type such as origination_date & first_payment_date were of string type and were converted to datetime format.
>> - The transactions information included customer transaction attributes such as interest_rate, unpaid_principal_bal, loan_term, debt_to_income_ratio, etc.
> 1. ### Model Building
>> #### Feature Engineering
>> - My approach is pretty straightforward which mainly revolves around feature engineering. I tried many different combination of features and found the below three feature sets to be most useful.
>> - I haven't filtered any of the records from the dataset as the data looks clean without any missing values.
>>> 1. `Category level features`: I have built categorical based aggregated features such as min, median, mean, max of borrower credit scores, interest rate, loan to value, unpaid principal balance, etc using groups in source, financial institution, etc.
>>> 1. `Temporal features`: Temporal features almost always helps boosted trees as most of the time these models can leverage the cross-sectional correlation of the data (e.g. interaction between different features at a given observation level) but there is no way for the model to tackle the time dimension (e.g. latent interaction between two or more observations recorded at different time points). by infusing these featues explicitly - the model can also learn the cross-time correlation e.g. how transactions of customer in the past affects the nature of transaction of a customer at present. This is very important.
>>>> - The temporal features that I considered are:
>>>> - Account Age - days between origination_date and first_payment_date
>>>> - No of months in loan term
>>>> - Daily loan value
>>>> - Montly loan value
>>> 1. `Ratio features`: I created a number of ratios between:
>>>> - Ratio of unpaid_principal_bal to loan_to_value
>>>> - Ratio of unpaid_principal_bal to debt_to_income_ratio
>>>> - Ratio of interest_amt to number_of_borrowers
>>>> - Ratio of credit_scores to number_of_borrowers
>>> 1. `Date features`: A lot of date based features such as origination_Year, origination_Month, origination_Day, etc.
>>> 1. `Delinquency level features`: I have built delinquency based aggregated features such as mean, total of columns m1...m12 by splitting them into time buckets. For example. Mean of m1, m2, m3 as First Quarter delinquency mean, Mean of m1...m6 as Half yearly delinquency average, etc.
>> #### Model Training & Testing: 
>>>> - The rich set of 328 features were built and the data was split into train and test sets. 
>>>> - As the data is highly unbalanced I set the class weights parameter in catboost as a list with [0.15, 0.85].
>>>> - I have played with different ML algorithms such as random forest, CatBoost, LightGBM, XGBoost,etc. but the solution will have only Catboost implementation as it gave the best results to predict loan delinquency on m13 with an Average crossvalidated f1 score of 0.55 on the validation sets.
> 1. ### Key Takeaways:
>> - Delinquency based features are the most important features determining the m13 dependent variable.
>> - I have learnt key concept on dealing with the imbalanced data in a better way
> 1. ### Focus Areas:
>> - Feature engineering is the key in these type of problems.
>> - Model building should be done keeping in mind the class imbalance.
>> - Always trust your local CV score and take of overfitting.
>> - Data is very clean in this problem but it is not always the case in the real world scenarios so be prepared with data imputation techniques.
>> - Try using ensembles to improve the score and model performance.
