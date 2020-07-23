import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.metrics import classification_report
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#Dataset can be found at
#https://www.kaggle.com/mishra5001/credit-card#application_data.csv

#Github repository can be found at 
#https://github.com/AGryga/Loan_Delinquency_ML

#Redirect file path to application data & previous application data CSVs on your machine.
current_applicants = pd.read_csv('C:/Users/rockn/OneDrive/Desktop/application_data.csv')
previous_applications = pd.read_csv('C:/Users/rockn/OneDrive/Desktop/previous_application.csv')


#------------------------------------------
#---         Data Pre-Processing        ---
#------------------------------------------

#Drops a series of unnecessary columns in current applications dataset based on domain knowledge
current_applicants = current_applicants.drop(labels = ['EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3','APARTMENTS_AVG',
'BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG','COMMONAREA_AVG',
'ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG','LANDAREA_AVG',
'LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG',
'APARTMENTS_MODE','BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE',
'COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE',
'LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE',
'NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI',
'YEARS_BUILD_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI',
'FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI',
'NONLIVINGAPARTMENTS_MEDI','NONLIVINGAREA_MEDI','FONDKAPREMONT_MODE','HOUSETYPE_MODE',
'TOTALAREA_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE','OBS_30_CNT_SOCIAL_CIRCLE',
'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'ORGANIZATION_TYPE'], axis = 'columns')


#Drops a series of unnecessary columns in previous application history dataset based on domain knowledge
previous_applications = previous_applications.drop(['SK_ID_PREV', 'AMT_DOWN_PAYMENT', 
'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT',
'NFLAG_LAST_APPL_IN_DAY', 'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 
'NAME_CASH_LOAN_PURPOSE', 'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 
'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 
'CHANNEL_TYPE', 'SELLERPLACE_AREA', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP',
'PRODUCT_COMBINATION', 'DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 
'DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE', 'DAYS_TERMINATION', 'NFLAG_INSURED_ON_APPROVAL'], axis = 'columns')

#Removes any cancelled loans that were present within the previous loan history dataset.
previous_applications = previous_applications[previous_applications['NAME_CONTRACT_STATUS'] != 'Canceled']

#Replaces unused offer value with approved. It's approved but unused,
previous_applications['NAME_CONTRACT_STATUS'].replace('Unused offer', 'Approved', inplace = True)

#Get full amount that will need to be repaid (Monthly annuity * # of payment cycles)
previous_applications['AMT_REPAYMENT'] = previous_applications['AMT_ANNUITY'] * previous_applications['CNT_PAYMENT'] 


previous_applications_approved = previous_applications[previous_applications['NAME_CONTRACT_STATUS'] == 'Approved']
previous_applications_rejected = previous_applications[previous_applications['NAME_CONTRACT_STATUS'] == 'Refused']


#How many loans were applied for? How many approved vs. cancelled? How much were they approved for, applied for?
agg_approved = previous_applications_approved.groupby('SK_ID_CURR').agg({'NAME_CONTRACT_STATUS' : 'count', 
'AMT_APPLICATION':'mean', 'AMT_CREDIT': 'mean', 'AMT_REPAYMENT' : 'mean'})

agg_approved.columns = ['APPROVED_COUNT', 'MEAN_APPLICATION_AMT_APPROVED', 'MEAN_CREDIT_GIVEN', 'APPROVED_LOAN_ANNUITY']

agg_rejected = previous_applications_rejected.groupby('SK_ID_CURR').agg({'NAME_CONTRACT_STATUS' : 'count', 'AMT_APPLICATION':'mean'})
agg_rejected.columns = ['REJECTED_COUNT', 'AMT_APPLIED_REJECTED']

#Aggregates the stats of the previous history dataset. Aggregated results consisting of how many applications were rejected vs. approved. 
# What was the average amount applied for by each customer, what was the average credit amount they were approved for, etc.?
agg_stats = pd.merge(agg_approved, agg_rejected, on = 'SK_ID_CURR', how = 'outer')
agg_stats.fillna(0, inplace = True)

#Merges the current applicants and summary of previous application history together
full_dataset = pd.merge(current_applicants, agg_stats, left_on = 'SK_ID_CURR', right_index = True, how = 'left')
full_dataset['TOTAL_APPLIED'] = full_dataset['APPROVED_COUNT'] + full_dataset['REJECTED_COUNT']

#Cleans up the total income column
full_dataset['TOTAL_INCOME'] = full_dataset[' AMT_INCOME_TOTAL '].str.replace(',', '').apply(pd.to_numeric, errors = 'coerce')


#Identify the amount of null values within each column. What is the percentage of null records per feature?
percent_missing = full_dataset.isnull().mean() * 100
print(percent_missing)

print('With Car Age missing 66% of its values and Occupation Type missing 31%, we look to remove these values')
full_dataset = full_dataset.drop(['SK_ID_CURR',' AMT_INCOME_TOTAL ', 'OWN_CAR_AGE', 'OCCUPATION_TYPE'], axis = 'columns')


#------------------------------------------
#---    EDA, Encoding & Standardizing   ---
#------------------------------------------

#Look at descriptive statistics of the full_dataset.
full_dataset.describe()


#Overall overview for loan related details. How much credit is being requested, approved. Total amount of prior loans, etc.
numerical_features = full_dataset[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
'MEAN_APPLICATION_AMT_APPROVED', 'MEAN_CREDIT_GIVEN', 'APPROVED_LOAN_ANNUITY', 'AMT_APPLIED_REJECTED']]

#Scales financial values by 1000 to simplify visuals
numerical_features = numerical_features.apply(pd.to_numeric, errors='coerce')#.apply(lambda x : x / 1000)
numerical_features.hist(figsize = [12,12])
plt.suptitle('Distribution of loan related details - Rupees', fontsize = 18)
#plt.savefig('Loan_Details.png', quality = 95)
plt.show()

#Overview of previous loan application.
application_overview = full_dataset[['TOTAL_APPLIED', 'APPROVED_COUNT', 'REJECTED_COUNT']]

application_overview.hist(figsize = [12,12], bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20])
plt.xticks(ticks = [0, 2, 4, 6, 8, 10, 15, 20])
plt.xlabel([0, 2, 4, 6, 8, 10, 15, 20])
plt.suptitle('Credit applications made distribution', fontsize = 18)
#plt.savefig('Prior_loan_history.png', quality = 95)
plt.show()

application_overview.describe()

#Overview of counts of credit bureau inquiries that are occuring for applicants within the last hour, day, week, month, etc.
credit_inquiries = full_dataset[['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 
'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']]
credit_inquiries.hist(figsize = [12,12], bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20])
plt.suptitle('Credit bureau inqury feature distribution', fontsize = 18)
plt.xlabel('Number of Credit Inquiries')
#plt.savefig('Credit_Inquiries.png', quality = 95)
plt.show()

f, ax = plt.subplots(figsize=(12,12))
full_dataset['AMT_CREDIT'].value_counts(bins = [0, 50000, 100000, 200000, 300000, 400000, 500000, 1000000, 2000000, 3000000, 5000000], sort=False).plot(kind='bar', color = '#1f77b4')
plt.title('Current credit offered by loan - Rupees', fontsize = 18)
#plt.savefig('Credit_offered.png', quality = 95)
plt.show()

f, ax = plt.subplots(figsize=(12,12))
full_dataset['AMT_GOODS_PRICE'].value_counts(bins = [0, 50000, 100000, 200000, 300000, 400000, 500000, 1000000, 2000000, 3000000, 5000000], sort = False).plot(kind='bar', color = '#1f77b4')
plt.title('Documented Cost of Goods on current loan - Rupees', fontsize = 18)
#plt.savefig('COGS.png', quality = 95)
plt.show()

print("""Both the distribution of the amount of credit being offered by the loan and 
the distribution of the price of the goods being bought are homogeneous. 
The loans seem to closely fall in line with the goods being purchased""")

#Due to the unbalanced nature of the income field, we look to bin the various incomes reported
#I used the bar chart after binning them with pd.cut. 
#Attempting to create a histogram with these variables took way too many resources + time
f, ax = plt.subplots(figsize=(12, 12))
full_dataset['TOTAL_INCOME'].value_counts(bins = [0, 25000, 50000, 100000, 150000, 200000, 300000, 400000, 500000, full_dataset['TOTAL_INCOME'].max()], sort = False).plot(kind='bar', color = '#1f77b4')
plt.title('Yearly income of applicants - Rupees', fontsize = 18)
#plt.savefig('Income.png', quality = 95)
plt.show()

#Boxplot income vs. target
f, ax = plt.subplots(figsize=(12, 12))
sns.boxplot(x='TARGET', y = 'TOTAL_INCOME', data = full_dataset, color= 'skyblue')
plt.title('Income (Rupees) compared to target variable', fontsize = 18)
#plt.savefig('IncomevTarget.png', quality = 95)
plt.show()
print('Generally, we see those who struggle to pay thier loans are in a lower income bracket with an outlier where a wealthy individual has challenges repaying the loan')

#Boxplot price of goods being bought vs. target. Only looking at loans used for goods
sns.boxplot(x='TARGET', y = 'AMT_GOODS_PRICE', data = (full_dataset[full_dataset['AMT_GOODS_PRICE'] > 0]), color= 'skyblue')
#plt.savefig('GoodsvTarget.png', quality = 95)
plt.title('Cost of goods (Rupees) compared to target variable', fontsize = 18)
plt.show()
print('There does not seem to be a drastic difference in the mean price of goods impacting the likelihood of loan delinquency')

#Boxplot income vs. credit given
f, ax = plt.subplots(figsize=(12, 12))
sns.boxplot(x='TARGET', y = 'AMT_CREDIT', data = full_dataset, color= 'skyblue')
plt.title('Credit (Rupees) provided compared to target variable', fontsize = 18)
#plt.savefig('CreditvTarget.png', quality = 95)
plt.show()
print('Those with difficulties repaying loans seem to have a smaller amount of creidt offered on average.')

#Cycle through various categorical features, providing the value counts for each.
categorical_features = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']

for value in categorical_features:
    print(f'For the {value} feature')
    print(full_dataset[value].value_counts())
    print('\n')

#Scatter plot of credit provided vs. the cost of goods by target variable
sns.set_style('whitegrid')
sns.FacetGrid(full_dataset, hue = 'TARGET').map(plt.scatter,'AMT_CREDIT','AMT_GOODS_PRICE').add_legend()
#plt.title('Comparison of credit versus cost of goods', fontsize = 18)
#plt.savefig('CreditvGood.png', quality = 95)
plt.show()
print('There seems to be a steady correlation between the credit they receive vs. the price of the good they are looking to buy')

#Scatter plot of historically provided avg. credit for an approved loan vs. the current loan's credit
sns.FacetGrid(full_dataset[full_dataset['MEAN_CREDIT_GIVEN'] > 0], hue = 'TARGET').map(plt.scatter, 'AMT_CREDIT', 'MEAN_CREDIT_GIVEN').add_legend()
#plt.title('Comparison of avg. historical credit offered vs. current credit extended', fontsize = 18)
#plt.savefig('CreditvHistoricalCredit.png', quality = 95)
plt.show()
print('Of those who will have repayment difficulties and have a previous approved loan history, it seems many received more credit on the current loan than they typically have before, potentially giving them a loan too large to financially manage well')

#Remove the 3 instances where gender is unknown.
m_f_df = full_dataset[full_dataset['CODE_GENDER'] != 'XNA']

#Stacked bar chart of family type compared to living situation
family = pd.crosstab(columns = m_f_df['NAME_FAMILY_STATUS'], index = m_f_df['NAME_HOUSING_TYPE'])
family.plot(kind='bar', figsize = [13,13], stacked=True)
plt.xticks(fontsize = 14)
#plt.savefig('Family_v_Housing', quality = 95)
plt.show()

#Stacked bar chart of family type compared to income type
val_counts = pd.crosstab(columns = full_dataset['NAME_FAMILY_STATUS'], index =full_dataset['NAME_INCOME_TYPE'])
val_counts.plot(kind='bar', figsize = [13,13], stacked=True)
plt.xticks(fontsize = 14)
#plt.savefig('Family_v_Employment', quality = 95)
plt.show()

#Cleans up the asset ownership columns, converting Y & N to boolean values
full_dataset[['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']] = full_dataset[['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']].replace({'Y': 1, 'N': 0})

#List comprehension to get remaining categorical features. Remove these booleans from the list as they've been remapped
categorical_features = [i for i in categorical_features if i != 'FLAG_OWN_CAR' if i != 'FLAG_OWN_REALTY']

#Create label encoding & encodes the remaining categorical features
labelencoder = LabelEncoder()

for value in categorical_features:
    full_dataset[value] = labelencoder.fit_transform(full_dataset[value].astype(str))

#Creates a correlation matrix for non-categorical variables
corr = full_dataset[~full_dataset.isin(categorical_features)].corr()
fig, ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(corr, cmap = 'coolwarm', center=0, square=True, linewidths=.5)
#plt.savefig('Heatmap.png', quality = 95)
plt.show()

#Normalize the data before completing principal component analysis to identify how many key features there are.
#Normalize all features (Excluding Target being 1st column)
full_dataset_na_filled = full_dataset.apply(lambda x: x.fillna(x.median()), axis=0)
full_dataset_na_filled = full_dataset_na_filled.drop('TARGET', axis = 'columns')

normalized_values_chart = StandardScaler().fit_transform(full_dataset_na_filled.values)

pca = PCA().fit(normalized_values_chart.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative Variance that is explained')
#plt.savefig('PCA.png', quality = 95)
plt.show()
print("""We see that 10 features represent around 40% of the variance of the model. 
The incremental 10 components add about 15% more cumulative variance""")

#Uncomment to export cleaned dataset
#full_dataset.to_csv('Full_Dataset.csv')

#Run function below to see cross-tabs used in literature review submission
#cross_tabs(full_dataset)


#------------------------------------------
#---          Machine Learning          ---
#------------------------------------------
#Gathers datasets to be fed into the train_test_split function after resampling.
#X contains all non-target features. Y contains the target features.
X = pd.DataFrame(StandardScaler().fit_transform(full_dataset_na_filled), columns = full_dataset_na_filled.columns)
y = full_dataset['TARGET']

#Look to resample the minority class (1's) to be even with the majority class (0's). 
#With the dataset being unbalanced(282,686 0's and 24,825 1's), the data needs to be resampled, oversampling the minority class
y.value_counts()

#Recreate full dataframe for just the minority class (target feature + all subsequent features)
#This minority class dataframe will then be resampled to create a 50/50 proportion for the training & test data
y_minority = pd.DataFrame(y[y == 1])
minority_dataset = y_minority.merge(X, left_index = True, right_index = True)

#Resamples the minority dataset, bringing the proportion of majority : minority classes to equilibrium
resampled_minority_dataset = resample(minority_dataset, replace = True, n_samples = 282686, random_state = 123)

#Recreate the unaltered majority class dataframe.
y_majority = pd.DataFrame(y[y == 0])
majority_dataset = y_majority.merge(X, left_index = True, right_index = True)

#Concatenates the resampled minority class & unaltered majority class
#This new dataframe contains a 50/50 proportion of the target feature classes
full_dataset_resampled = pd.concat([majority_dataset, resampled_minority_dataset])

#Following the oversampling of the minority class, the data is returned back 
# into the X and y dataframes for training and testing splits
X = full_dataset_resampled.iloc[:, 1:]
y = full_dataset_resampled.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 123)
kfold = KFold(n_splits = 5, random_state = 7)

#    Logistic Regression   
print('\n Logistic regression model - Features Selected')

#Use SelectFromModel to identify key features utilized by the Logistical Regression model
#Class weight is balanced as the dataset contains a 50/50 proportion of classes
log_reg = SelectFromModel(LogisticRegression(class_weight = "balanced", random_state = 123))
log_reg.fit(X_train, y_train)

#Identify the features selected as drivers within the model. Reshape the X_train dataset to contain those 6 key features
log_selected_features = X_train.columns[(log_reg.get_support())].tolist()
X_train_log_selected = X_train[X_train.columns.intersection(log_selected_features)]

print(log_selected_features)

#Plot feature importance
pd.Series(pd.to_numeric(log_reg.estimator_.coef_.tolist()[0]), index=X_train.columns).plot.bar(color='skyblue', figsize=(12, 12))
plt.title('Logistic Regression Feature Importance')
plt.ylabel('Feature Importance', fontsize = 12)
#plt.savefig('LogReg_Importance.png', quality=95, bbox_inches = 'tight')
plt.show()

#Utilize GridSearchCV to finetune and identify optimal parameters. 
#Used to identify the best options for the regularization type and learning rate originally.
#param_grid = {"penalty": ["l1", "l2"], "C": [0.001, 0.005, 0.05, 0.1]}

log_param_grid = {"penalty": ["l1"], "C": [0.1]}
logistic_reg = GridSearchCV(LogisticRegression(class_weight = "balanced", random_state = 123), 
                                                log_param_grid, n_jobs = -1, refit = True, scoring="roc_auc")

logistic_reg.fit(X_train_log_selected, y_train)

print(f'Best score: {logistic_reg.best_score_} with param: {logistic_reg.best_params_}')

X_test_log_selected = X_test[X_test.columns.intersection(log_selected_features)]
y_log_predictions = logistic_reg.predict(X_test_log_selected)

conf_matrix = metrics.confusion_matrix(y_test, y_log_predictions)
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, fmt = 'g', cmap = 'coolwarm_r')
plt.title('Logisitic Regression')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#plt.savefig('LogReg_CM.png', quality = 95)
plt.show()

print(f"Accuracy: {metrics.accuracy_score(y_test, y_log_predictions)}")
print(classification_report(y_test, y_log_predictions))

y_pred_prob = logistic_reg.predict_proba(X_test_log_selected)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_prob)
auc = metrics.roc_auc_score(y_test, y_pred_prob)
plt.plot(fpr,tpr)
plt.title(f'Logistic Regression - Area Under Curve : {str(auc)[:4]}')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig('LogReg_AUC.png',quality = 95)
plt.show()

print(f'Area Under Curve : {auc}')

#    Random Forests
print('\n Random Forest model')
# rf_param_grid = {'n_estimators': [100, 200, 300, 400], 'max_features': ['auto', 'sqrt', 'log2'], 
#                 'max_depth' : [5, 10, 15, 20],'criterion' :['gini', 'entropy']}

rf_param_grid = {'n_estimators': [400], 'max_features': ['auto'], 
                'max_depth' : [20], 'criterion' :['gini']}

random_forest =  SelectFromModel(RandomForestClassifier(class_weight = "balanced", random_state = 123))
random_forest = random_forest.fit(X_train, y_train)


rf_selected_features = X_train.columns[(random_forest.get_support())].tolist()
X_train_rf_selected = X_train[X_train.columns.intersection(rf_selected_features)]
print(rf_selected_features)

#Plot feature importance
pd.Series(pd.to_numeric(random_forest.estimator_.feature_importances_), index=X_train.columns).plot.bar(color='skyblue', figsize=(12, 12))
plt.title('Random Forest Feature Importance  - Features Selected')
plt.ylabel('Feature Importance', fontsize = 12)
#plt.savefig('RF_Feat_Importance.png', quality = 95, bbox_inches = 'tight')
plt.show()

random_forest = GridSearchCV(RandomForestClassifier(class_weight = "balanced", random_state = 123), 
                                                rf_param_grid, cv = kfold, n_jobs = -1, refit = True, scoring = "roc_auc")

random_forest.fit(X_train_rf_selected, y_train)

print(f'Best score: {random_forest.best_score_} with param: {random_forest.best_params_}')

X_test_rf_selected = X_test[X_test.columns.intersection(rf_selected_features)]
y_rf_predictions = random_forest.predict(X_test_rf_selected)

conf_matrix = metrics.confusion_matrix(y_test, y_rf_predictions)
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, fmt = 'g', cmap = 'coolwarm_r')
plt.title('Random Forests')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#plt.savefig('RF_CM.png', quality=95)
plt.show()

print(f"Accuracy: {metrics.accuracy_score(y_test, y_rf_predictions)}")
print(classification_report(y_test, y_rf_predictions))

y_pred_prob_rf = random_forest.predict_proba(X_test_rf_selected)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_prob_rf)
auc = metrics.roc_auc_score(y_test, y_pred_prob_rf)
plt.plot(fpr,tpr)
plt.title(f'Random Forests - Area Under Curve : {str(auc)[:4]}')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig('RF_AUC.png', quality = 95)
plt.show()

print(f'Area Under Curve : {auc}')


#    Boosted Trees
print('\n Gradient Boosted Trees model')
data_dmatrix = xgb.DMatrix(data = X_train, label = y_train)

# gbt_param_grid = {'n_estimators' : [100, 200, 300, 400], 'max_depth' : [5, 10, 15, 20], learning_rate' : [0.01, 0.05, 0.1, 1]}

gbt_param_grid = {'n_estimators' : [400], 'max_depth' : [20], 'learning_rate' : [0.1]}
                    
xgb_class = xgb.XGBClassifier(seed = 123)

#Use the function below to retrieve a chart of feature importance for boosted tree models.
#Needed to be in a function as it can't work alongside the GridSearchCV function
def boosted_trees_ft_importance(xgb_class):
    xgb_class.fit(X_train, y_train)

    pd.Series(pd.to_numeric(xgb_class.feature_importances_), index=X_train.columns).plot.bar(color='skyblue', figsize=(12, 12))
    plt.title('Boosted Trees Feature Importance  - Features Selected')
    plt.ylabel('Feature Importance', fontsize = 12)
    #plt.savefig('GBT_Feat_Importance.png', quality = 95, bbox_inches = 'tight')
    plt.show()

#boosted_trees_ft_importance(xgb_class)

gbt = GridSearchCV(xgb_class, gbt_param_grid, cv=kfold, n_jobs = 2, scoring='roc_auc')

gbt.fit(X_train, y_train)

print(f'Best score: {gbt.best_score_} with param: {gbt.best_params_}')

# Perform K-Fold Cross Validation
#results = cross_val_score(gbt, X_train, y_train, cv=kfold, n_jobs = -1)

y_pred_gbt = gbt.predict(X_test)

conf_matrix = metrics.confusion_matrix(y_test, y_pred_gbt)
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, fmt = 'g', cmap = 'coolwarm_r')
plt.title('Gradient Boosted Trees')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('XGB_CM.png',quality = 95)
plt.show()

print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred_gbt)}")
print(classification_report(y_test, y_pred_gbt))

y_pred_prob_gbt = gbt.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_prob_gbt)
auc = metrics.roc_auc_score(y_test, y_pred_prob_gbt)
plt.plot(fpr, tpr)
plt.title(f'Boosted Trees - Area Under Curve : {str(auc)[:4]}')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('XGB_AUC.png', quality = 95)
plt.show()

print(f'Area Under Curve : {auc}')


plot_tree(gbt)

def cross_tabs(df):
    #Uses list comphrension drop all FLAG_DOCUMENT_INT columns for descriptive statistics
    focused_ca_data = df.drop([col for col in df.columns if 'FLAG_DOCUMENT' in col], axis = 'columns')

    quant_DS_df = pd.DataFrame(focused_ca_data.describe())
    #quant_DS_df.to_csv('Quant_Desc_Stat.csv', index = True, header = True)

    qual_DS_df = focused_ca_data.describe(include=['object'])
    #qual_DS_df.to_csv('Qual_Desc_Stat.csv', index = True, header = True)

    #Cross tab of target variable
    target_var = pd.crosstab(index = df['TARGET'], columns = 'count')
    target_var
    
    #Cross tab of loan types by target
    loan_type = pd.crosstab(index = df['TARGET'], columns = df['NAME_CONTRACT_TYPE'])
    loan_type

    #Cross tab of car & real estate assets by target
    assets = pd.crosstab(index = df['TARGET'], columns = [df['FLAG_OWN_CAR'], df['FLAG_OWN_REALTY']])
    assets

    #Cross tab of gender by target
    gender = pd.crosstab(index = df['CODE_GENDER'], columns = 'Count')
    gender

def logreg_no_ft_selection():
    print('\n Logistic regression model - No Features Selected')

    log_reg = LogisticRegression(class_weight = "balanced", penalty = 'l1', random_state = 123, n_jobs = -1)
    log_reg.fit(X_train, y_train)

    y_log_predictions = log_reg.predict(X_test)

    conf_matrix = metrics.confusion_matrix(y_test, y_log_predictions)
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, fmt = 'g', cmap = 'coolwarm_r')
    plt.title('Logisitic Regression - No Feature Selection')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    #plt.savefig('LogReg_CM_No_Slct.png', quality = 95)
    plt.show()

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_log_predictions)}")
    print(classification_report(y_test, y_log_predictions))

    y_pred_prob = log_reg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_prob)
    auc = metrics.roc_auc_score(y_test, y_pred_prob)
    plt.plot(fpr,tpr)
    plt.title(f'Logistic Regression No Feature Selection \n Area Under Curve : {str(auc)[:4]}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.savefig('LogReg_AUC_No_Slct.png',quality = 95)
    plt.show()

    print(f'Area Under Curve : {auc}')

def random_forest_no_ft_selection():
    print('\n Random Forest model - No Features Selected')

    random_forest = RandomForestClassifier(class_weight = "balanced", criterion='gini', 
                                        n_estimators= 400, max_features= 'auto', max_depth=20,
                                        random_state = 123)

    random_forest.fit(X_train, y_train)

    y_rf_predictions = random_forest.predict(X_test)

    conf_matrix = metrics.confusion_matrix(y_test, y_rf_predictions)
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, fmt = 'g', cmap = 'coolwarm_r')
    plt.title('Random Forests - No Feature Selection')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig('RF_CM_No_FT_Select.png', quality=95)
    plt.show()

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_rf_predictions)}")
    print(classification_report(y_test, y_rf_predictions))

    y_pred_prob_rf = random_forest.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_prob_rf)
    auc = metrics.roc_auc_score(y_test, y_pred_prob_rf)
    plt.plot(fpr,tpr)
    plt.title(f'Random Forests No Feature Selection \n Area Under Curve : {str(auc)[:4]}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('RF_AUC_NoFT_Select.png', quality = 95)
    plt.show()

    print(f'Area Under Curve : {auc}')


#logreg_no_ft_selection()
#random_forest_no_ft_selection()



