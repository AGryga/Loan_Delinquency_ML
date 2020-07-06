import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

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

fig, axes = plt.subplots()
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

#principalComponents = pca.fit_transform(normalized_full_dataset)
#print(pca.explained_variance_ratio_)


#Uncomment to export cleaned dataset
#full_dataset.to_csv('Full_Dataset.csv')

#Run function below to see cross-tabs used in literature review submission
#cross_tabs(full_dataset)


#------------------------------------------
#---          Machine Learning          ---
#------------------------------------------
#Gathers datasets to be fed into the train_test_split function. X contains all non-target features. Y contains the target features.
X = pd.DataFrame(StandardScaler().fit_transform(full_dataset_na_filled), columns = full_dataset_na_filled.columns)
y = full_dataset['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 123)

#Look to resample the minority class (1's) to be even with the majority class (0's). 
#Need to resample to around 226,000 records for the minority class. Currently at 19,860 based on splitting the training data
y_train.value_counts()

#Should I combine the X & y to be one full dataset, resample and then resplit into X & y?

#Recreate full dataframe for just minority class (target feature + all subsequent features)
y_train_minority = pd.DataFrame(y_train[y_train == 1])
full_training_minority_dataset = y_train_minority.merge(X_train, left_index = True, right_index = True)

#Increase training data sample set for the minority class through resampling to oversample the minority class in the model training.
full_train_minority_resampled = resample(full_training_minority_dataset, replace = True, n_samples = 226000, random_state = 123)
#full_train_minority_resampled.to_csv('Resampled_Minority_Class.csv')

#Recreate the majority class, bringing together the resampled minority class & unaltered majority class. This new dataframe contains a 50/50 proportion of the target feature classes
y_train_majority = pd.DataFrame(y_train[y_train == 0])
full_train_majority = y_train_majority.merge(X_train, left_index = True, right_index = True)

#Concatenates the two target class dataframes together.
full_training_data_resampled = pd.concat([full_train_majority, full_train_minority_resampled])

#Following the oversampling of the minority class, the data is returned back into the X_train and y_train format to develop and train the models
X_train = full_training_data_resampled.iloc[:, 1:]
y_train = full_training_data_resampled.iloc[:, 0]

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




