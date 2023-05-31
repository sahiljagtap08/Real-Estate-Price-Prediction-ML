#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real Estate Price Predictor
#

# In[1]:


from joblib import dump, load
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")
housing.head()


# In[3]:


housing["CHAS"].value_counts()


# In[4]:


housing.describe()


# In[5]:


# %matplotlib inline
# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20,15))
# plt.show()


# ## Train test splitting
#
#

# In[6]:


# manual training and testing


def split_train_test(data, test_ratio):
    # this ensures that random values are fixed and not change every time
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled[:20]
    train_indices = shuffled[20:]
    return data.iloc[train_indices], data.iloc[test_indices]

# train_set, test_set = split_train_test(housing, 0.2)


# In[7]:


# training and testing by help of inbuilt fn
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# here there is a problem, for eg CHAS hos only two values 0 and 1, 0-475 and 1-35
# but in spliting train and test data suppose traindata = 402, testdata=104
# if there are no 1's in train data then our program forms wrong pattern that
# there is only one possibility of CHAS i.e, 1


# In[8]:


# to solve above problem
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[9]:


strat_test_set['CHAS'].value_counts()


# In[10]:


strat_train_set['CHAS'].value_counts()


# In[11]:


# 95/7 ~= 376/28


# In[12]:


# this is to be done for large data and not include test data in it
housing = strat_train_set.copy()


# ## Looking for correlations

# In[13]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[14]:


# if value is 1 meaning strong positive correlation
# if value is -1 meaning strong negative correlation
# next Rm value is 0.69 which is high positive correlation which means if RM increases chances
# of increasing MEDV increases, then ZN and B are weak positive correlation
# similarly lstat is high neg corr, lesser value of lstat higher value of medv


# In[15]:


# from pandas.plotting import scatter_matrix
# attributes = ["MEDV", "RM","ZN","LSTAT"]
# scatter_matrix(housing[attributes], figsize=(12,8))


# In[16]:


# housing.plot(kind="scatter", x="RM",y="MEDV",alpha=0.8)


# ## Trying out Attribute combinations

# In[17]:


housing["TAXRM"] = housing["TAX"]/housing["RM"]  # you can try any combination
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
# finally we are not adding this in our main data for now........


# In[18]:


# housing.plot(kind="scatter", x="TAXRM",y="MEDV",alpha=0.8)


# In[19]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Missing attributes

# In[20]:


# To take care of missing attributes, you have three options:
#     1. Get rid of the missing data points
#     2. Get rid of the whole attribute
#     3. Set the value to some value(0, mean or median)


# In[21]:


a = housing.dropna(subset=["RM"])  # option 1
a.shape


# In[22]:


housing.drop("RM", axis=1).shape  # option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged


# In[23]:


median = housing["RM"].median()  # option 3
median


# In[24]:


housing["RM"].fillna(median)
# Note that the original housing dataframe will remain unchanged


# In[25]:


housing.describe()    # before we started filling missing attributes


# In[26]:


# for reflecting the changes in original train and test data i.e, null alues to median
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[27]:


imputer.statistics_    # calculates for every row and automatically fills


# In[28]:


X = imputer.transform(housing)


# In[29]:


housing_tr = pd.DataFrame(X, columns=housing.columns)


# In[30]:


housing_tr.describe()


# ## Scikit learn Design

# Primarily, three types of objects
#
# 1) Estimators - It estimates some parameter based on a dataset.
# Eg. imputer. It has a fit method and transform method.
# Fit method - Fits the dataset and calculates internal parameters
#
# 2) Transformers - transform method takes input and returns output based on the learnings from fit().
# It also has a convenience function called fit_transform() which fits and then transforms.
#
# 3) Predictors - LinearRegression model is an example of predictor.
# fit() and predict() are two common functions. It also gives score() function which will
# evaluate the predictions.

# ## Feature Scaling
#

# Primarily, two types of feature scaling methods:
#
# 1) Min-max scaling (Normalization) (value - min)/(max - min)
# Sklearn provides a class called MinMaxScaler for this
#
# 2) Standardization (value - mean)/std
# Sklearn provides a class called StandardScaler for this

# ## Creating a pipeline

# In[31]:


# instead of doing imputer, directly you can opt for pipeline which automates things
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    #     add as many as you want...
    ('std_scaler', StandardScaler())
])


# In[32]:


housing_num_tr = my_pipeline.fit_transform(housing)
housing_num_tr    # this is a numpy array


# ## Selecting a desired model for the problem

# In[33]:


# model = LinearRegression()
# model = DecisionTreeRegressor()    # this model is bad becouse mse=0, which means it is overfitting
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)
housing_num_tr.shape


# In[34]:


some_data = housing.iloc[:5]


# In[35]:


some_labels = housing_labels.iloc[:5]


# In[36]:


prepared_data = my_pipeline.transform(some_data)


# In[37]:


model.predict(prepared_data)


# In[38]:


np.array(some_labels)


# ## Evaluating the model

# In[39]:


housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[40]:


rmse


# ## Using better evaluation techniques - Cross Validation

# In[41]:


# for eg 1 2 3 4 5 6 7 8 9, firstly it will train expect 1 and check for 1, then repeat it for 2, 3..
# so on.. to last value
scores = cross_val_score(
    model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[42]:


rmse_scores


# In[43]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[44]:


print_scores(rmse_scores)


# ## Saving the model

# In[45]:


dump(model, 'Dragon.joblib')


# ## Testing the model on test data

# In[49]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))


# In[47]:


final_rmse


# In[48]:


prepared_data[0]


# ## Using the model

# In[50]:


model = load('Dragon.joblib')
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
                      -11.44443979304, -49.31238772,  7.61111401, -26.0016879, -0.5778192,
                      -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)


# In[ ]:
