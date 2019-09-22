#Perform an exploratory data analysis, implement a Linear, Ridge, and Lasso regression model. (3 models total)
'''
The idea is to create a baseline model using all of the
variables in a simple linear regression first, and then
to apply the regularization techniques to constrain or
eliminate some of the variables using Lasso and Ridge.
There is a difference between visualization and model fitting, however.
For your EDA you should produce a table of summary statistics
for each of the 13 explanatory variables.  I would then produce
a 13x13 correlation matrix, which could be displayed as a heatmap.
 You can try adjusting the font smaller, or the size larger or
 set ANNOT=False if it doesn't look good.
'''

import numpy as np
from sklearn.datasets.samples_generator import  make_regression
import pandas as pd
df = pd.read_csv("F:/MSFE/machine_learning/HW4/housing2.csv")
df_1= pd.read_csv("F:/MSFE/machine_learning/HW4/housing.csv")

print('Number of rows of data:', df.shape[0])
print('Number of columns of data:', df.shape[1])
print(df.info())
df = df.dropna()
y = df['MEDV']
X = df.drop(['MEDV'],axis=1)
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
print(df.info())
import matplotlib.pyplot as plt
import seaborn as sns

#EDA
percentiles = np.array([2.5,25,50,75,97.5])
ptiles_X = np.percentile(X,percentiles)
ptiles_y = np.percentile(y,percentiles)


print(ptiles_X )
summary_X = df[cols].describe()
print("The summary of X:", summary_X)

print(ptiles_y )
summary_y = df[cols].describe()
print("The summary of y:",summary_y)

sns.set()


X_ = plt.hist(df_1.drop(['MEDV'],axis=1),bins=50,density=True, facecolor = 'blue', alpha=0.5)
plt.xlabel('feature_value')
plt.ylabel('feature_number')
plt.title('Histogram of features')
plt.show()
plt.clf()

y_n = len(y)
bins_2 = np.sqrt(y_n)
y_bins = int(bins_2)
y_ = plt.hist(df_1['MEDV'],bins=y_bins,density=True, facecolor = 'green', alpha=0.5)
plt.xlabel('target_value')
plt.ylabel('target_number')
plt.title('Histogram of target')
plt.show()
plt.clf()
# summary plot
sns.pairplot(df[cols],size=1.5)
plt.tight_layout()
plt.show()
#heatmap
cm=np.corrcoef(df[cols].values.T)
sns.set(font_scale=0.8)
sns.set_style("dark")
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':8},yticklabels=df_1.columns,xticklabels=df_1.columns)
plt.show()
#box plot
cols = ['CRIM','INDUS']
fig,ax = plt.subplots(len(cols),figsize = (8,40))
for i, col_val in enumerate(cols):
    sns.boxplot(y=X[col_val],ax=ax[i])
    ax[i].set_title('Box Plot - {}'.format(col_val),fontsize=10)
    ax[i].set_xlabel(col_val,fontsize=8)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


df_2 = pd.read_csv("F:/MSFE/machine_learning/HW4/housing2.csv")

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:,np.newaxis]).flatten()
print('X_std:',X_std)
print('y_std:',y_std)

X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size= 0.2, random_state=42)
from sklearn.metrics import mean_squared_error
print('Shape of X_train: ', X_train.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of X_test: ', X_test.shape)
print('Shape of y_test: ', y_test.shape)
#LinearRegression
from sklearn.linear_model import LinearRegression
# Create the regressor: lr_reg
lr_reg = LinearRegression()
lr_reg.fit(X_train,y_train)
lr_y_train_pred = lr_reg.predict(X_train)
lr_y_test_pred = lr_reg.predict(X_test)
# Compute and print R^2 and RMSE
print("LinearRegression -> coefficient: {}".format(lr_reg.coef_))
print("LinearRegression -> y intercept: {}".format(lr_reg.intercept_))
print("R^2: {}".format(lr_reg.score(X_test,y_test)))
print('Linear Regression MSE Train: %.3f, Test: %.3f' % ((mean_squared_error(y_train,lr_y_train_pred)),
      mean_squared_error(y_test,lr_y_test_pred)))
print("--------------------------------------------------------------")
#Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
# Setup the array of alphas and lists to store scores
alpha_space = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2,0.4,0.5,0.6,0.7,0.8,0.9,
               1]
# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

min_test_MSE = 100
find_min_MSE = []
for alpha in alpha_space:
    ridge.alpha = alpha
    ridge.fit(X_train, y_train)
    ridge_y_train_pred = ridge.predict(X_train)
    ridge_y_test_pred = ridge.predict(X_test)
    find_min_MSE.append(mean_squared_error(y_test,ridge_y_test_pred))
    if mean_squared_error(y_test,ridge_y_test_pred) < min_test_MSE:
        min_alpha = alpha
        min_test_MSE = mean_squared_error(y_test,ridge_y_test_pred)
        train_MSE = mean_squared_error(y_train,ridge_y_train_pred)
        coef = ridge.coef_
        intercept = ridge.intercept_
        train_R2 = ridge.score(X_train,y_train)
        test_R2 = ridge.score(X_test,y_test)


plt.plot(alpha_space,find_min_MSE)
plt.xlabel('alpha')
plt.ylabel('Ridge_test_MSE')
plt.show()
print("Ridge Regression-> Lowest MSE of TEST: {} and its alpha is {} ".format(min_test_MSE,min_alpha))
print("Ridge Regression-> Lowest MSE of Train: {}".format(train_MSE))
print("Ridge Regression-> coefficient: {}".format(ridge.coef_))
print("Ridge Regression -> intercept: {}".format(ridge.intercept_))
print("Train R^2: {}".format(train_R2))
print("Test R^2: {}".format(test_R2))
print("--------------------------------------------------------------")



#Lasso Regression
# Import Lasso
from sklearn.linear_model import Lasso
lasso = Lasso(normalize=True)


min_test_MSE = 100
find_min_MSE = []
for alpha in alpha_space:
    lasso.alpha = alpha
    lasso.fit(X_train, y_train)
    lasso_y_train_pred = lasso.predict(X_train)
    lasso_y_test_pred = lasso.predict(X_test)
    find_min_MSE.append(mean_squared_error(y_test,lasso_y_test_pred))
    if mean_squared_error(y_test,lasso_y_test_pred) < min_test_MSE:
        min_alpha = alpha
        min_test_MSE = mean_squared_error(y_test,lasso_y_test_pred)
        train_MSE = mean_squared_error(y_train,lasso_y_train_pred)
        coef = lasso.coef_
        intercept = lasso.intercept_
        train_R2 = lasso.score(X_train,y_train)
        test_R2 = lasso.score(X_test,y_test)


plt.plot(alpha_space,find_min_MSE)
plt.xlabel('alpha')
plt.ylabel('Lasso_test_MSE')
plt.show()
print("Lasso Regression-> Lowest MSE of TEST: {} and its alpha is {} ".format(min_test_MSE,min_alpha))
print("Lasso Regression-> Lowest MSE of Train: {}".format(train_MSE))
print("Lasso Regression-> coefficient: {}".format(lasso.coef_))
print("Lasso Regression -> intercept: {}".format(lasso.intercept_))
print("Train R^2: {}".format(train_R2))
print("Test R^2: {}".format(test_R2))



# display_plot(lasso_scores, lasso_scores_std)

print("-------------------------------------------------------------------------")
print("My name is Han Wang")
print("My NetID is: 'hanw8'")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")