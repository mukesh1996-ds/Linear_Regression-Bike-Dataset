from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

# creating a method for linear regression so that we can re use it again with different features
def lin_reg(X_train, y_train, X_test, y_test):
  X_train = sm.add_constant(X_train) # adding a constant
  model = sm.OLS(y_train, X_train).fit()
  X_test = sm.add_constant(X_test)
  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)
  print("R2 Score of the training data:",r2_score(y_pred = y_pred_train, y_true = y_train))
  print("R2 Score of the testing data:",r2_score(y_pred = y_pred_test, y_true = y_test))
  print(model.summary())
  sns.regplot(y_test,y_pred_test)
  return model

''' 
although we have a function created above for linear regression, we have created here another one as we do not need
to print summary and the chart everytime we create a model to check the p values.
'''
def lin_reg_2(X_train, y_train, X_test, y_test):
  X_train = sm.add_constant(X_train) # adding a constant
  model = sm.OLS(y_train, X_train).fit()
  X_test = sm.add_constant(X_test)
  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)
  print("R2 Score of the training data:",r2_score(y_pred = y_pred_train, y_true = y_train))
  print("R2 Score of the testing data:",r2_score(y_pred = y_pred_test, y_true = y_test))
  return model