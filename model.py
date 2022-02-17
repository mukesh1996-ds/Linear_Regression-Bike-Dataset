from load_csv import load_csv
from information_data import check_shape, check_top_records, check_info, check_describe
from datetime_handle import handle_data
import pandas as pd
from algorithm import lin_reg, lin_reg_2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns



df = load_csv('G:\Kaggle_compitation\Linear Regression\Dataset\day.csv')
print(df.head())

print(check_shape(df))
print(check_top_records(df))
print(check_info(df))
print(handle_data(df))

# Dummy variable
# creating dummies 
season_dummy = pd.get_dummies(df.season,drop_first=True)
weather_dummy = pd.get_dummies(df.weathersit,drop_first=True)
month_dummy = pd.get_dummies(df.mnth,drop_first=True)
weekday_dummy = pd.get_dummies(df.weekday,drop_first=True)
final_df = df.join(season_dummy)
final_df = final_df.join(weather_dummy)
final_df = final_df.join(month_dummy)
final_df = final_df.join(weekday_dummy)
final_df.drop(['season','weathersit','mnth','weekday'], axis=1, inplace=True)
print(final_df.shape)
final_df.head()

print(check_describe(df))


# Train test split ######################################################################
X = final_df.drop('cnt', axis=1)
y = final_df.cnt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
# Sacling the numerical variables #######################################################
num_cols = ['temp','atemp','hum','windspeed']
scaler = MinMaxScaler()
scaler.fit(X_train[num_cols])
train_scaled = scaler.transform(X_train[num_cols])
test_scaled = scaler.transform(X_test[num_cols])
# Scaling training data
X_train[num_cols] = train_scaled
# Scaling test data
X_test[num_cols] = test_scaled
# Applying Linear Regression ############################################################
# Creating a base model to check the initial results
model_1 = lin_reg(X_train, y_train, X_test, y_test)
print(model_1)
model_2 = lin_reg_2(X_train, y_train, X_test, y_test)
print(model_2)