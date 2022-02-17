from load_csv import load_csv
from information_data import check_info,check_shape,check_top_records
import pandas as pd
from datetime import date
    

data = load_csv('G:\Kaggle_compitation\Linear Regression\Dataset\day.csv')
print(check_info(data))
print(check_shape(data))
print(check_top_records(data))

# date handling
def handle_data(data):
    data.dteday = pd.to_datetime(data.dteday)
    data["day"] = [i.day for i in data.dteday]
    data.drop(['dteday','registered','casual'],axis=1, inplace=True)
    data.weathersit = data.weathersit.map({1:'Clear',2:'Mist',3:'Light_Snow_Rain',4:'Heavy_Snow_Rain'})
    data.season = data.season.map({1:'Spring',2:'Summer',3:'Fall',4:'Winter'})
    data.mnth = data.mnth.replace((1,2,3,4,5,6,7,8,9,10,11,12),('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
    data.weekday = data.weekday.map({0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'})
    return data

# Testing 

print(handle_data(data))
print(check_top_records(data))