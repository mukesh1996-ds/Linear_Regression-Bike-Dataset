import pandas as pd
import numpy as np

def load_csv(data):
    data = pd.read_csv('G:\Kaggle_compitation\Linear Regression\Dataset\day.csv')
    return data

# Testing 

df = load_csv('G:\Kaggle_compitation\Linear Regression\Dataset\day.csv')
print(df.head())
