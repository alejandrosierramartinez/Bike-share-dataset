# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import datetime

#load data
bikes = pd.read_csv('201902-fordgobike-tripdata.csv')
print(bikes.head())

#data info
print(bikes.info())

#check for missing values
print(bikes.isnull().sum())

#null values % of the dataset
print(bikes['member_gender'].isnull().sum()/bikes['member_gender'].count())

#remove missing values
bikes.dropna(inplace = True)
print(bikes.isnull().sum())

#check for unique values
print(bikes.nunique())

#member_birth_year exploration
print(bikes['member_birth_year'].min())
print(bikes['member_birth_year'].max())

#member_gender exploration
print(bikes['member_gender'].unique())

#convert start_time and end_time to datetype
## year, month, etc can be accesed with df.dt.year
bikes['start_time'] = pd.to_datetime(bikes.start_time, format = '%Y/%m/%d')
bikes['end_time'] = pd.to_datetime(bikes.end_time, format = '%Y/%m/%d')

#float to int
bikes['start_station_id'] = bikes['start_station_id'].astype(int)
bikes['end_station_id'] = bikes['end_station_id'].astype(int)
bikes['member_birth_year'] = bikes['member_birth_year'].astype(int)

#descriptive statistics for variables of interest
stats_col = ['duration_sec','member_birth_year']
print(bikes[stats_col].describe())