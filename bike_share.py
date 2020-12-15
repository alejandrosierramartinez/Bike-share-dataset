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

#create age column
bikes['user_age'] = 2019 - bikes['member_birth_year']

#descriptive statistics for variables of interest
stats_col = ['duration_sec','member_birth_year', 'user_age']
print(bikes[stats_col].describe())

#univariate exploration
#duration_sec
binsize = 600
bins = np.arange(0, bikes['duration_sec'].max() + binsize, binsize)
plt.hist(data = bikes, x ='duration_sec', bins = bins)
plt.xlabel('Ride duration (s)')
plt.show()

#show high outliers
high_outliers = (bikes['duration_sec'] > 43200)
print(high_outliers.sum())

#convert to log scale
log_binsize = 0.025
bins = 10 ** np.arange(0, np.log10(bikes['duration_sec'].max())+log_binsize, log_binsize)
plt.figure(figsize=[8, 5])
plt.hist(data = bikes, x = 'duration_sec', bins = bins)
plt.xscale('log')
plt.xticks([600, 1e3, 2e3, 5e3, 1e4, 2e4], [600, '1k', '2k', '5k', '10k', '20k'])
plt.xlabel('Ride duration log scale (s)')
plt.show()

#start_time
#per year
month = bikes['start_time'].dt.month
plt.hist(month)
plt.xlabel('Month')
plt.show()