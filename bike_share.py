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
#expore variable
year = bikes['start_time'].dt.year
print(year.min())
print(year.max())
month = bikes['start_time'].dt.month
print(month.min())
print(month.max())

#per day
bikes['day'] = bikes['start_time'].dt.day
binsize = 1
bins = np.arange(1, 28, binsize)
plt.hist(data = bikes, x ='day', bins = bins)
plt.xlabel('Rides per day')
plt.show()

#weekday
bikes['weekday'] = bikes['start_time'].dt.day_name()
bikes.groupby('weekday').count()['day'].sort_values(ascending=False).plot(kind='bar', fontsize=7, rot=0, width=0.9)
plt.xlabel('Rides per weekday')
plt.show()

#hourly
bikes['hour'] = bikes['start_time'].dt.hour
bikes.groupby('hour').count()['day'].plot(kind='bar', rot=0, width=1)
plt.xlabel('Rides per hour')
plt.show()

#gender
sb.countplot(data = bikes, x = 'member_gender', color = sb.color_palette()[0])
plt.xlabel('Member gender')
plt.show()

#gender ratio
male_proportion = (bikes['member_gender'] == 'Male').sum()/bikes['member_gender'].count()
female_proportion = (bikes['member_gender'] == 'Female').sum()/bikes['member_gender'].count()
other_proportion = (bikes['member_gender'] == 'Other').sum()/bikes['member_gender'].count()
print(male_proportion)
print(female_proportion)
print(other_proportion)

#member age
bins = np.arange(18, 80, 2)
plt.hist(bikes['user_age'], bins = bins)
plt.xlabel('Member age')
plt.show()

#bike id
print(bikes['bike_id'].nunique())
plt.hist(bikes['bike_id'])
plt.xlabel('Bike id counts')
plt.show()

#user type
print(bikes['user_type'].unique())
customer_proportion = (bikes['user_type'] == 'Customer').sum()/bikes['user_type'].count()
suscriber_proportion = (bikes['user_type'] == 'Subscriber').sum()/bikes['user_type'].count()
print(customer_proportion)
print(suscriber_proportion)

#bike share for all
print(bikes['bike_share_for_all_trip'].unique())
bike_share_for_all_trip_proportion = (bikes['bike_share_for_all_trip'] == 'Yes').sum()/bikes['bike_share_for_all_trip'].count()
print(bike_share_for_all_trip_proportion)


#bivariate exploration
#corr coef
corr_coef = np.corrcoef(bikes['duration_sec'],bikes['user_age'])
print(corr_coef)

#rides by user
print(bikes.groupby('user_type').duration_sec.count())
print(bikes.groupby('user_type').duration_sec.mean())
print(bikes.groupby('user_type').duration_sec.count().Customer/(bikes.groupby('user_type').duration_sec.count().Customer + bikes.groupby('user_type').duration_sec.count().Subscriber))
print(bikes.groupby('user_type').duration_sec.count().Subscriber/(bikes.groupby('user_type').duration_sec.count().Customer + bikes.groupby('user_type').duration_sec.count().Subscriber))

#plot duration by user type box
base_color = sb.color_palette()[0]
sb.boxplot(y='duration_sec', x='user_type', data=bikes, showfliers=False, color = base_color);
plt.xlabel('Duration by User Type')
plt.ylabel('Duration in seconds')
plt.show()

#duration by age
bikes['qbin'] = pd.qcut(bikes['user_age'], 5)
sb.boxplot(x="qbin", y='duration_sec', data=bikes, showfliers=False, color = base_color)
plt.xticks(np.arange(5), ['18-26', '27-30', '31-34', '35-41', '> 41'])  # Set text labels.
plt.xlabel('Duration by User Age')
plt.ylabel('Duration in seconds')
plt.show()

#violin plot
bikes['duration_log'] = np.log10(bikes['duration_sec'])
sb.violinplot(x="qbin", y='duration_log', data=bikes, showfliers=False, color = base_color)
plt.xticks(np.arange(5), ['18-26', '27-30', '31-34', '35-41', '> 41'])  # Set text labels.
plt.xlabel('Duration by User Age')
plt.ylabel('Duration in seconds log scale')
plt.show()

#duration by weekday or weekend
#create column is weekend with 1 as true 0 as false
bikes['weekday'] = pd.to_datetime(bikes['start_time']).dt.dayofweek  # monday = 0, sunday = 6
bikes['is_weekend'] = 0          # Initialize the column with default value of 0
bikes.loc[bikes['weekday'].isin([5, 6]), 'is_weekend'] = 1  # 5 and 6 correspond to Sat and Sun

# plot grouped by weekday or weekend
sb.boxplot(y='duration_sec', x='is_weekend', data=bikes, showfliers=False, color = base_color);
plt.xticks(np.arange(2), ['Weekday', 'Weekend'])  # Set text labels.
plt.xlabel('Duration by Weekday or Weekend')
plt.ylabel('Duration in seconds')
plt.show()