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
plt.title('Ride duration')
plt.xlabel('duration (s)')
plt.ylabel('Frequency')
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
plt.title('Ride duration (log scale)')
plt.xlabel('Duration (log s)')
plt.ylabel('Frequency')
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
plt.title('Rides per day')
plt.xlabel('Day of month')
plt.ylabel('Frequency')
plt.show()

#weekday
bikes['weekday'] = bikes['start_time'].dt.day_name()
bikes.groupby('weekday').count()['day'].sort_values(ascending=False).plot(kind='bar', fontsize=7, rot=0, width=0.9)
plt.title('Rides per weekday')
plt.xlabel('Day of week')
plt.ylabel('Frequency')
plt.show()

#hourly
bikes['hour'] = bikes['start_time'].dt.hour
bikes.groupby('hour').count()['day'].plot(kind='bar', rot=0, width=1)
plt.title('Rides per hour')
plt.xlabel('Hour of the day')
plt.ylabel('Frequency')
plt.show()

#gender
bikes.groupby('member_gender').count()['day'].sort_values(ascending=False).plot(kind='bar', fontsize=7, rot=0, width=0.5)
#sb.countplot(data = bikes, x = 'member_gender', color = sb.color_palette()[0], set_width=1)
plt.title('Member gender')
plt.xlabel('Gender')
plt.ylabel('Frequency')
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
plt.title('Member age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#bike id
print(bikes['bike_id'].nunique())
plt.hist(bikes['bike_id'])
plt.title('Bike_id usage')
plt.xlabel('Bike id')
plt.ylabel('Frequency')
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
sb.boxplot(y='duration_sec', x='user_type', data=bikes, showfliers=False, color = base_color, width=0.3)
plt.title('Duration by User Type')
plt.xlabel('User type')
plt.ylabel('Duration in seconds')
plt.show()

#duration by age
bikes['qbin'] = pd.qcut(bikes['user_age'], 5)
sb.boxplot(x="qbin", y='duration_sec', data=bikes, showfliers=False, color = base_color, width=0.3)
plt.xticks(np.arange(5), ['18-26', '27-30', '31-34', '35-41', '> 41'])  # Set text labels.
plt.title('Duration by User Age')
plt.xlabel('Age')
plt.ylabel('Duration in seconds')
plt.show()

#violin plot
bikes['duration_log'] = np.log10(bikes['duration_sec'])
sb.violinplot(x="qbin", y='duration_log', data=bikes, showfliers=False, color = base_color)
plt.xticks(np.arange(5), ['18-26', '27-30', '31-34', '35-41', '> 41'])  # Set text labels.
plt.title('Duration by User Age')
plt.xlabel('Age')
plt.ylabel('Duration in seconds log scale')
plt.show()

#duration by weekday or weekend
#create column is weekend with 1 as true 0 as false
bikes['weekday'] = pd.to_datetime(bikes['start_time']).dt.dayofweek  # monday = 0, sunday = 6
bikes['is_weekend'] = 0          # Initialize the column with default value of 0
bikes.loc[bikes['weekday'].isin([5, 6]), 'is_weekend'] = 1  # 5 and 6 correspond to Sat and Sun

# plot grouped by weekday or weekend
sb.boxplot(y='duration_sec', x='is_weekend', data=bikes, showfliers=False, color = base_color, width=0.3)
plt.xticks(np.arange(2), ['Weekday', 'Weekend'])  # Set text labels.
plt.title('Duration by Weekday or Weekend')
plt.xlabel('Day type')
plt.ylabel('Duration in seconds')
plt.show()

#duration by morning, afternoon and night
hour_bins = [0, 6, 12, 18, 24]
labels = ['00:00-05:59', '06:00-11:59', '12:00-17:59', '18:00-23:59']
bikes['hour_bin'] = pd.cut(bikes.start_time.dt.hour, hour_bins, labels=labels, right=False)

sb.boxplot(y='duration_sec', x='hour_bin', data=bikes, showfliers=False, color = base_color, width=0.3)
plt.title('Duration by hour')
plt.xlabel('Hour of the day')
plt.ylabel('Duration in seconds')
plt.show()

#rides by station
#Take 500 observation as sample
samples = np.random.choice(bikes.shape[0], 5000, replace = False)
bikes_samp = bikes.loc[bikes.index.intersection(samples),:]

#count rides
plt.scatter(bikes_samp['start_station_latitude'], bikes_samp['start_station_longitude'], alpha=0.005)
plt.title('Rides by station')
plt.xlabel('Station latitude')
plt.ylabel('Station longitude')
plt.show()

#duration rides
bikes_samp['scaled_duration'] = bikes['duration_sec']*0.0005
plt.scatter(bikes_samp['start_station_latitude'], bikes_samp['start_station_longitude'], s = bikes_samp['scaled_duration'])
plt.title('Duration by station location')
plt.xlabel('Station latitude')
plt.ylabel('Station longitude')
plt.show()

#subscription by age and gender
plt.figure(figsize = [12, 12])
plt.subplot(2, 1, 1)
sb.countplot(data = bikes, x = 'qbin', hue = 'user_type', palette = 'Blues')
plt.legend(frameon=False)
plt.title('User type by age')
plt.xticks(np.arange(5), ['18-26', '27-30', '31-34', '35-41', '> 41'])  # Set text labels.
plt.xlabel('Age')
ax = plt.subplot(2, 1, 2)
sb.countplot(data = bikes, x = 'member_gender', hue = 'user_type', palette = 'Blues')
plt.legend(frameon=False)
plt.title('User type by gender')
plt.xlabel('Gender')
plt.show()

#subscription by station
#count rides
#filter by subscribers from sample
bikes_subs = bikes_samp.query('user_type == "Subscriber"')
plt.scatter(bikes_subs['start_station_latitude'], bikes_subs['start_station_longitude'], alpha=0.01)
plt.title('Bike subscriptors location')
plt.xlabel('Station latitude')
plt.ylabel('Station longitude')
plt.show()

#bike_share_for_all location
#filter bike_share_for_all trips from sample
bike_share_for_all = bikes_samp.query('bike_share_for_all_trip == "Yes"')
plt.scatter(bike_share_for_all['start_station_latitude'], bike_share_for_all['start_station_longitude'], alpha=0.05)
plt.title('Bike share for all trip location')
plt.xlabel('Station latitude')
plt.ylabel('Station longitude')
plt.show()

#multivariate exploration
#create column for subscription
bikes.loc[bikes['bike_share_for_all_trip'].isin(['Yes']), 'user_type'] = 'Bike share for all'

#duration across user type and weekday
ax = sb.pointplot(data = bikes, hue = 'user_type', y = 'duration_sec', x = 'is_weekend',
           palette = 'Blues', linestyles = '')
plt.title('Duration by user type and weekday')
plt.xticks(np.arange(2),['Weekday', 'Weekend'])
plt.legend(frameon=False )
plt.xlabel('Day of week')
plt.ylabel('Duration in seconds')
plt.show()

#age by day of week and user type
sb.boxplot(data=bikes, hue='user_type', y='user_age', x='is_weekend', palette = 'Blues', showfliers=False, width=0.5)
plt.xticks(np.arange(2),['Weekday', 'Weekend'])
plt.legend(frameon=False)
plt.title('User age by day of week rides and subscription type')
plt.xlabel('')
plt.ylabel('Age')
plt.show()

#age by gender and user type
sb.boxplot(data=bikes, hue='user_type', y='user_age', x='member_gender', palette = 'Blues', showfliers=False, width=0.5)
plt.legend(frameon=False)
plt.title('User age by gender and subscription type')
plt.xlabel('')
plt.ylabel('Age')
plt.show()