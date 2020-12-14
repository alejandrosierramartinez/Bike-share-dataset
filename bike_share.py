# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

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