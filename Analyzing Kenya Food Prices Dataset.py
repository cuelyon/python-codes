#!/usr/bin/env python
# coding: utf-8

# ## SERIOUS KENYA FOOD PRICES DATASET

# This data set offers a rich repository of essentials from the World Food Programme Price Database. From different regions, scaled down to counties and markets, we get to take a look at the trend of prices of different food commoditities. Being a vast vast resource, we immerse ourselves and take a look at what's inside

# #### We Import the necessary modules

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import time

rc = {
    "axes.facecolor": "#CCFFE6",
    "figure.facecolor": "#CCFFE6",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}

sns.set(rc = rc)

from colorama import Style, Fore
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
mgt = Style.BRIGHT + Fore.MAGENTA
gld = Style.BRIGHT + Fore.YELLOW
res = Style.RESET_ALL


# #### We load the dataset 

# In[2]:


#Loading the data and dropping the first row;

data1 = pd.read_csv(r"C:\Users\User\Desktop\work\wfp_food_prices_ken.csv").drop(0)

#Reseting the index

data1.reset_index(drop = True, inplace = True)

#Print the reculting DataFrame

data1.head().style.set_properties(**{'background-color':'lightgreen','color':'royalblue','border-color':'#8b8c8c'})


# # Data cleaning and preprocesssing

# ### Checking for the null values

# In[6]:


missing_data1 = data1.isnull()

#Creating a heat map to visualize missing values
plt.figure(figsize=(12,6))

plt.subplot(131)

sns.heatmap(missing_data1, cmap = 'viridis', cbar = False)

plt.title('Missing data in data1', fontsize = 14, fontweight = 'bold', color = 'darkblue' ) 
plt.savefig('Missing data.png')
plt.show()


# The dataset is clean as it has no missing values

# ### We check the average start day and average end day for the data

# In[34]:


data1['date'] = pd.to_datetime(data1['date']) 
#data1['end_date'] = pd.to_endtime(data1['end_date'])
avg_date = data1['date'].mean() 
#avg_end_date = data1['end_date'].mean() 
print(f'Average Date is : {avg_date}') 
#print(f'Average End Date is : {avg_end_date}')


# ### We check the number of unique category and commodity in the data

# In[35]:


unique_categories = data1['category'].nunique()
unique_commodities = data1['commodity'].nunique()
print(f'The number of unique categories is: {unique_categories}')
print(f'The number of unique commodities is: {unique_commodities}')


# ### We check the unique values of the category and commodity

# In[36]:


unique_categorieslist = {'category':data1['category'].unique() for category in data1}
unique_commoditieslist = {'commodity':data1['commodity'].unique() for commodity in data1}
print(f'The list of unique categories is: {unique_categorieslist}')
print(f'The list of unique commodities is: {unique_commoditieslist}')


# ### We calculate the mean latitude and longitude

# In[37]:


# Convert latitude and longitude to numeic values
data1['latitude'] = pd.to_numeric(data1['latitude'])
data1['longitude'] = pd.to_numeric(data1['longitude'])

# Calculate the mean latitude and longitude

mean_latitude = data1['latitude'].mean()
mean_longitude = data1['longitude'].mean()

print(f'The mean latitude in the data is : {mean_latitude}')
print(f'The mean longitude in the data is : {mean_longitude}')


# # Explolatory Data Analysis

# In[38]:


import pandas as pd

# Load the CSV file
dataframe = pd.read_csv(r"C:\Users\User\Desktop\work\wfp_food_prices_ken.csv", header=None)

# Access the first sheet
first_sheet = dataframe.iloc[0]

# Access the second sheet
second_sheet = dataframe.iloc[1]

# Print the dataframes
print("First sheet:")
print(first_sheet)
print("\nSecond sheet:")
print(second_sheet)


# ## Data Visualization

# We create a bar graph to visualize the distribution of commodities in the data

# In[39]:


plt.figure(figsize = (10,6))
sns.countplot(data = data1, x = 'category', palette = 'viridis')
plt.xticks(rotation = 45)
plt.title('Distribution of commodity category in dataframe', fontsize = 14, fontweight = 'bold', color = 'magenta')
plt.xlabel('category', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.ylabel('count', fontsize = 12, fontweight = 'bold', color ='darkblue')
plt.savefig('Distribution of commodity category in dataframe.png')

#print the number o unique categories

unique_categories = data1['category'].nunique()
plt.text(0.7, 0.9, f"Unique Categories:{unique_categories}", transform=plt.gca().transAxes)

plt.show()


# The bar graph shows the different commodity category and their count. You can understand the composition of the data much better.

# In[20]:


num_rows = data1.shape[0]
print(f"Number of rows in the dataset: {num_rows}")


# In[43]:


import pandas as pd


# Convert the date strings to datetime objects
data1['date'] = pd.to_datetime(data1['date'])

# Access the first and last dates
first_date = data1['date'].min()
last_date = data1['date'].max()

print(f"First date: {first_date}")
print(f"Last date: {last_date}")


# In[44]:


# Calculate the difference between the first and last dates
days_difference = (last_date - first_date).days

print(f"Number of days between the first and last dates: {days_difference} days")


# In[52]:


data1['days_difference'] = (last_date - first_date).days

# Create a histogram to visualize the duration
plt.figure(figsize=(10, 6))
sns.histplot(data=data1, x='days_difference', bins=20, kde=True, color='skyblue')
plt.title("Distribution of Duration between Start and End Dates in data1", fontsize = 14, fontweight = 'bold', color = 'magenta')
plt.xlabel("Duration (days)", fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.ylabel("Frequency", fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.savefig('Distribution of Duration between Start and End Dates in data1.png')

# Print summary statistics
mean_duration = data1['days_difference'].mean()
median_duration = data1['days_difference'].median()
plt.text(0.6, 0.8, f"Mean Duration: {mean_duration:.2f} days", transform=plt.gca().transAxes)
plt.text(0.6, 0.7, f"Median Duration: {median_duration:.2f} days", transform=plt.gca().transAxes)

plt.show()


# # Correlation Analysis

# In[53]:


data1['latitude'] = pd.to_numeric(data1['latitude'])
data1['longitude'] = pd.to_numeric(data1['longitude'])
data1['price'] = pd.to_numeric(data1['price'])
data1['usdprice'] = pd.to_numeric(data1['usdprice'])


# In[19]:


# Assuming you have already loaded your 'data1' DataFrame
corr = data1.corr(numeric_only = True)
# Select only columns with numeric data
numeric_data1 = data1.select_dtypes(include='number')

# Check if numeric_data1 is empty
if numeric_data1.empty:
    print("No numeric columns found in the DataFrame.")
else:
    # Calculate the correlation between numeric columns
    correlation_data1 = numeric_data1.corr()

    # Plot a heatmap to show the correlation
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_data1, annot=True,)
    plt.title('Correlation for data1 (Numeric Values)', fontsize=14, fontweight='bold', color='magenta')
    plt.show()


# # Machine Learning Model

# We'll perform a simple linear regression prediction using data1. We'll use 'days_difference' as the dependent variable and 'admin2' as an independent variable to predict the duration of data availability

# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Changing date from string to float

#Select independent and dependent variables
X = pd.get_dummies(data1['admin2'], drop_first = True)
y = data1['float_date']

#split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Print the results
print(f' Mean Squared Error: {mse:.2f}')
print(f' R Squared (R2) Score: {r2:.2f}')
      
#Create a scatter plot of actual vs predicted values
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, color='skyblue')
plt.title('Linear Regression: Actual vs Predicted time', fontsize = 14, fontweight='bold', color='magenta')
plt.xlabel('Actual duration (days)', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.ylabel('Predicted duration (days)', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.savefig('Linear Regression: Actual vs Predicted.png')

plt.show()


# In[ ]:




