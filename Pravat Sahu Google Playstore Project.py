#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse


# In[2]:


#Step 1. Load the data using pandas.
appdata = pd.read_csv('/Users/priya/Pravat/Simplilearn Data Analytics/Class2/project/googleplaystore.csv')
appdata.head(5)


# In[3]:


appdata.info()


# In[4]:


appdata.columns


# In[5]:


#Step 2. Check for null values in the data. Get the number of null values for each column.
appdata.isnull().sum()


# In[6]:


#Step 3. Drop records with nulls in any of the columns.
appdata.dropna(axis = 0, inplace=True)


# In[7]:


#verify
appdata.isna().sum()


# In[8]:


appdata.shape


# In[9]:


#Step 4. Fix incorrect type and inconsistent formatting.
appdata['Size']


# In[10]:


appdata['Size'].unique()


# In[11]:


#Format Size coloumn
appdata['Size'] = appdata.Size.replace('Varies with device','0k')
appdata['Size'] = appdata.Size.str.replace('M','000')
appdata['Size'] = appdata.Size.str.replace('k','')
appdata['Size'] = appdata.Size.replace('1,000+',1000)
appdata['Size'] = appdata['Size'].astype(float)


# In[12]:


#verify
appdata['Size'].dtype           


# In[13]:


#verify
appdata['Size']


# In[14]:


#change datatype for Reviews column to float
appdata['Reviews']= appdata['Reviews'].astype(float)


# In[15]:


#Check Installs coloumn
appdata['Installs'].unique()


# In[16]:


#Format Installs coloumn
appdata['Installs'] = appdata.Installs.str.replace(',','')
appdata['Installs'] = appdata.Installs.str.replace('+','')
appdata['Installs'] = appdata.Installs.str.replace('Free','')
appdata['Installs'] = appdata['Installs'].astype(float)
appdata['Installs'] = appdata['Installs'].astype(float)


# In[17]:


#verify data type for the coloumn
appdata['Installs'].dtype


# In[18]:


appdata.Installs


# In[19]:


#Check Price coloumn
appdata['Price'].unique()


# In[20]:


#Format Price coloumn
appdata['Price'] = appdata.Price.str.replace('$','').astype(float)


# In[21]:


#verify
appdata['Price'].dtype


# In[22]:


appdata.Price


# In[23]:


#Step 5. Sanity checks: 
# all ratings are between 1 to 5
appdata['Rating'].unique()


# In[24]:


appdata['Rating'].dtype


# In[25]:


# drop all rows with Ratings outside the 1-5 range
RatingOut = appdata[(appdata['Rating'] < 0) & (appdata['Rating'] > 5)].index
appdata.drop(RatingOut , inplace = True)


# In[26]:


#verify the rows and coloumns
appdata.shape 


# In[27]:


#Reviews should not be more than installs as only those who installed can review the app. If there are any such records, drop them.
appdata = appdata[appdata['Reviews'] <= appdata['Installs']]


# In[28]:


#verify the rows and coloumns after drop
appdata.shape


# In[29]:


#For free apps (type = “Free”), the price should not be >0. Drop any such rows.
#get indexes where free Types have a price over 0
priceindexOut = appdata[(appdata['Price'] >= 0.1) & (appdata['Type'] == 'Free')].index
# drop these row 
appdata.drop(priceindexOut ,inplace = True)


# In[30]:


#verify after drop
appdata.shape


# In[31]:


#Step 6. Performing univariate analysis:
#find possible outliers in Price colomns and Review columns using Box Plot


# In[32]:


appdata['Price'].describe()


# In[33]:


#Boxplot for Price
plt.figure(figsize= (15, 5))
sns.boxplot(x = appdata.Price, color = 'mediumaquamarine',)
plt.show()


# - From the statistical analysis table and price box plot it is observed that apps over $100 are outliers.

# In[34]:


#Boxplot for Review
plt.figure(figsize= (15, 5))
sns.boxplot(x = appdata.Reviews, color = 'mediumaquamarine',)
plt.show()


# In[35]:


appdata['Reviews'].describe()


# - From the statistcial analysis table and box plot, it is observed that the averge number of reviews are 5,14,760 with a standard deviation of 31,46,169 between values. This deviation is due to several outliers in reviews column.

# In[36]:


#Histogram for Rating
plt.figure(figsize= (10,5))
sns.histplot(appdata.Rating, bins = 100, color =  'darkgreen', edgecolor = 'black')
plt.show()


# - From the rating histogram it is observed that most apps lean/skewed towards high ratings.

# In[37]:


#Histogram for Size
plt.figure(figsize= (10,5))
sns.displot(appdata.Size, kind = 'kde', color= 'steelblue')
plt.show()


# - From the size displot histogram, it is observed that most apps size are below 20,000 kb.

# In[38]:


#Step 7. Outlier treatment:


# In[39]:


#drop Price rows which are above 200
appdata = appdata[appdata['Price'] < 200]
#verify
appdata.shape


# In[40]:


#Drop Review rows with over 2 million reviews
appdata = appdata[appdata['Reviews'] <= 2000000]
#verify
appdata.shape


# In[41]:


#Apps having very high number of installs should be dropped from the analysis so drop rows with 100,000,000 and more Installs
appdata = appdata[appdata['Installs'] <= 100000000]
#verify
appdata.shape


# In[42]:


#Find out the different percentiles – 10, 25, 50, 70, 90, 95, 99
percentiles = appdata[['Rating','Reviews','Size','Installs','Price']]


# In[43]:


#10, 25, 50, 70, 90, 95, 99 percentiles
print("10th percentile : ",
       np.percentile(percentiles, 10))
print("25th percentile : ",
       np.percentile(percentiles, 25))
print("50th percentile : ", 
       np.percentile(percentiles, 50))
print("70th percentile : ",
       np.percentile(percentiles, 70))
print("90th percentile : ",
       np.percentile(percentiles, 90))
print("95th percentile : ",
       np.percentile(percentiles, 95))
print("99th percentile  : ",
       np.percentile(percentiles, 99))


# In[44]:


#find out remining outliers with boxplots 
percentiles.boxplot(rot = 50)


# In[45]:


#zoomed into Rating box plot to find out outliers
percentiles.boxplot(column=['Rating'])


# In[46]:


#remove outliers from Ratings
RatingOut1 = appdata[(appdata['Rating'] < 3.5) ].index
appdata.drop(RatingOut1 , inplace = True)


# In[47]:


appdata.shape


# In[48]:


#zoomed into outliers from price coloumn
percentiles.boxplot(column = ['Price'], figsize = (6,6))


# In[49]:


#remove outliers from price coloumn, anything above $40 seems outliers
PriceOut = appdata[(appdata['Price'] > 40)].index
appdata.drop(PriceOut , inplace = True)
#verify
appdata.shape


# In[50]:


#zoomed into outliers in Installs coloumn
percentiles.boxplot(column = ['Installs'])


# In[51]:


#remove outliers from Installs coloumns
Installsout = appdata[(appdata['Installs'] >= 100000000)].index
appdata.drop(Installsout , inplace = True)
#verify
appdata.shape


# In[52]:


#Step 8. Bivariate analysis
#Scatter plot for Rating vs. Price
plt.figure(figsize=(10, 9))
sns.scatterplot(
    data=appdata, x="Rating", y="Price", hue="Rating",
    sizes=(20, 200),  legend="full")


# observation in Rating vs Price:
# - Most Ratings for the apps are with in 4.4 to 5.0 and apps Prices are between 0 and $10. It is also observed that higher Price apps does not mean better ratings.

# In[53]:


#Scatter plot for Rating vs. Reviews
plt.figure(figsize=(10, 9))
sns.scatterplot(
    data=appdata, x="Rating", y="Reviews", hue="Rating",
    sizes=(20, 200), legend="full")


# Observation in Rating vs Reviews:
# - Better ratings apps have most reviews although not everytime is the case.

# In[54]:


#Box plot for Rating vs. Content Rating
plt.figure(figsize=(12, 5))
rvcr = sns.boxplot(data = appdata,x ='Content Rating', y ='Rating', palette ='Set3')


# Observation in Rating vs Content Ratings:
# - From the box plot, there does not seem to be much difference between Content Ratings in relation to Ratings.

# In[55]:


#Box plot for Rating vs. Category
plt.figure(figsize=(12, 5))
rvca = sns.boxplot(data = appdata,x ='Category', y ='Rating', palette ='Set3')
plt.show(plt.setp(rvca.get_xticklabels(), rotation = 80))


# In[56]:


#categorical data in relation to Genres
plt.figure(figsize=(20, 10))
cgen = sns.barplot(data = appdata, x ='Genres', y ='Rating', palette ='Set2') 
plt.show (plt.setp(cgen.get_xticklabels(), rotation=90))


#  Box plot for Rating vs. Genres
# - Comics;Creativity and Board Pretend Play has rate best ratings.

# In[57]:


#Step 9.Data preprocessing


# In[58]:


inp1 = appdata.copy()


# In[59]:


inp1.head(2)


# In[60]:


inp1.describe()


# In[61]:


# Apply log transformation to reduce the skew in Reviews and Installs.
inp1.Reviews = np.log1p(inp1.Reviews.values)
inp1.Installs = np.log1p(inp1.Installs.values)


# In[62]:


#verify after apply log transformation.
inp1.describe()


# In[63]:


#Drop columns App, Last Updated, Current Ver, and Android Ver.
inp1 = inp1.drop(['App', 'Last Updated', 'Current Ver', 'Android Ver'], axis = 1)


# In[64]:


#verify
inp1.head(2)


# In[65]:


## convert the object type variable and convert them to dumies 
inp2 = pd.get_dummies(inp1, columns = ['Category','Type','Content Rating','Genres'])


# In[66]:


inp2


# In[67]:


#Step 10. Train test split  and apply 70-30 split. Name the new dataframes df_train and df_test
df_train, df_test = split(inp2, test_size = 0.30, random_state = 12)


# In[68]:


df_train.shape


# In[69]:


df_test.shape


# In[70]:


#Step 11. Model bulding
lm = LinearRegression()


# In[71]:


# fit the model 
X = df_train.drop(columns=['Rating'])
Y = df_train.Rating
lm = lm.fit(X,Y)


# In[72]:


lm.coef_


# In[73]:


lm.intercept_


# In[74]:


lm.score(X,Y)  # R squared value for the df_train data


# In[75]:


#ycap that is prediction for the df_train data 
ycap = lm.predict(X)
print(ycap)


# In[76]:


#Step 12. Make predictions for df_test data
df_test_x = df_test.drop(columns = ['Rating'])


# In[77]:


y_pred = lm.predict(df_test_x)
print (y_pred)


# In[78]:


mse(y_true = df_test.Rating, y_pred = y_pred, squared = False) #MSE value for df_test data

