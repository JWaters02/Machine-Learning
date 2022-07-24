#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[218]:


import os
import pandas as pd
import numpy as np
import wandb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import sklearn.model_selection as model_selection
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings("ignore")

survey = pd.read_csv("kaggle_survey_2021_responses.csv")
survey.head(5)


# ### Remove question descriptions row

# In[219]:


survey = survey.iloc[1:, :]
survey.head(5)


# ### Select specific features

# In[220]:


# df = pd.DataFrame(columns=['Gender', 'Country', 'Years Coding', 'Programming Languages Used', 'Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash', 'MATLAB', 'No Language', 'Other Language', 'Years Using ML', 'Income'])
df = pd.DataFrame(columns=['Gender', 'Country', 'Years Coding', 'Years Using ML', 'Income'])
df['Gender'] = survey['Q2']
df['Country'] = survey['Q3']
df['Years Coding'] = survey['Q6']
# df['Programming Languages Used'] = survey[survey.columns[7:21]].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)
# df['Python'] = survey['Q7_Part_1']
# df['R'] = survey['Q7_Part_2']
# df['SQL'] = survey['Q7_Part_3']
# df['C'] = survey['Q7_Part_4']
# df['C++'] = survey['Q7_Part_5']
# df['Java'] = survey['Q7_Part_6']
# df['Javascript'] = survey['Q7_Part_7']
# df['Julia'] = survey['Q7_Part_8']
# df['Swift'] = survey['Q7_Part_9']
# df['Bash'] = survey['Q7_Part_10']
# df['MATLAB'] = survey['Q7_Part_11']
# df['No Language'] = survey['Q7_Part_12']
# df['Other Language'] = survey['Q7_OTHER']
df['Years Using ML'] = survey['Q15']
df['Income'] = survey['Q25']
df


# ### Check for number of unknowns per feature

# In[221]:


print (df['Gender'].isna().sum())
print (df['Country'].isna().sum())
print (df['Years Coding'].isna().sum())
# print (df['Programming Languages Used'].isna().sum())
print (df['Years Using ML'].isna().sum())
# print (df['Income'].isna().sum())


# ### Clean missing data

# In[222]:


df['Years Using ML'].fillna('I do not use machine learning methods', inplace=True)
df = df.dropna(subset=['Income'])
df = df.reset_index()
df = df.drop(columns=['index'])
df


# ### Handling conflicting data

# In[223]:


# rows_to_remove = []
# for i in df.index:
#     if df['Years Coding'][i] == 'I have never written code' and 'None' not in df['Programming Languages Used'][i] and df['Programming Languages Used'].any():
#         rows_to_remove.append(i)
# # for i in reversed(rows_to_remove):
# #     df = df.drop(df.index[i])
# df = df.drop(df.index[rows_to_remove])
# df


# ### Encode income levels

# In[224]:


income_level_order = [['$0-999',
                        '1,000-1,999',
                        '2,000-2,999',
                        '3,000-3,999',
                        '4,000-4,999',
                        '5,000-7,499',
                        '7,500-9,999',
                        '10,000-14,999',
                        '15,000-19,999',
                        '20,000-24,999',
                        '25,000-29,999',
                        '30,000-39,999',
                        '40,000-49,999',
                        '50,000-59,999',
                        '60,000-69,999',
                        '70,000-79,999',
                        '80,000-89,999',
                        '90,000-99,999',
                        '100,000-124,999',
                        '125,000-149,999',
                        '150,000-199,999',
                        '200,000-249,999',
                        '250,000-299,999',
                        '300,000-499,999',
                        '$500,000-999,999',
                        '>$1,000,000']] 
encoder = OrdinalEncoder(categories = income_level_order)
df['Income Level'] = encoder.fit_transform(df['Income'].values.reshape(-1, 1))
df['High Income'] = df['Income Level'].apply(lambda x:0 if x < 16 else 1)
df


# ### Clean income level outliers

# In[225]:


df['Income Level'].describe()


# In[226]:


plt.figure() 
df['Income Level'].plot(kind='box', fontsize=15)
plt.show()


# In[227]:


plt.figure()
df['Income Level'].plot(kind='hist',grid=True, bins=20, fontsize=15) 
plt.xlabel('Income Level')
plt.ylabel('Count')
plt.show()


# ### Group genders

# In[228]:


other_list = ['Prefer not to say', 'Prefer to self-describe', 'Nonbinary']
for i in df.index:
    if df['Gender'][i] in other_list:
        df['Gender'][i] = 'Other'
df


# ### Plot genders

# In[229]:


title = 'Genders'
fig, ax  = plt.subplots(figsize=(16, 8))
fig.suptitle(title, fontsize = 20)
explode = (0.05, 0.2, 0.4)
labels = list(df.iloc[1:].Gender.value_counts().index)
sizes = df.iloc[1:].Gender.value_counts().values
ax.pie(sizes, explode=explode,startangle=60, labels=labels,autopct='%1.0f%%', pctdistance=0.7, colors=["#d45d00","#ff9100","#eaaa00"])
ax.add_artist(plt.Circle((0,0),0.4,fc='white'))
plt.show()


# 

# ### Country Chloropleth map

# In[230]:


countries = df['Country'].value_counts()
fig = go.Figure(data=go.Choropleth(
    locations = countries.index, 
    z = countries,
    locationmode = 'country names',
    colorbar_title = "Respondants"
))

fig.update_layout(
    title_text = 'Survey Respondants by Country'
)

fig.show()


# ### Top countries

# In[231]:


def select_countries(x):
    if x in top_countries.index: x = x 
    else: x = "Other"
    return x
   
top_countries = df['Country'].value_counts().head(8)
df['Top Country'] = df['Country'].apply(lambda x: select_countries(x))
df


# ### Plot top countries vs income

# In[232]:


crosstab = pd.crosstab(df["Top Country"], df['High Income'], normalize=True)
crosstab.plot(kind = 'bar', stacked = 'true', title = 'Proportions of respondants in high or low income per top country')
plt.xlabel('Country')
plt.ylabel('Proportion')
plt.show() 


# ### Encode years coding & years using ML

# In[233]:


years_coding_order = [['I have never written code',
                        '< 1 years',
                        '1-3 years',
                        '3-5 years',
                        '5-10 years',
                        '10-20 years',
                        '20+ years']] 
encoder = OrdinalEncoder(categories = years_coding_order)
df['Years Coding Level'] = encoder.fit_transform(df['Years Coding'].values.reshape(-1, 1))

years_using_ML_order = [['I do not use machine learning methods',
                        'Under 1 year',
                        '1-2 years',
                        '2-3 years',
                        '3-4 years',
                        '4-5 years',
                        '5-10 years',
                        '10-20 years',
                        '20 or more years']] 
encoder = OrdinalEncoder(categories = years_using_ML_order)
df['Years Using ML Level'] = encoder.fit_transform(df['Years Using ML'].values.reshape(-1, 1))
df


# In[234]:


# # Make temp df
# df_temp = pd.DataFrame(columns=['Gender', 'Income'])
# df_temp['Gender'] = df['Gender']
# df_temp['Income'] = df['Income Level']

# # Get mean income for each top 8 countries
# gender_dict = {
#     'Male' : [0, 0, 0],
#     'Female' : [0, 0, 0],
#     'Other' : [0, 0, 0]
# }
# for gender in gender_dict:
#     for i in df.index:
#         if df_temp['Gender'][i] == gender:
#             gender_dict[gender][0] += df_temp['Income'][i]
#             gender_dict[gender][1] += 1
#     gender_dict[gender][2] = gender_dict[gender][0] / gender_dict[gender][1]

# # Put dict into data frame
# ret = pd.DataFrame.from_dict(gender_dict)
# ret = ret.iloc[2:, :]
# ret = ret.T
# ret.columns = ['Mean Income Level']

# # Plot dataframe into bar chart
# ax = ret.plot.barh(title='Mean Income Level by Gender')
# ax.set_ylabel('Gender')


# ### Get mean income level per country

# In[235]:


# Make temp df
df_temp = pd.DataFrame(columns=['Country', 'Income'])
df_temp['Country'] = df['Top Country']
df_temp['Income'] = df['Income Level']

# Get mean income for each top 8 countries
country_dict = {
    'Brazil' : [0, 0, 0],
    'China' : [0, 0, 0],
    'India' : [0, 0, 0],
    'Japan' : [0, 0, 0],
    'Nigeria' : [0, 0, 0],
    'Other' : [0, 0, 0],
    'Russia' : [0, 0, 0],
    'United States of America' : [0, 0, 0]
}
for country in country_dict:
    for i in df.index:
        if df_temp['Country'][i] == country:
            country_dict[country][0] += df_temp['Income'][i]
            country_dict[country][1] += 1
    country_dict[country][2] = country_dict[country][0] / country_dict[country][1]

# Put dict into data frame
ret = pd.DataFrame.from_dict(country_dict)
ret = ret.iloc[2:, :]
ret = ret.T
ret.columns = ['Mean Income Level']

# Plot dataframe into bar chart
ax = ret.plot.barh(title='Mean Income Level by Top 8 Countries')
ax.set_ylabel('Country')


# ### Get mean income level per years coding

# In[236]:


# Make temp df
df_temp = pd.DataFrame(columns=['Years Coding', 'Income'])
df_temp['Years Coding'] = df['Years Coding']
df_temp['Income'] = df['Income Level']

# Get mean income for each top 8 countries
coding_dict = {
    'I have never written code' : [0, 0, 0],
    '< 1 years' : [0, 0, 0],
    '1-3 years' : [0, 0, 0],
    '3-5 years' : [0, 0, 0],
    '5-10 years' : [0, 0, 0],
    '10-20 years' : [0, 0, 0],
    '20+ years' : [0, 0, 0]
}
for level in coding_dict:
    for i in df.index:
        if df_temp['Years Coding'][i] == level:
            coding_dict[level][0] += df_temp['Income'][i]
            coding_dict[level][1] += 1
    coding_dict[level][2] = coding_dict[level][0] / coding_dict[level][1]

# Put dict into data frame
ret = pd.DataFrame.from_dict(coding_dict)
ret = ret.iloc[2:, :]
ret = ret.T
ret.columns = ['Mean Income Level']

# Plot dataframe into bar chart
ax = ret.plot.barh(title='Mean Income Level by Years Coding Categories')
ax.set_ylabel('Years Spent Coding')


# ### Get mean income level per years using ML

# In[237]:


# Make temp df
df_temp = pd.DataFrame(columns=['Years Using ML', 'Income'])
df_temp['Years Using ML'] = df['Years Using ML']
df_temp['Income'] = df['Income Level']

# Get mean income for each top 8 countries
years_ML_dict = {
    'I do not use machine learning methods' : [0, 0, 0],
    'Under 1 year' : [0, 0, 0],
    '1-2 years' : [0, 0, 0],
    '2-3 years' : [0, 0, 0],
    '3-4 years' : [0, 0, 0],
    '5-10 years' : [0, 0, 0],
    '10-20 years' : [0, 0, 0],
    '20 or more years' : [0, 0, 0]
}
for years in years_ML_dict:
    for i in df.index:
        if df_temp['Years Using ML'][i] == years:
            years_ML_dict[years][0] += df_temp['Income'][i]
            years_ML_dict[years][1] += 1
    years_ML_dict[years][2] = years_ML_dict[years][0] / years_ML_dict[years][1]

# Put dict into data frame
ret = pd.DataFrame.from_dict(years_ML_dict)
ret = ret.iloc[2:, :]
ret = ret.T
ret.columns = ['Mean Income Level']

# Plot dataframe into bar chart
ax = ret.plot.barh(title='Mean Income Level by Years Using Machine Learning')
ax.set_ylabel('Years Using ML')


# # Clustering

# In[238]:


def cluster(isLow, column, clusters, title, xlabel):
    def get_groups(column):
        grouped = df[[column,'High Income']].groupby('High Income')
        low = pd.DataFrame(grouped.get_group(0), columns = [column,'High Income'])
        high = pd.DataFrame(grouped.get_group(1), columns = [column,'High Income'])
        return low, high
    
    low, high = get_groups(column)
    line = low if isLow else high
    
    X_line = pd.get_dummies(line)  
    km = KMeans(n_clusters = clusters)  
    y_line = km.fit_predict(X_line)
    line['Cluster'] = y_line

    churn_crosstab = pd.crosstab(line['Cluster'], line[column], normalize=True)
    churn_crosstab.plot(kind = 'bar', title = title)
    plt.xlabel(xlabel)
    plt.ylabel('Proportion Of Respondants')
    plt.legend(bbox_to_anchor=(1.0, 0.5))
    plt.show()


# ### By country

# In[239]:


cluster(True, 'Top Country', 3, 'Proportion of respondants against clusters of low income in countries', 'Top Countries')


# In[240]:


cluster(False, 'Top Country', 3, 'Proportion of respondants against clusters of high income in countries', 'Top Countries')


# ### By years coding

# In[241]:


cluster(True, 'Years Coding', 2, 'Proportion of respondants against clusters of low income in years coding', 'Years Coding')


# In[242]:


cluster(False, 'Years Coding', 4, 'Proportion of respondants against clusters of high income in years coding', 'Years Coding')


# 

# ### By years using ML

# In[243]:


cluster(True, 'Years Using ML', 2, 'Proportion of respondants against clusters of low income in years using machine learning', 'Years Using ML')


# In[244]:


cluster(False, 'Years Using ML', 3, 'Proportion of respondants against clusters of high income in years using machine learning', 'Years Using ML')


# # Machine learning

# ### Rest of encoding

# In[245]:


df


# In[246]:


def encode_data(field):
    global df
    encoder = OneHotEncoder()
    X_fields = encoder.fit_transform(df[field].values.reshape(-1, 1)).toarray()
    df_Encode = pd.DataFrame(X_fields)
    df_Encode.columns = encoder.get_feature_names([field])
    df.drop(field, inplace=True, axis=1)
    df = pd.concat([df, df_Encode], axis=1).reindex(df.index)
encode_data('Gender')
encode_data('Country')


# In[247]:


df


# In[248]:


X = df
X = X.drop(columns = ['High Income'])
y = df.iloc[:, 8] 
print("Features X\n",X)
print("Target y\n", y)

