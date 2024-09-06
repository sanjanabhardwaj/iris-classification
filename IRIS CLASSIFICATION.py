#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib inline displays matplotlib plots directly within Jupyter notebooks,


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[7]:


df  = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\Iris.csv")


# In[5]:


df.head()


# In[6]:


df.describe()


# In[9]:


df.shape


# In[36]:


df.columns


# In[38]:


#for finding the data type 
df.dtypes


# In[39]:


#for checking the unique data
df.nunique


# In[35]:


print(df['Species'].value_counts())


# In[12]:


#checking for the missing value 
df.isnull().sum()


# In[13]:


print(df['Species'].value_counts().reset_index())


# # plotting graph 

# In[14]:


df.head(3)


# In[26]:


plt.figure(figsize=(10, 6))
plt.bar(df['Species'], df['SepalLengthCm'])
plt.xlabel('Species')
plt.ylabel('SepalLengthCm')
plt.title('Species by length')
plt.xticks(rotation = 0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[10]:


#couting the gender
Species_counts = df['Species'].value_counts()
#ploting the pie charts 
plt.figure(figsize =(7,5))
plt.title("PIE CHART FOR Species")
plt.pie(Species_counts, labels = Species_counts.index,autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#FFD700'], startangle=90)
#plt.legend(Species_counts.index, title="Species")
plt.show()


# # relation between PetalLengthCm  and Species
# 

# In[8]:


plt.figure(figsize=(6,7))
sns.boxplot(x='PetalLengthCm', y = 'Species',data = df.sort_values('PetalLengthCm'))
plt.show()


# In[14]:


#relation between the width and length of flower
plt.figure(figsize = (6,6))
df.plot(kind='scatter',x='SepalWidthCm',y='SepalLengthCm')
#legend()
plt.show()


# In[13]:


# Plot scatter plot for Sepal Width vs. Sepal Length
plt.figure(figsize=(6, 6))
plt.scatter(df['SepalWidthCm'], df['SepalLengthCm'], label='Species')

# Adding legend and titles
plt.legend(title="Flower Dimensions")
plt.xlabel('SepalWidthCm')
plt.ylabel('SepalLengthCm')
plt.title('Relation between Sepal Width and Sepal Length of Iris Flowers')

# Display the plot
plt.show()


# In[18]:


#Scatter plot of different species
sns.FacetGrid(df, hue ='Species', height = 4).map(plt.scatter,"PetalLengthCm","SepalWidthCm").add_legend()
plt.show()


# In[19]:


#Scatter plot of different species
sns.FacetGrid(df, hue ='Species', height = 4).map(plt.scatter,"PetalWidthCm","SepalLengthCm").add_legend()
plt.show()


# In[20]:


#Scatter plot of different species
sns.FacetGrid(df, hue ='Species', height = 4).map(plt.scatter,"SepalWidthCm","SepalLengthCm").add_legend()
plt.show()


# In[23]:


import plotly.express as px


# In[26]:


fig = px.scatter_3d(df, x='SepalLengthCm', y='PetalWidthCm', z='PetalLengthCm', color='Species')
fig.show()


# In[27]:


# Set the figure size
plt.figure(figsize=(12, 8))

# Histogram for Sepal Length
plt.subplot(221)
plt.hist(df['SepalLengthCm'], color='#1f77b4')  # Using a consistent blue color
plt.xlabel('Sepal Length in Cm')
plt.ylabel('Frequency')
plt.title('Histogram for Sepal Length')
plt.grid(True)

# Histogram for Sepal Width
plt.subplot(222)
plt.hist(df['SepalWidthCm'], color='#ff7f0e')  # Using a consistent red color
plt.xlabel('Sepal Width in Cm')
plt.ylabel('Frequency')
plt.title('Histogram for Sepal Width')
plt.grid(True)

# Histogram for Petal Length
plt.subplot(223)
plt.hist(df['PetalLengthCm'], color='#ff7f0e')  # Consistent color for Petal Length
plt.xlabel('Petal Length in Cm')
plt.ylabel('Frequency')
plt.title('Histogram for Petal Length')
plt.grid(True)

# Histogram for Petal Width
plt.subplot(224)
plt.hist(df['PetalWidthCm'], color='#1f77b4')  # Consistent color for Petal Width
plt.xlabel('Petal Width in Cm')
plt.ylabel('Frequency')
plt.title('Histogram for Petal Width')
plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()


# In[31]:


sns.pairplot(df.drop(columns = ['PetalWidthCm']), hue = "Species", size = 2)


# # co relation matrix

# In[33]:


corr = df.corr()
plt.figure(figsize=(8,6))

sns.heatmap(
    corr,                # Data to plot
    annot=True,          # Display the data values in each cell
    fmt='.2f',          # Format of the annotations
    cmap='YlGnBu',      # Color map (YlGnBu is a gradient from yellow to blue)
    center=0,           # Center the colormap at zero
    linewidths=0.5,    # Width of the lines that will divide each cell
    linecolor='black',  # Color of the lines that divide each cell
    cbar=True,          # Display the color bar
    cbar_kws={"shrink": .8},  # Customize color bar size
    square=True,        # Make the heatmap squares instead of rectangles
    xticklabels=True,   # Show x-tick labels
    yticklabels=True,   # Show y-tick labels
    vmin=-1,            # Minimum value for colormap scaling
    vmax=1              # Maximum value for colormap scaling
)

plt.title('Correlation Heatmap', fontsize=10)#title for heatmap
plt.show()


# In[41]:


sns.lineplot(x = df['SepalLengthCm'] , y = df['PetalLengthCm'], color ='green')
plt.show()


# In[47]:


plt.plot(df['SepalLengthCm'])
plt.plot(df['PetalLengthCm'])
plt.plot(df['SepalWidthCm'])
plt.plot(df['PetalWidthCm'])
#plt.legend('SepalLengthCm','PetalLengthCm','SepalWidthCm','PetalWidthCm')
plt.show()


# In[49]:


sns.lmplot( x="SepalLengthCm", y="SepalWidthCm", data=df, hue='Species', legend=False)
plt.show()


# # bar plot for petal length , petal width , sepal length and sepal width 

# In[56]:


fig,axes=plt.subplots(1,4,figsize=(20,3))
df['SepalLengthCm'].hist(ax=axes[0],color="r").set_title("sepal length in cm ")
df['PetalLengthCm'].hist(ax=axes[1],color="b").set_title("petal length in cm")
df['SepalWidthCm'].hist(ax=axes[2],color="g").set_title("sepal width in cm")
df['PetalWidthCm'].hist(ax=axes[3],color="m").set_title("petal width in cm ")
plt.show()


# In[63]:


df.hist()


# cleaning the duplicate values 

# In[58]:


df[df.duplicated()]


# counting the duplicate values 

# In[61]:


df.duplicated().value_counts()


# In[62]:


#checking the missing values 
df.isnull().sum()


# In[64]:


#distribution based on species

sns.countplot(x='Species', data= df)


# In[ ]:





# In[ ]:




