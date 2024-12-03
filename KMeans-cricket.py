#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("Cricket.csv",encoding = "latin-1")


# In[4]:


print(df.shape)


# In[5]:


#EDA:

df.info()


# In[9]:


df[["Start","End"]] = df["Span"].str.split("-",expand=True)


# In[7]:


df


# In[10]:


df.info()


# In[11]:


df["Start"] = df["Start"].astype(int)
df["End"] = df["End"].astype(int)


# In[12]:


df.info()


# In[13]:


df["Exp"] = df["End"] - df["Start"]


# In[14]:


df


# In[15]:


df = df.drop(["Span","Start","End"],axis=1)


# In[16]:


df


# In[17]:


df[["HS","Extra"]] = df["HS"].str.split("*",expand=True)


# In[18]:


df


# In[19]:


df = df.drop(["Extra"],axis=1)


# In[20]:


df


# In[21]:


df.info()


# In[22]:


df["HS"] = df["HS"].astype(int)


# In[23]:


df.info()


# In[24]:


df.isnull().sum()


# In[25]:


df.duplicated().sum()


# In[26]:


#VISUALIZATION


# In[31]:


plt.figure(figsize = (30,5))
ax = sns.barplot(x='Player', y='Mat', data=df)
ax.set(xlabel = '', ylabel= 'Match Played')
plt.xticks(rotation=90)
plt.show()


# In[33]:


plt.figure(figsize = (30,5))
ax = sns.barplot(x='Player', y='100', data=df)
ax.set(xlabel = '', ylabel= 'Centuries')
plt.xticks(rotation=90)
plt.show()


# In[34]:


plt.figure(figsize = (30,5))
ax = sns.barplot(x='Player', y='50', data=df)
ax.set(xlabel = '', ylabel= 'Half-Centuries')
plt.xticks(rotation=90)
plt.show()


# In[35]:


df


# In[36]:


# outlier detection


# In[39]:


sns.violinplot(df["Mat"])


# In[40]:


f, axes = plt.subplots(4,3, figsize=(20, 10))
s=sns.violinplot(y=df.Exp,ax=axes[0, 0])
axes[0, 0].set_title('Exp')
s=sns.violinplot(y=df.Mat,ax=axes[0, 1])
axes[0, 1].set_title('Mat')
s=sns.violinplot(y=df.Inns,ax=axes[0, 2])
axes[0, 2].set_title('Inns')

s=sns.violinplot(y=df.NO,ax=axes[1, 0])
axes[1, 0].set_title('NO')
s=sns.violinplot(y=df.Runs,ax=axes[1, 1])
axes[1, 1].set_title('Runs')
s=sns.violinplot(y=df.HS,ax=axes[1, 2])
axes[1, 2].set_title('HS')

s=sns.violinplot(y=df.Ave,ax=axes[2, 0])
axes[2, 0].set_title('Ave')
s=sns.violinplot(y=df.SR,ax=axes[2, 1])
axes[2, 1].set_title('SR')
s=sns.violinplot(y=df['100'],ax=axes[2, 2])
axes[2, 2].set_title('100')
s=sns.violinplot(y=df.BF,ax=axes[3, 0])
axes[3, 0].set_title('BF')
s=sns.violinplot(y=df['50'],ax=axes[3, 1])
axes[3, 1].set_title('50s')
s=sns.violinplot(y=df['0'],ax=axes[3, 2])
axes[3, 2].set_title('0s')
plt.show()


# In[41]:


import warnings
warnings.filterwarnings("ignore")


# In[47]:


Q3 = df.Mat.quantile(0.99)
Q1 = df.Mat.quantile(0.01)
df['Mat'][df['Mat']<=Q1]=Q1
df['Mat'][df['Mat']>=Q3]=Q3

Q1 = df.Ave.quantile(0.01)
df['Ave'][df['Ave']<=Q1]=Q1
df['Ave'][df['Ave']>=Q3]=Q3

Q3 = df.BF.quantile(0.99)
Q1 = df.BF.quantile(0.01)
df['BF'][df['BF']<=Q1]=Q1
df['BF'][df['BF']>=Q3]=Q3

Q3 = df.SR.quantile(0.99)
Q1 = df.SR.quantile(0.01)
df['SR'][df['SR']<=Q1]=Q1
df['SR'][df['SR']>=Q3]=Q3

Q3 = df.Exp.quantile(0.99)
Q1 = df.Exp.quantile(0.01)
df['Exp'][df['Exp']<=Q1]=Q1
df['Exp'][df['Exp']>=Q3]=Q3

Q3 = df['100'].quantile(0.99)
Q1 = df['100'].quantile(0.01)
df['100'][df['100']<=Q1]=Q1
df['100'][df['100']>=Q3]=Q3

Q3 = df['50'].quantile(0.99)
Q1 = df['50'].quantile(0.01)
df['50'][df['50']<=Q1]=Q1
df['50'][df['50']>=Q3]=Q3

Q3 = df['0'].quantile(0.99)
Q1 = df['0'].quantile(0.01)
df['0'][df['0']<=Q1]=Q1
df['0'][df['0']>=Q3]=Q3


# In[48]:


f, axes = plt.subplots(4,3, figsize=(20, 10))
s=sns.violinplot(y=df.Exp,ax=axes[0, 0])
axes[0, 0].set_title('Exp')
s=sns.violinplot(y=df.Mat,ax=axes[0, 1])
axes[0, 1].set_title('Mat')
s=sns.violinplot(y=df.Inns,ax=axes[0, 2])
axes[0, 2].set_title('Inns')

s=sns.violinplot(y=df.NO,ax=axes[1, 0])
axes[1, 0].set_title('NO')
s=sns.violinplot(y=df.Runs,ax=axes[1, 1])
axes[1, 1].set_title('Runs')
s=sns.violinplot(y=df.HS,ax=axes[1, 2])
axes[1, 2].set_title('HS')

s=sns.violinplot(y=df.Ave,ax=axes[2, 0])
axes[2, 0].set_title('Ave')
s=sns.violinplot(y=df.SR,ax=axes[2, 1])
axes[2, 1].set_title('SR')
s=sns.violinplot(y=df['100'],ax=axes[2, 2])
axes[2, 2].set_title('100')
s=sns.violinplot(y=df.BF,ax=axes[3, 0])
axes[3, 0].set_title('BF')
s=sns.violinplot(y=df['50'],ax=axes[3, 1])
axes[3, 1].set_title('50s')
s=sns.violinplot(y=df['0'],ax=axes[3, 2])
axes[3, 2].set_title('0s')
plt.show()


# In[51]:


#making a copy of original data
df_new = df.copy()


# In[52]:


df_new


# In[53]:


df_new = df_new.drop(["Player"],axis=1)


# In[54]:


df_new


# In[55]:


# standardization


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_new)
df_scaled


# In[56]:


df_df1 = pd.DataFrame(df_scaled,columns=[ 'Mat', 'Inns', 'NO', 'Runs', 'HS', 'Ave', 'BF', 'SR', '100',
                                            '50', '0', 'Exp'])
df_df1


# In[57]:


# start Kmeans

from sklearn.cluster import KMeans


# In[58]:


# WCSS- within the cluster sum squares
# SSD- sum squared distances
# inertia


# In[59]:


clusters=list(range(2,8))
ssd = []
for num_clusters in clusters:
    model_clus = KMeans(n_clusters = num_clusters, max_iter=150,random_state= 50)
    model_clus.fit(df_df1)
    ssd.append(model_clus.inertia_)


# In[60]:


ssd


# In[61]:


plt.plot(clusters,ssd)


# In[62]:


kmodel = KMeans(n_clusters = 4, max_iter=150,random_state= 50)
kmodel.fit(df_df1)


# In[63]:


kmodel.labels_


# In[65]:


df["ClusterId"] = kmodel.labels_


# In[66]:


df


# In[67]:


df[df["ClusterId"]==0]


# In[68]:


df[df["ClusterId"]==1]


# In[69]:


df[df["ClusterId"]==2]


# In[70]:


df[df["ClusterId"]==3]


# In[72]:


df.ClusterId.value_counts(ascending=True)


# In[ ]:




