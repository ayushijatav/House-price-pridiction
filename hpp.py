import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r'C:\Users\91846\OneDrive\b\data\MagicBricks (1).csv')
#print(df.head())
df


# In[4]:


df2=df.copy()


# In[5]:


print(df2.isnull().sum())


# In[6]:


df2.info()


# In[7]:


df2['Area'] = df2['Area'].fillna(df2['Area'].mean())
df2['BHK'] = df2['BHK'].fillna(df2['BHK'].mean())
df2['Bathroom'] = df2['Bathroom'].fillna(df2['Bathroom'].mean())
df2['Per_Sqft'] = df2['Per_Sqft'].fillna(df2['Per_Sqft'].mean())

# Fill NaN values with the mode for categorical columns
df2['Furnishing'] = df2['Furnishing'].fillna(df2['Furnishing'].mode()[0])
df2['Locality'] = df2['Locality'].fillna(df2['Locality'].mode()[0])
df2['Parking'] = df2['Parking'].fillna(df2['Parking'].mode()[0])
df2['Price'] = df2['Price'].fillna(df2['Price'].mode()[0])
df2['Status'] = df2['Status'].fillna(df2['Status'].mode()[0])
df2['Transaction'] = df2['Transaction'].fillna(df2['Transaction'].mode()[0])
df2['Type'] = df2['Type'].fillna(df2['Type'].mode()[0])


# In[8]:


df2.shape


# In[9]:


df2.isnull().sum()


# In[10]:


for i in ("Area","BHK","Bathroom","Parking","Price","Per_Sqft"):
    q1 = df2[i].quantile(0.25)
    q3= df2[i].quantile(0.75)
    IQR = q3 - q1
    lb = q1 - 1.5 * IQR
    ub = q3 + 1.5 * IQR
    df2 = df2[(df2[i] > lb) & (df2[i] < ub)]


# In[11]:


df2.shape


# In[12]:


print(df2.shape)
df2.columns


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

# Assuming your data is in a pandas DataFrame named 'data'
X = df2.drop(['Price'],axis=1)
y = df2['Price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for different types of columns
numeric_features = ['Area', 'BHK', 'Bathroom', 'Parking', 'Per_Sqft']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['Furnishing', 'Locality', 'Status', 'Transaction', 'Type']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessing and the linear regression model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Train the model
model.fit(X_train, y_train)
    


# In[14]:


predictions=model.predict(X_test)
predictions


# In[15]:


train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R^2 score: {train_score}")
print(f"Test R^2 score: {test_score}")


# In[16]:


import pickle
with open('pipelinemodel.pkl','wb') as file:
    pickle.dump(model,file)


# In[ ]:





# In[ ]:





# In[ ]:

