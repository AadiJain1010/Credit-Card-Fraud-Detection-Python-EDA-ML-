#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd        # for working with data tables
import matplotlib.pyplot as plt   # for making charts
import seaborn as sns      # for prettier charts



# In[3]:


df = pd.read_csv('creditcard.csv')   # load the dataset



# In[ ]:





# In[4]:


df.head()


# In[5]:


df.shape



# In[6]:


df.info()


# In[7]:


df['Class'].value_counts()


# In[8]:


fraud_percent = (df['Class'].value_counts(normalize=True) * 100)
print(fraud_percent)


# In[9]:


# Draw a bar chart to see how many normal (0) vs fraud (1) transactions we have
sns.countplot(x='Class', data=df)

# Add a title to the chart
plt.title('Class Distribution')

# Display the chart
plt.show()


# In[10]:


# Normal transactions Amount stats
print("Normal transactions Amount stats:")
print(df[df['Class'] == 0]['Amount'].describe())

# Fraud transactions Amount stats
print("\nFraud transactions Amount stats:")
print(df[df['Class'] == 1]['Amount'].describe())

# Boxplot to compare visually
plt.figure(figsize=(10, 5))
sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Transaction Amount by Class')
plt.show()


# In[11]:


plt.figure(figsize=(10, 5))
sns.boxplot(x='Class', y='Amount', data=df)
plt.yscale('log')  # log scale for y-axis
plt.title('Transaction Amount by Class (Log Scale)')
plt.show()


# In[12]:


plt.figure(figsize=(12, 9))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:





# In[13]:


sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')


# In[14]:


plt.figure(figsize=(12, 9))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[15]:


plt.figure(figsize=(10,5))

# Plot for normal transactions
sns.kdeplot(df[df['Class'] == 0]['Amount'], label='Normal', fill=True)

# Plot for fraud transactions
sns.kdeplot(df[df['Class'] == 1]['Amount'], label='Fraud', fill=True, color='red')

plt.title('Distribution of Transaction Amount')
plt.xlabel('Transaction Amount')
plt.ylabel('Density')
plt.legend()
plt.show()


# In[16]:


plt.figure(figsize=(10,5))

# Plot normal transactions
sns.histplot(df[df['Class'] == 0]['Time'] / 3600, bins=100, color='blue', label='Normal', alpha=0.6)

# Plot fraud transactions
sns.histplot(df[df['Class'] == 1]['Time'] / 3600, bins=100, color='red', label='Fraud', alpha=0.6)

plt.title('Transactions Over Time (in hours)')
plt.xlabel('Time (hours)')
plt.ylabel('Number of Transactions')
plt.legend()
plt.show()


# In[17]:


plt.figure(figsize=(10,5))
sns.histplot(df[df['Class'] == 1]['Time'] / 3600, bins=100, color='red')
plt.title('Fraud Transactions Over Time (in hours)')
plt.xlabel('Time (hours)')
plt.ylabel('Number of Fraud Transactions')
plt.show()


# In[18]:


plt.figure(figsize=(10,5))
sns.histplot(df[df['Class'] == 0]['Time'] / 3600, bins=100, color='blue', label='Normal', alpha=0.6, stat='density')
sns.histplot(df[df['Class'] == 1]['Time'] / 3600, bins=100, color='red', label='Fraud', alpha=0.6, stat='density')
plt.title('Proportion of Transactions Over Time (in hours)')
plt.xlabel('Time (hours)')
plt.ylabel('Density')
plt.legend()
plt.show()


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, cmap="coolwarm_r", annot_kws={'size':20})
plt.title("Correlation Matrix Heatmap")
plt.show()


# In[20]:


# Extract correlation of all features with Class
corr_with_class = corr_matrix['Class'].drop('Class')

# Sort by correlation strength
corr_with_class_sorted = corr_with_class.sort_values()

# Plot as horizontal bar chart
plt.figure(figsize=(10,6))
corr_with_class_sorted.plot(kind='barh', color='teal')
plt.title('Feature Correlation with Class (Fraud)')
plt.xlabel('Correlation Coefficient')
plt.show()


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt

# List of features to plot
features = ['V14', 'V17', 'V12']

# Plot each feature
for feature in features:
    plt.figure(figsize=(10,5))
    sns.kdeplot(df[feature][df['Class'] == 0], label='Normal', fill=True, color='blue', alpha=0.5)
    sns.kdeplot(df[feature][df['Class'] == 1], label='Fraud', fill=True, color='red', alpha=0.5)
    plt.title(f'Distribution of {feature} by Class')
    plt.legend()
    plt.show()


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

features = ['V17', 'V14', 'V12']

for feature in features:
    plt.figure(figsize=(10,5))
    sns.boxplot(
        x='Class', 
        y=feature, 
        data=df, 
        palette={0: 'blue', 1: 'red'}
    )
    plt.title(f'Boxplot of {feature} by Class')
    plt.xlabel('Transaction Class (0=Normal, 1=Fraud)')
    plt.ylabel(f'{feature} Value')
    plt.show()


# In[23]:


from sklearn.preprocessing import StandardScaler

# Create scaler object
scaler = StandardScaler()

# Scale 'Amount' column
df['Amount_Scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))

# Scale 'Time' column
df['Time_Scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))

# Check the result
df[['Amount', 'Amount_Scaled', 'Time', 'Time_Scaled']].head()


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X = df.drop(['Amount', 'Time', 'Class'], axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,       
    random_state=42,     
    stratify=y           
)
print(X_train.shape)
print(X_test.shape)
print(y_train.value_counts())
print(y_test.value_counts())


# In[26]:


# 1️⃣ Import the model
from sklearn.linear_model import LogisticRegression

# 2️⃣ Initialize the model
model = LogisticRegression(max_iter=1000, random_state=42)

# 3️⃣ Train the model on training data
model.fit(X_train, y_train)

# 4️⃣ Predict on test data
y_pred = model.predict(X_test)

# 5️⃣ Evaluate model performance
from sklearn.metrics import classification_report, confusion_matrix

# Show confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Show detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))


# In[27]:


# 1️⃣ Import the classifier
from sklearn.ensemble import RandomForestClassifier

# 2️⃣ Create the model
model_rf = RandomForestClassifier(
    n_estimators=100,     # Number of trees
    random_state=42,      # For reproducibility
    class_weight='balanced'  # 🔑 Helps with imbalanced data
)

# 3️⃣ Train the model
model_rf.fit(X_train, y_train)

# 4️⃣ Make predictions
y_pred_rf = model_rf.predict(X_test)

# 5️⃣ Evaluate
from sklearn.metrics import confusion_matrix, classification_report

print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred_rf))


# In[28]:


# Get probabilities of class 1 (fraud)
y_proba_rf = model_rf.predict_proba(X_test)[:, 1]

# Try a custom threshold, e.g. 0.3
threshold = 0.3
y_pred_custom = (y_proba_rf >= threshold).astype(int)

# Evaluate
from sklearn.metrics import confusion_matrix, classification_report

print("✅ Confusion Matrix with threshold =", threshold)
print(confusion_matrix(y_test, y_pred_custom))

print("\n✅ Classification Report with threshold =", threshold)
print(classification_report(y_test, y_pred_custom))


# In[30]:


from sklearn.ensemble import RandomForestClassifier

# New model with more trees
model_rf = RandomForestClassifier(
    n_estimators=300,        # Increased from 100 to 300
    random_state=42,         # For reproducibility
    class_weight='balanced'  # To handle imbalance (if you were using this)
)

# Fit (train) the model again
model_rf.fit(X_train, y_train)


# In[31]:


# Get new probabilities
y_proba_rf = model_rf.predict_proba(X_test)[:, 1]

# Apply your threshold
threshold = 0.3
y_pred_custom = (y_proba_rf >= threshold).astype(int)

# Evaluate
from sklearn.metrics import confusion_matrix, classification_report

print("✅ Confusion Matrix with threshold =", threshold)
print(confusion_matrix(y_test, y_pred_custom))

print("\n✅ Classification Report with threshold =", threshold)
print(classification_report(y_test, y_pred_custom))


# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances from trained model_rf
importances = model_rf.feature_importances_
features = X_train.columns

# Put in a DataFrame
feat_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top 10 important features
plt.figure(figsize=(10,6))
sns.barplot(data=feat_importance_df.head(10), x='Importance', y='Feature', palette='viridis')
plt.title('Top 10 Feature Importances from Random Forest')
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [100, 300],       # Number of trees
    'max_depth': [None, 10, 15],      # Max depth of trees
    'min_samples_split': [2, 5],      # Min samples to split an internal node
    'min_samples_leaf': [1, 3]        # Min samples at a leaf node
}

# Initialize Random Forest
rf_base = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'  # Still handle imbalance
)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    scoring='f1',    # Because we care about balance between precision/recall on fraud
    cv=3,            # 3-fold cross-validation
    n_jobs=-1        # Use all CPU cores for speed
)

# Run grid search
grid_search.fit(X_train, y_train)

# Best parameters
print("✅ Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Evaluate on test set
y_pred_grid = grid_search.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print("\n✅ Confusion Matrix (Tuned RF):")
print(confusion_matrix(y_test, y_pred_grid))

print("\n✅ Classification Report (Tuned RF):")
print(classification_report(y_test, y_pred_grid))


# In[ ]:




