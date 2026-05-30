# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# %%
# Data Loading
bitcoin_data = pd.read_csv("../dataset/bitcoin.csv")

# Initial Data Review
print("Data Head:")
print(bitcoin_data.head())

print("\nData Description:")
print(bitcoin_data.describe())

print("\nMissing Values:")
print(bitcoin_data.isnull().sum())

# %%
# Close Price Trend
plt.close()
plt.figure(figsize=(10, 6))
plt.plot(bitcoin_data['Close'], color='orange')
plt.title('Bitcoin Close Price')
plt.xlabel('Time')
plt.ylabel('Price in Dollars')
plt.grid(alpha=0.3)
plt.show()

# %%
# Close Price Distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
sns.histplot(bitcoin_data['Close'], kde=True, ax=axes[0, 0], color='teal')
axes[0, 0].set_title('Bitcoin Price Distribution')
axes[0, 1].axis('off')
axes[1, 0].axis('off')
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()

# %%
# Date Feature Extraction
date_parts = bitcoin_data['Date'].str.split('-', expand=True)
bitcoin_data['year'] = date_parts[0].astype(int)
bitcoin_data['month'] = date_parts[1].astype(int)
bitcoin_data['day'] = date_parts[2].astype(int)

# %%
# Yearly Average Close Price
plt.close()
yearly_grouped_data = bitcoin_data.groupby('year').mean(numeric_only=True)
plt.figure(figsize=(8, 5))
yearly_grouped_data['Close'].plot(kind='bar', color='purple')
plt.title('Average Bitcoin Close Price by Year')
plt.xlabel('Year')
plt.ylabel('Average Close Price')
plt.tight_layout()
plt.show()

# %%
# Feature Engineering
bitcoin_data['is_quarter_end'] = np.where(bitcoin_data['month'] % 3 == 0, 1, 0)
bitcoin_data['open_close_diff'] = bitcoin_data['Open'] - bitcoin_data['Close']
bitcoin_data['high_low_diff'] = bitcoin_data['High'] - bitcoin_data['Low']
bitcoin_data['target'] = np.where(bitcoin_data['Close'].shift(-1) > bitcoin_data['Close'], 1, 0)

# %%
# Target Distribution
plt.close()
plt.pie(
    bitcoin_data['target'].value_counts().values,
    labels=[0, 1],
    autopct='%1.1f%%',
    colors=['lightcoral', 'skyblue']
)
plt.title('Target Distribution')
plt.show()

# %%
# Feature Selection
model_features = bitcoin_data[['open_close_diff', 'high_low_diff', 'is_quarter_end']]
model_target = bitcoin_data['target']

# Feature Scaling
feature_scaler = StandardScaler()
scaled_features = feature_scaler.fit_transform(model_features)

# %%
# Train Validation Split
x_train, x_valid, y_train, y_valid = train_test_split(
    scaled_features,
    model_target,
    test_size=0.15,
    random_state=2022
)

print("Train Shape:", x_train.shape)
print("Validation Shape:", x_valid.shape)

# %%
# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(x_train, y_train)

validation_predictions = logistic_model.predict(x_valid)
validation_auc = metrics.roc_auc_score(y_valid, logistic_model.predict_proba(x_valid)[:, 1])
print("\nLogistic Regression Validation ROC AUC:", validation_auc)

# %%
# Model Comparison
comparison_models = [
    LogisticRegression(max_iter=10000),
    KNeighborsClassifier(),
    DecisionTreeClassifier()
]

for current_model in comparison_models:
    current_model.fit(x_train, y_train)

    print(f"\n{type(current_model).__name__}:")
    train_auc = metrics.roc_auc_score(y_train, current_model.predict_proba(x_train)[:, 1])
    valid_auc = metrics.roc_auc_score(y_valid, current_model.predict_proba(x_valid)[:, 1])

    print("Training ROC AUC:", train_auc)
    print("Validation ROC AUC:", valid_auc)


