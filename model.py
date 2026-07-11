
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)




df = pd.read_csv("data.csv")   



print("First 5 Rows:")
print(df.head())


print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())



print("\nMissing Values:")
print(df.isnull().sum())

 

if "Unnamed: 32" in df.columns:
    df.drop("Unnamed: 32", axis=1, inplace=True)

if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)



print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)


encoder = LabelEncoder()
df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

plt.figure(figsize=(5,5))
sns.countplot(x='diagnosis', data=df)
plt.title("Diagnosis Distribution")
plt.xticks([0,1],["Benign","Malignant"])
plt.show()


plt.figure(figsize=(15,12))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


df.hist(figsize=(18,15))
plt.show()



X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)



model = SVC(kernel='linear')

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)



print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=["Benign","Malignant"],
            yticklabels=["Benign","Malignant"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



sample = X_test[0].reshape(1,-1)

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nPrediction: Malignant")
else:
    print("\nPrediction: Benign")
