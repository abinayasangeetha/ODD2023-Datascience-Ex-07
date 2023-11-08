# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

## ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

## CODE
#### DEVELOPED BY: ABINAYA S
#### Register no:212222230002

#### Importing library
```
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
```
#### Data loading
```
data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()
```
#### Now, we are checking start with a pairplot, and check for missing values
```
sns.heatmap(data.isnull(),cbar=False)
```
#### Data Cleaning and Data Drop Process
```
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())
```
#### Change to categoric column to numeric
```
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1
```
#### Instead of nan values
```
data['Embarked']=data['Embarked'].fillna('S')
```
#### Change to categoric column to numeric
```
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2
```
#### Drop unnecessary columns
```
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)
data.head(11)
```
##### Heatmap for train dataset
```
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
```
#### Now, data is clean and read to a analyze
```
sns.heatmap(data.isnull(),cbar=False)
```
#### How many people survived or not... %60 percent died %40 percent survived
```
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()
```
#### Age with survived
```
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()
```
#### Count the pessenger class
```
fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
```
#### Split the data into training and test sets
```
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
```
#### Create a Random Forest classifier
```
my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5, criterion='entropy')
```
#### Fit the model to the training data
```
my_forest.fit(X_train, y_train)
```
#### Make predictions on the test data
```
target_predict = my_forest.predict(X_test)
```
#### Evaluate the model's performance
```
accuracy = accuracy_score(y_test, target_predict)
mse = mean_squared_error(y_test, target_predict)
r2 = r2_score(y_test, target_predict)

print("Random forest accuracy: ", accuracy)
print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2) Score: ", r2)
```
# OUPUT

#### Initial data
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex-07/assets/119393675/8109ee26-6402-4d89-80f6-4619e44b76f9)

#### Null values
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex-07/assets/119393675/30de508f-eb67-49fd-a69d-0768e0cc7cd4)


#### Describing the data
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex-07/assets/119393675/c46a5bca-6623-4d08-b6c8-aa8928c5cf43)

#### Missing values
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex-07/assets/119393675/8005fbbd-4d9d-4c39-8837-e2688909cfde)

#### Data after cleaning
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex-07/assets/119393675/435f3017-24ed-4f9d-bd8f-606af2e19d3c)

#### Data on Heatmap
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex-07/assets/119393675/92df7374-388a-407d-9eab-f622a0960c52)

#### Report of(people survied & died)
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex-07/assets/119393675/ff6f659f-cf65-4c0b-be97-87222e27f64f)

#### Cleaned null values
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex-07/assets/119393675/0f93096e-17d9-4e4c-910c-c82a4c783b68)

#### Report of survied people's age
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex-07/assets/119393675/02a17016-4a7f-486f-a707-b91757eedb7f)


#### Report of pessengers
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex-07/assets/119393675/f1cdeef8-4add-4eed-83a5-bf46b7b8fedd)


#### Report
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex-07/assets/119393675/6857031e-38e1-494a-a039-3859dd68a623)

## RESULT:
Thus, Sucessfully performed the various feature selection techniques on a given dataset.
