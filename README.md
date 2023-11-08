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
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex-07/assets/121418522/34e84532-21ba-406f-b6ae-43e468ba7fae)

#### Null values
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex-07/assets/121418522/dc89eb9f-733d-44a3-8620-bbce3da68f17)

#### Describing the data
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex-07/assets/121418522/3758004d-5608-4ac4-a0c5-6ffa36f6afa2)

#### Missing values
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex-07/assets/121418522/0628fd80-58ca-4fd5-9169-cb53b50f6a58)

#### Data after cleaning
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex-07/assets/121418522/2f2da446-94b4-42f3-966b-edad86c7e89a)

#### Data on Heatmap
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex-07/assets/121418522/23f125c4-66e2-4bfb-87ec-fa2df989f238)

#### Report of(people survied & died)
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex-07/assets/121418522/6785d0e3-41b2-498f-8723-28b0795c2003)

#### Cleaned null values
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex-07/assets/121418522/2729d6f8-d07e-4d60-ad12-ae64fa94b17a)

#### Report of survied people's age
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex-07/assets/121418522/41429727-cc7a-4e53-a198-6b28e14d1c13)

#### Report of pessengers
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex-07/assets/121418522/784a3ead-2996-48ee-a1f4-9d9ee6b3e32b)

#### Report
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex-07/assets/121418522/56c8d8a9-8bdb-4190-afd5-7fe442241d56)

## RESULT:
Thus, Sucessfully performed the various feature selection techniques on a given dataset.
