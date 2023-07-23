#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import sys

# In[2]:

file = sys.argv[1]
file1 = sys.argv[2]
file = pd.read_csv(file)
file1 = pd.read_csv(file1)
file.to_csv('wine-training.csv')
file1.to_csv('wine-test.csv')


# In[3]:


wineTrain_DF = pd.read_csv('wine-training.csv')
wineTest_DF = pd.read_csv('wine-test.csv')


# In[4]:


result = pd.DataFrame('Alcohol Malic_acid Ash Alcalinity_of_ash Magnesium Total_phenols Flavanoids Nonflavanoid_phenols Proanthocyanins Color_intensity Hue OD280%2FOD315_of_diluted_wines Proline Class'.split(" "))
result2 = pd.DataFrame('Alcohol Malic_acid Ash Alcalinity_of_ash Magnesium Total_phenols Flavanoids Nonflavanoid_phenols Proanthocyanins Color_intensity Hue OD280%2FOD315_of_diluted_wines Proline Class'.split(" "))
for i in range(len(wineTest_DF)):
    data = pd.DataFrame(wineTest_DF['Alcohol Malic_acid Ash Alcalinity_of_ash Magnesium Total_phenols Flavanoids Nonflavanoid_phenols Proanthocyanins Color_intensity Hue OD280%2FOD315_of_diluted_wines Proline Class'][i].split(" "))
    result = pd.concat([result, data], axis=1)
for i in range(len(wineTest_DF)):
    data = pd.DataFrame(wineTrain_DF['Alcohol Malic_acid Ash Alcalinity_of_ash Magnesium Total_phenols Flavanoids Nonflavanoid_phenols Proanthocyanins Color_intensity Hue OD280%2FOD315_of_diluted_wines Proline Class'][i].split(" "))
    result2 = pd.concat([result2, data], axis=1)


# In[5]:


wineTestData = result.transpose()
wineTestData.columns = wineTestData.iloc[0]
wineTestData = wineTestData.iloc[1:]
for i in wineTestData.columns:
    wineTestData[i] = pd.to_numeric(wineTestData[i])
wineTestData


# In[6]:


wineTrainData = result2.transpose()
wineTrainData.columns = wineTrainData.iloc[0]
wineTrainData = wineTrainData.iloc[1:]
for i in wineTrainData.columns:
    wineTrainData[i] = pd.to_numeric(wineTrainData[i])
wineTrainData


# In[7]:


featureList = ['Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols',
               'Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280%2FOD315_of_diluted_wines','Proline']
gigaset = pd.DataFrame(np.concatenate((wineTrainData, wineTestData), axis=0))
columnList = ['Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols',
               'Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280%2FOD315_of_diluted_wines','Proline', 'Class']
gigaset.columns = columnList
gigaset = gigaset.sample(frac=1)


# In[115]:


def accuracyScore(predicted_vals, y):
    score=0
    for i in range(len(y)):
        if y[i] == predicted_vals[i]:
            score = score + 1
    return score/len(y)

class KNN:
    def __init__(self, k):
        self.k = k
   
    def fit(self, train_data_X, train_data_Y):
        # min-max normalisation happens here
        self.train_data_X = (train_data_X - train_data_X.min(axis=0))/(train_data_X.max(axis=0) - train_data_X.min(axis=0))
        self.train_data_Y = train_data_Y
    
    def predict(self, test_data, features):
        # min-max normalisation happens here
        testData = test_data
        testData[features] = (testData[features] - testData[features].min(axis=0))/(testData[features].max(axis=0) - testData[features].min(axis=0))
        predictions = []
        for i in range(len(test_data)):
            distances = []
            for j in range(len(self.train_data_X)):
                x1 = testData[features].iloc[i]
                x2 = self.train_data_X.iloc[j]
                # distance is calculated here
                distance = np.sqrt(np.sum((x1-x2)**2))
                distances.append((distance, self.train_data_Y.iloc[j]))
            distances.sort()
            neighbors = distances[:self.k]
            classes = [neighbor[1] for neighbor in neighbors]
            prediction = max(set(classes), key=classes.count)
            predictions.append(prediction)
        return predictions


# In[116]:


# Make predictions
model = KNN(1)
model2 = KNN(3)
model3 = KNN(89)
model.fit(wineTrainData[featureList], wineTrainData['Class'])
model2.fit(wineTrainData[featureList], wineTrainData['Class'])
model3.fit(wineTrainData[featureList], wineTrainData['Class'])
predictions = model.predict(wineTestData,featureList)
predictions2 = model2.predict(wineTestData,featureList)
predictions3 = model3.predict(wineTestData,featureList)


# In[117]:
print("predicted labels when K=1: ", predictions)

classWT = wineTestData['Class'].tolist()


# In[118]:


print("K=1 Accuracy Score: ", accuracyScore(predictions, classWT))
print("K=3 Accuracy Score: ", accuracyScore(predictions2, classWT))
print("K=89 Accuracy Score: ", accuracyScore(predictions3, classWT))

