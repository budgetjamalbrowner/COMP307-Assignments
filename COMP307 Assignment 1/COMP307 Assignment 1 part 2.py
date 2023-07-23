#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import collections
from collections import Counter
import sys

# ## Question 3 

# a)

# In[2]:


file = sys.argv[1]
file1 = sys.argv[2]
file2 = sys.argv[3]
file3 = sys.argv[4]
file4 = sys.argv[5]
file = pd.read_csv(file)
file1 = pd.read_csv(file1)
file2 = pd.read_csv(file2)
file.to_csv('hepatitis.csv')
file1.to_csv('hepatitis-training.csv')
file2.to_csv('hepatitis-test.csv')
file3 = pd.read_csv(file3)
file4 = pd.read_csv(file4)
file3.to_csv('golf-training.csv')
file4.to_csv('golf-test.csv')


# In[3]:


hepatitisDF = pd.read_csv('hepatitis.csv')
hepatitisTrainDF = pd.read_csv('hepatitis-training.csv')
hepatitisTestDF = pd.read_csv('hepatitis-test.csv')
golfTrainDF= pd.read_csv('golf-training.csv')
golfTestDF = pd.read_csv('golf-test.csv')
golfTestDF


# In[4]:


result4 = pd.DataFrame('Class Cloudy Raining Hot Cold Humid Windy'.split(" "))
result5 = pd.DataFrame('Class Cloudy Raining Hot Cold Humid Windy'.split(" "))
for i in range(len(golfTrainDF)):
    data = pd.DataFrame(golfTrainDF['Class Cloudy Raining Hot Cold Humid Windy'][i].split(" "))
    result4 = pd.concat([result4, data], axis=1)
for i in range(len(golfTestDF)):
    data = pd.DataFrame(golfTestDF['Class Cloudy Raining Hot Cold Humid Windy'][i].split(" "))
    result5 = pd.concat([result5, data], axis=1)


# In[5]:


golfTrainData = result4.transpose()
golfTrainData.columns = golfTrainData.iloc[0]
golfTrainData = golfTrainData.iloc[1:]
golfTrainData = golfTrainData.reset_index(drop=True)
golfTrainDataX = golfTrainData.drop('Class', axis = 1)
golfTrainDataY = golfTrainData['Class']


# In[6]:


golfTestData = result5.transpose()
golfTestData.columns = golfTestData.iloc[0]
golfTestData = golfTestData.iloc[1:]
golfTestData = golfTestData.reset_index(drop=True)
golfTestDataX = golfTestData.drop('Class', axis = 1)
golfTestDataY = golfTestData['Class']


# In[7]:


result = pd.DataFrame('Class AGE FEMALE STEROID ANTIVIRALS FATIGUE MALAISE ANOREXIA BIGLIVER FIRMLIVER SPLEENPALPABLE SPIDERS ASCITES VARICES BILIRUBIN SGOT HISTOLOGY'.split(" "))
result2 = pd.DataFrame('Class AGE FEMALE STEROID ANTIVIRALS FATIGUE MALAISE ANOREXIA BIGLIVER FIRMLIVER SPLEENPALPABLE SPIDERS ASCITES VARICES BILIRUBIN SGOT HISTOLOGY'.split(" "))
result3 = pd.DataFrame('Class AGE FEMALE STEROID ANTIVIRALS FATIGUE MALAISE ANOREXIA BIGLIVER FIRMLIVER SPLEENPALPABLE SPIDERS ASCITES VARICES BILIRUBIN SGOT HISTOLOGY'.split(" "))
for i in range(len(hepatitisDF)):
    data = pd.DataFrame(hepatitisDF['Class AGE FEMALE STEROID ANTIVIRALS FATIGUE MALAISE ANOREXIA BIGLIVER FIRMLIVER SPLEENPALPABLE SPIDERS ASCITES VARICES BILIRUBIN SGOT HISTOLOGY'][i].split(" "))
    result = pd.concat([result, data], axis=1)
for i in range(len(hepatitisTrainDF)):
    data = pd.DataFrame(hepatitisTrainDF['Class AGE FEMALE STEROID ANTIVIRALS FATIGUE MALAISE ANOREXIA BIGLIVER FIRMLIVER SPLEENPALPABLE SPIDERS ASCITES VARICES BILIRUBIN SGOT HISTOLOGY'][i].split(" "))
    result2 = pd.concat([result2, data], axis=1)
for i in range(len(hepatitisTestDF)):
    data = pd.DataFrame(hepatitisTestDF['Class AGE FEMALE STEROID ANTIVIRALS FATIGUE MALAISE ANOREXIA BIGLIVER FIRMLIVER SPLEENPALPABLE SPIDERS ASCITES VARICES BILIRUBIN SGOT HISTOLOGY'][i].split(" "))
    result3 = pd.concat([result3, data], axis=1)


# In[8]:


hepatitisData = result.transpose()
hepatitisData.columns = hepatitisData.iloc[0]
hepatitisData = hepatitisData.iloc[1:]
hepatitisData = hepatitisData.reset_index(drop=True)


# In[9]:


hepatitisTrainData = result2.transpose()
hepatitisTrainData.columns = hepatitisTrainData.iloc[0]
hepatitisTrainData = hepatitisTrainData.iloc[1:]
hepatitisTrainData = hepatitisTrainData.reset_index(drop=True)
hepatitisTrainDataX = hepatitisTrainData.drop('Class', axis = 1)
hepatitisTrainDataY = hepatitisTrainData['Class']


# In[10]:


hepatitisTestData = result3.transpose()
hepatitisTestData.columns = hepatitisTestData.iloc[0]
hepatitisTestData = hepatitisTestData.iloc[1:]
hepatitisTestData = hepatitisTestData.reset_index(drop=True)
hepatitisTestDataX = hepatitisTestData.drop('Class', axis = 1)
hepatitisTestDataY = hepatitisTestData['Class']


# In[64]:


# define node used in DecisionTree
class Node:
    def __init__(self, featureName,prob=None, feature=None, threshold=None, left=None, right=None, *, value=None, gain = None, depth = 0):
        self.feature = feature
        self.featureName = featureName[feature]
        self.threshold = threshold 
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain
        self.depth = depth
        self.prob = prob
    def isLeafNode(self):
        return self.value is not None
    def report(self, indent,  classVal):
        if self.feature != None:
            print("{}{} = True\n".format(indent, self.featureName))
        if self.left != None:
            self.left.report(indent + "    ",  classVal)
        if self.feature != None:
            print("{}{} = False\n".format(indent, self.featureName))
        if self.right != None:
            self.right.report(indent + "    ",  classVal)
        if self.isLeafNode():
            if self.value == 1:
                n = list(classVal)[1]
                print(indent + "Class: " + n + ", probability = " + str(self.prob))
            if self.value == 0:
                n = list(classVal)[0]
                print(indent + "Class: " + n + ",  probability = " + str(self.prob))


# In[65]:


# define functions used in DecisionTree
class DecisionTree:
    def __init__(self, minSamplesSplit, maxDepth, featureCount):
        self.minSamplesSplit = minSamplesSplit
        self.maxDepth = maxDepth
        self.featureCount = featureCount
        self.root=None
    
    def mostCommonLabel(self,y):
        counter = Counter(y)
        val = counter.most_common(1)[0][0]
        return val
    
    # chosen impurity measurement: entropy
    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist/len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    def split(self, xCol, splitThreshold):
        leftIdxs = np.argwhere(xCol<=splitThreshold).flatten()
        rightIdxs = np.argwhere(xCol>splitThreshold).flatten()
        return leftIdxs, rightIdxs
    
    # infoGain utilises entropy 
    def infoGain(self, xCol, y, threshold):
        parentEntropy = self.entropy(y)
        leftIdxs, rightIdxs = self.split(xCol, threshold)
        if len(leftIdxs) == 0 or len(rightIdxs) == 0:
            return 0
        n = len(y)
        nLeft, nRight = len(leftIdxs), len(rightIdxs)
        eLeft, eRight = self.entropy(y[leftIdxs]), self.entropy(y[rightIdxs])
        childEntropy = (nLeft/n) * eLeft + (nRight/n) * eRight
        infoGain = parentEntropy - childEntropy
        return infoGain
    
    def fit(self, X, y):
        numY = y.replace(list(set(y))[1], 1)
        numY = numY.replace(list(set(y))[0], 0)
        numY = numY.astype('int32')
        self.featureCount = X.values.shape[1] if not self.featureCount else min(X.values.shape[1], self.featureCount)
        self.y = y
        self.root = self.growTree(X.columns, X.values, numY.values)
    
    def bestSplit(self, X, y, featIdxs):
        bestGain = -1
        splitIdx, splitThresh = None, None
        for f in featIdxs:
            XCol = X[:, f]
            thresholds = np.unique(XCol)
            for t in thresholds:
                gain = self.infoGain(XCol, y, t)
                if gain > bestGain:
                    bestGain = gain
                    splitIdx = f
                    splitThresh = t
        return splitIdx, splitThresh, bestGain
        
    def growTree(self, featurelist, X, y, depth = 0):
        sampleCount, featureCount2 = X.shape
        labelCount = len(np.unique(y))
        if (depth>=self.maxDepth or labelCount==1 or sampleCount<self.minSamplesSplit):
            leafValue = self.mostCommonLabel(y)
            onesCount = np.count_nonzero(y==1)
            oneProb = onesCount/len(y)
            probability = np.prod(y*oneProb+(1-y)*(1-oneProb))
            return Node(featureName = featurelist, prob = probability, value=leafValue)
        featureIdx = np.random.choice(featureCount2, self.featureCount, replace=False)
        bestFeature, bestThreshold, bGain = self.bestSplit(X, y, featureIdx)
        leftIdxs, rightIdxs = self.split(X[:,bestFeature], bestThreshold)
        left = self.growTree(featurelist,X[leftIdxs, :], y[leftIdxs], depth+1)
        right = self.growTree(featurelist,X[rightIdxs, :], y[rightIdxs], depth+1)
        return Node(featureName = featurelist, feature = bestFeature, threshold = bestThreshold, left = left, right = right, gain = bGain, depth = depth)
    
    def predict(self, X):
        predList = np.array([self.traverseTree(x, self.root) for x in X.values]).tolist()
        winCount = predList.count(1)
        self.root.report(' ', classVal = set(self.y))
        return np.array([self.traverseTree(x, self.root) for x in X.values])
    
    def traverseTree(self, x, node):
        if node.isLeafNode():
            return node.value
        if x[node.feature]<= node.threshold:
            return self.traverseTree(x, node.left)
        else: 
            return self.traverseTree(x, node.right)


# In[66]:


model = DecisionTree(2, 10000000, None)

def accuracyScore(predictions, yTest):
    return np.sum(yTest == predictions)/len(yTest)

model.fit(hepatitisTrainDataX, hepatitisTrainDataY)


# In[67]:


predictions = model.predict(hepatitisTestDataX)


# In[15]:


hepatitisTestY = hepatitisTestDataY.replace(list(set(hepatitisTestDataY))[1], 1)
hepatitisTestY = hepatitisTestY.replace(list(set(hepatitisTestDataY))[0], 0)
hepatitisTestY.astype('int32')
accuracyScore(predictions, hepatitisTestY)


# In[16]:


predictions


# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[18]:


baseline = DecisionTreeClassifier(random_state = 100)
hepatitisTrainDataX = hepatitisTrainDataX.replace('false', 0)
hepatitisTrainDataX = hepatitisTrainDataX.replace('true', 1)
hepatitisTestDataX = hepatitisTestDataX.replace('true', 1)
hepatitisTestDataX = hepatitisTestDataX.replace('false', 0)


# In[19]:


baseline.fit(hepatitisTrainDataX, hepatitisTrainDataY.values)


# In[20]:


basePredictions = baseline.predict(hepatitisTestDataX)


# In[21]:


accuracyScore(basePredictions, hepatitisTestDataY.values)


# Model tested on golf data:

# In[69]:


model2 =  DecisionTree(2, 100, None)
model2.fit(golfTrainDataX, golfTrainDataY)


# In[70]:


golfPreds = model2.predict(golfTestDataX)


# In[71]:


golfTestY = golfTestDataY.replace(list(set(golfTestDataY))[1], 1)
golfTestY = golfTestY.replace(list(set(golfTestDataY))[0], 0)
golfTestY.astype('int32')
accuracyScore(golfPreds, golfTestY)


# There is no difference between the baseline classifier and my own decision tree classifier, since I used default parameters for my classifier that are identical to the baseline sklearn decision tree classifier, where minsamplesplit = 2, maxDepth = redundant number and featureCount is set to None.

# b)

# 
