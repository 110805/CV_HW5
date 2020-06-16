import pandas as pd
import numpy as np
import os
import cv2

def train_img_feats():
    df = pd.read_csv('train.csv')
    xTrain = df['image_id']
    yTrain, labels = pd.factorize(df['label'], sort=True)
    features = np.empty((len(xTrain), 256))
    for i in range(len(xTrain)):
        img = cv2.imread(os.path.join('hw5_data/train', xTrain[i]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (16, 16))
        img = img.ravel()
        features[i] = (img - img.mean()) / img.std() # standardize the feature vector to improve the accuracy
    
    return features, yTrain

def knn(features, yTrain, k):
    df = pd.read_csv('test.csv')
    xTest = df['image_id']
    yTest, labels = pd.factorize(df['label'], sort=True)
    acc = 0
    for i in range(len(yTest)):
        img = cv2.imread(os.path.join('hw5_data/test', xTest[i]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (16, 16))
        img = img.ravel()
        img = (img - img.mean()) / img.std()
        dist = []
        for j in range(len(yTrain)):
            dist.append(np.linalg.norm(features[j] - img))
        
        dist = np.array(dist)
        neighbors = np.argsort(dist)
        neighbors = neighbors[:k]
        candidate = []
        for idx in neighbors:
            candidate.append(yTrain[idx])
        
        predict = max(set(candidate), key = candidate.count) # find the most frequent element in a list 
        if predict == yTest[i]:
            acc += 1
        
    acc /= len(yTest)
    return acc

if __name__ == '__main__':
    k = 7
    features, yTrain= train_img_feats()
    acc = knn(features, yTrain, k)
    print(acc)