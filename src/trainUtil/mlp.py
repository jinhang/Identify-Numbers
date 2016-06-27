# coding:utf-8
'''
Created on 2015年8月31日

@author: HadXu
'''

import pandas as pd
import multilayer_perceptron
from sklearn.externals import joblib


data = pd.read_csv('Data/train.csv')
y_train,X_train = data.values[:,0],data.values[:,1:]
 
mlp = multilayer_perceptron.MultilayerPerceptronClassifier()
mlp.fit(X_train, y_train)
 
joblib.dump(mlp,'ok.m')
 
print 'train is finished'

"""data_test = pd.read_csv('Data/test.csv')
X_test = data_test.values
mlp = joblib.load('ok.m')
test = mlp.predict(X_test)

df = pd.DataFrame({'ImageID':range(1, 28001), 'label':test})
df.to_csv('Data/predict.csv', index=False)"""




