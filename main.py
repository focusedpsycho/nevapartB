from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.utils import np_utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
irisData= pd.read_csv("Iris.csv")
irisData.loc[irisData["Species"]=="Iris-setosa","Species"]=0
irisData.loc[irisData["Species"]=="Iris-versicolor","Species"]=1
irisData.loc[irisData["Species"]=="Iris-virginica","Species"]=2
irisData=irisData.iloc[np.random.permutation(len(irisData))]
X=irisData.iloc[:,1:5].values
Y=irisData.iloc[:,5].values
X_normalized=normalize(X,axis=0)
train_length= 120
test_length= 30
X_train=X_normalized[:train_length]
X_test=X_normalized[train_length:]
y_train= Y[:train_length]
y_test = Y[train_length:]
y_train=np_utils.to_categorical(y_train,num_classes=3)
y_test=np_utils.to_categorical(y_test,num_classes=3)
model= Sequential()
model.add(Dense(400,input_dim=4,activation='selu'))
model.add(Dense(300,activation='selu'))
model.add(Dense(150,activation='selu'))
model.add(Dropout(0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=10,verbose=1)
prediction=model.predict(X_test)
length=len(prediction)
y_label=np.argmax(y_test,axis=1)
predict_label=np.argmax(prediction,axis=1)
accuracy=np.sum(y_label==predict_label)/length * 100
print("Accuracy of the dataset",accuracy )
