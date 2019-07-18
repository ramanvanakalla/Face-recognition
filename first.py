
"""Untitled9.ipynb





"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2

from keras.optimizers import SGD
from keras.optimizers import Adam

import matplotlib.pyplot as plt

data=pd.read_csv('wild_faces_labels.csv')
import mtcnn
from mtcnn.mtcnn import MTCNN

detector=MTCNN()


from keras.models import load_model

print("Bro")

model1 = load_model('facenet_keras.h5')

print("yo yo")


labels=pd.read_csv("wild_faces_labels.csv")
labels=np.asarray(labels)

def find_faces(name,a,b):
  l=[]
  for i in range(a,b+1):
    print(i)
    n=len(str(i))
    img=Image.open(name+str(0)*(6-n)+str(i)+".jpg")
    img = img.convert('RGB')
    img=np.asarray(img)
    results=detector.detect_faces(img)
    if(len(results)==0):
      l.append(face)
      labels[i-1][1]=labels[i-2][1]
      continue
    x=abs(results[0]['box'][0])
    y=abs(results[0]['box'][1])
    w=abs(results[0]['box'][2])
    h=abs(results[0]['box'][3])
    face=img[y:y+h,x:x+w]
    l.append(face)
  return l

def embeddings(faces_list):
  li=[]
  n=len(faces_list)
  for i in range(n):
    face=faces_list[i]
    face=Image.fromarray(face)
    samples=face.resize((160,160))
    samples =np.expand_dims(samples, axis=0)
    res=model1.predict(samples)
    li.append(res)
  return li

train_faces_list=find_faces("images/img_align_celeba/",1,30000)

embed=embeddings(train_faces_list)

def prepareXY(X,Y):
  pos=0
  neg=0
  for i in range(30000):
    print("i ",i,"pos ",pos,"neg ",neg)
    for j in range(i+1,30000):
      if(labels[i][1]==labels[j][1]):
        pos+=1
        Y.append(1)
        Y.append(1)
        a=embed[i]
        b=embed[j]
        c=np.concatenate((a,b),axis=1)
        X.extend(c)
        d=np.concatenate((b,a),axis=1)
        X.extend(d)
      elif(neg<50000):
        neg+=1
        Y.append(0)
        Y.append(0)
        a=embed[i]
        b=embed[j]
        c=np.concatenate((a,b),axis=1)
        X.extend(c)
        d=np.concatenate((b,a),axis=1)
        X.extend(d)
      


X=[]
Y=[]
prepareXY(X,Y)


print(np.shape(X))
print(np.shape(Y))

model=Sequential()
model.add(Dense(1,activation='sigmoid',input_shape=(256,)))
model.summary()

model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False),metrics=['acc'])

history = model.fit(np.array(X),Y,epochs=100,verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("my_model.h5")
print("Saved model to disk")




