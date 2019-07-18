from numpy import expand_dims
from matplotlib import pyplot
import pickle
import numpy as np
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from scipy.spatial import distance

import mtcnn
from mtcnn.mtcnn import MTCNN


detector=MTCNN()
def extract_face(filename, required_size=(224, 224)):
  pixels = pyplot.imread(filename)
  detector = MTCNN()
  results = detector.detect_faces(pixels)
  x1, y1, width, height = results[0]['box']
  x2, y2 = x1 + width, y1 + height
  face = pixels[y1:y2, x1:x2]
  image = Image.fromarray(face)
  image = image.resize(required_size)
  face_array = asarray(image)
  return face_array


def get_embeddings(filenames):
  faces = [extract_face(f) for f in filenames]
  samples = asarray(faces, 'float32')
  samples = preprocess_input(samples, version=2)
  model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
  yhat = model.predict(samples)
  return yhat

a=get_embeddings(['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg'])

#dbfile = open('newEmb', 'ab') 
#pickle.dump(a, dbfile)                      
#dbfile.close()

b=get_embeddings(['10.jpg','20.jpg','30.jpg','40.jpg','50.jpg'])

for i in range(len(b)):
  print(i)
  for j in range(len(a)):
     if(distance.cosine(b[i],a[j])<0.5):
          print("Matched ",j)
        
c=get_embeddings(['100.jpg','200.jpg','300.jpg','400.jpg','500.jpg'])

for i in range(len(c)):
  print(i)
  for j in range(len(a)):
     if(distance.cosine(c[i],a[j])<0.5):
          print("Matched ",j)

