Flower Recognition CNN Keras
[ Please upvote / star if you like it ;) ]
import os
print(os.listdir('../input/flowers/flowers'))
['sunflower', 'tulip', 'daisy', 'rose', 'dandelion']
 
CONTENTS ::
1 ) Importing Various Modules

2 ) Preparing the Data

3 ) Modelling

4 ) Evaluating the Model Performance

5 ) Visualizing Predictons on the Validation Set


1 ) Importing Various Modules.
# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
Using TensorFlow backend.

2 ) Preparing the Data
2.1) Making the functions to get the training and validation set from the Images
X=[]
Z=[]
IMG_SIZE=150
FLOWER_DAISY_DIR='../input/flowers/flowers/daisy'
FLOWER_SUNFLOWER_DIR='../input/flowers/flowers/sunflower'
FLOWER_TULIP_DIR='../input/flowers/flowers/tulip'
FLOWER_DANDI_DIR='../input/flowers/flowers/dandelion'
FLOWER_ROSE_DIR='../input/flowers/flowers/rose'
def assign_label(img,flower_type):
    return flower_type
    
def make_train_data(flower_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))
        
        
        
make_train_data('Daisy',FLOWER_DAISY_DIR)
print(len(X))
100%|██████████| 769/769 [00:03<00:00, 215.70it/s]
769
make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
print(len(X))
100%|██████████| 734/734 [00:03<00:00, 206.81it/s]
1503
make_train_data('Tulip',FLOWER_TULIP_DIR)
print(len(X))
100%|██████████| 984/984 [00:04<00:00, 224.01it/s]
2487
make_train_data('Dandelion',FLOWER_DANDI_DIR)
print(len(X))
  9%|▉         | 97/1055 [00:00<00:04, 235.89it/s]
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
<ipython-input-9-95c78ead0c98> in <module>
----> 1 make_train_data('Dandelion',FLOWER_DANDI_DIR)
      2 print(len(X))

<ipython-input-5-001b1f747236> in make_train_data(flower_type, DIR)
      4         path = os.path.join(DIR,img)
      5         img = cv2.imread(path,cv2.IMREAD_COLOR)
----> 6         img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
      7 
      8         X.append(np.array(img))

error: OpenCV(3.4.3) /io/opencv/modules/imgproc/src/resize.cpp:4044: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
make_train_data('Rose',FLOWER_ROSE_DIR)
print(len(X))
100%|██████████| 784/784 [00:03<00:00, 235.31it/s]
3386
2.2 ) Visualizing some Random Images
fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Z[l])
        
plt.tight_layout()
        

