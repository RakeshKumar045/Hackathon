# coding: utf-8

# In[3]:


import cv2
import numpy as np
import os
import pandas as pd
import re
from keras import backend as K
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

# In[5]:


TRAIN_DATASET = 'train'
TEST_DATASET = 'test1'
IMG_SIZE = 50


# In[6]:


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat':
        return [1, 0]
    #                             [no cat, very doggo]
    elif word_label == 'dog':
        return [0, 1]


# In[7]:


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DATASET)):
        label = label_img(img)
        path = os.path.join(TRAIN_DATASET, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    # np.save('train_data.npy', training_data)
    return training_data


# In[8]:


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DATASET)):
        path = os.path.join(TEST_DATASET, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    # np.save('test_data.npy', testing_data)
    return testing_data


# In[9]:


# **Data dimensions and paths**
img_width = 150
img_height = 150
TRAIN_DIR = create_train_data()
TEST_DIR = process_test_data()

# In[10]:


train_images_dogs_cats = [TRAIN_DIR + i for i in TRAIN_DIR]  # use this for full dataset_D
test_images_dogs_cats = [TEST_DIR + i for i in TEST_DIR]

# In[11]:


train_images_dogs_cats


# In[ ]:


# **Helper function to sort the image files based on the numeric value in each file name.**
def atoi(text):
    return int(text) if text.isdigit() else text


# In[ ]:


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


# In[ ]:


# print(train_images_dogs_cats)


# In[ ]:


# **Sort the traning set. Use 1300 images each of cats and dogs instead of all 25000 to speed up the learning process.**

# **Sort the tensorbroad_pb_android set**

train_images_dogs_cats.sort(key=natural_keys)
train_images_dogs_cats = train_images_dogs_cats[0:1300] + train_images_dogs_cats[12500:13800]

test_images_dogs_cats.sort(key=natural_keys)


# In[ ]:


# print(train_images_dogs_cats)


# In[ ]:


# **Now the images have to be represented in numbers. For this, using the openCV library read and resize the image.  **

# **Generate labels for the supervised learning set.**

# **Below is the helper function to do so.**

def prepare_data(list_of_images):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """
    x = []  # images as arrays
    y = []  # labels

    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image), (img_width, img_height), interpolation=cv2.INTER_CUBIC))

    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)
        # else:
        # print('neither cat nor dog name present in images')

    return x, y


# In[ ]:


# **Generate X and Y using the helper function above**

# **Since K.image_data_format() is channel_last,  input_shape to the first keras layer will be (img_width, img_height, 3). 
# '3' since it is a color image**

X, Y = prepare_data(train_images_dogs_cats)
print(K.image_data_format())

# In[ ]:


# **Split the data set containing 2600 images into 2 parts, training set and validation set. Later,
# you will see that accuracy and loss on the validation set will also be reported 
# while fitting the model using training set.**

# First split the data in two sets, 80% for training, 20% for Val/Test_Amat)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=1)

# In[ ]:


nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 16

# In[ ]:


# **We will be using the Sequential model from Keras to form the Neural Network.
# Sequential Model is  used to construct simple models with linear stack of layers. **

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

# In[ ]:


# **This is the augmentation configuration we will use for training and validation**

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# In[ ]:


# **Prepare generators for training and validation sets**

train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)

# In[ ]:


# **Start training the model!**

# **For better accuracy and lower loss, we are using an epoch of 30. Epoch value can be increased for better results.**

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

# In[ ]:


# **Saving the model in Keras is simple as this! ** 

# **It is quite helpful for reuse.**


# model.save_weights('model_wieghts.h5')
model.save('book_model_keras1.h5')

# In[ ]:


# **Time to predict classification using the model on the tensorbroad_pb_android set.**

# **Generate X_test and Y_test**

X_test, Y_test = prepare_data(test_images_dogs_cats)  # Y_test in this case will be []

# In[ ]:


# **This is the augmentation configuration we will use for testing. Only rescaling.**

test_datagen = ImageDataGenerator(rescale=1. / 255)

# In[ ]:


# **Prepare generator for tensorbroad_pb_android set and start predicting on it.**

test_generator = val_datagen.flow(np.array(X_test), batch_size=batch_size)
prediction_probabilities = model.predict_generator(test_generator, verbose=1)

# In[ ]:


# **Generate .csv for submission**

counter = range(1, len(test_images_dogs_cats) + 1)
solution = pd.DataFrame({"id": counter, "label": list(prediction_probabilities)})
cols = ['label']

for col in cols:
    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)

solution.to_csv("dogsVScats.csv", index=False)
