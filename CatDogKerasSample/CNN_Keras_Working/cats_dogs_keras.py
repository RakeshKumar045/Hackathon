# coding: utf-8

# In[18]:


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# In[19]:


get_ipython().run_line_magic('pwd', '')

# In[20]:


# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# img = load_img('/Users/ashok/Downloads/cats_dogs/train/cats/cat.0.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='/Users/ashok/Downloads/preview', save_prefix='cat', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely


# In[21]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)


# In[22]:


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# In[23]:


batch_size = 16
# batch_size = 1

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    'cats_dogs/train',  # this is the target directory
    target_size=(128, 128),  # all images will be resized to 150x150
    batch_size=batch_size,
    class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    'cats_dogs/validation',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary')

# In[24]:


model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=800 // batch_size)
model.save_weights('cat_dog_model_128.h5')  # always save your weights after training or during training

# In[25]:


# model.load_weights('first_try.h5')


# In[32]:


model.save("cat_dog_model_128.h5")

# In[33]:


from keras.preprocessing.image import load_img, img_to_array

# In[34]:


img = load_img('test_dog1.jpg')  # this is a PIL image
x = img_to_array(img.resize([224, 224]))  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# In[35]:


img

# In[36]:


if (model.predict_classes(x) == 1):
    print("It is a DOG")
else:
    print("It is a Cat")

# In[31]:


model.predict_classes(x)
