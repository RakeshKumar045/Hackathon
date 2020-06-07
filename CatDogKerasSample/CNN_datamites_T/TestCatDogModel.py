# coding: utf-8

# In[26]:


from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# In[27]:


# img
get_ipython().run_line_magic('pwd', '')

# In[33]:


model = load_model("cat_dog_model.h5")

# In[42]:


img = load_img('test_cat3.jpg')  # this is a PIL image
img

# In[43]:


x = img_to_array(img.resize([224, 224]))  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)

if (model.predict_classes(x) == 1):
    print("It is a DOG")
else:
    print("It is a Cat")

# In[ ]:


# In[19]:


# img
