
# coding: utf-8

# In[15]:


import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, avg_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import numpy as np


# In[16]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[25]:


# Loading the data from local directory

train_data1 = unpickle("/home/skhan/cifar-10-batches-py/data_batch_1")
train_data2 = unpickle("/home/skhan/cifar-10-batches-py/data_batch_2")
train_data3 = unpickle("/home/skhan/cifar-10-batches-py/data_batch_3")
train_data4 = unpickle("/home/skhan/cifar-10-batches-py/data_batch_4")
train_data5 = unpickle("/home/skhan/cifar-10-batches-py/data_batch_5")
label_name = unpickle("/home/skhan/cifar-10-batches-py/batches.meta")
test_data = unpickle("/home/skhan/cifar-10-batches-py/test_batch")


# In[26]:


# Intializing the variables

X = np.concatenate((train_data1[b'data'], train_data2[b'data'], train_data3[b'data'], train_data4[b'data'], train_data5[b'data'])) 
y = np.concatenate((train_data1[b'labels'], train_data2[b'labels'], train_data3[b'labels'], train_data4[b'labels'], train_data5[b'labels']))
X_test = test_data[b'data']
y_test = test_data[b'labels']

# Shuffle the data
X, y = shuffle(X, y)

# Reshape the data

X = np.reshape(X,(50000,32,32,3))
X_test = np.reshape(X_test, (32,32,3,10000))


# In[31]:


# Data loading and preprocessing

from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical
(X, y), (X_test, y_test) = cifar10.load_data()
X, y = shuffle(X, y)
y = to_categorical(y, 10)
y_test = to_categorical(y_test, 10)


# In[19]:


# Make sure the data is normalized

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()


# In[20]:


# Create extra synthetic training data by flipping, rotating and blurring the images on our data set.

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)


# In[ ]:


# Network Architecture

# Input layer 32*32*3 dimension image with 3 color dimensions
model = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)

# Convolution layer
model = conv_2d(model, 32, 3, activation='relu')

# Average pooling
model = avg_pool_2d(model, 2)

# Convolution layer
model = conv_2d(model, 64, 3, activation='relu')

# Convolution layer
model = conv_2d(model, 64, 3, activation='relu')

# Average pooling
model = avg_pool_2d(model, 2)

# Fully-connected
model = fully_connected(model, 512, activation='relu')

# Dropout
model = dropout(model, 0.5)

# Fully-connected
model = fully_connected(model,10, activation='softmax')


# In[ ]:


# Provied the training estimator the network
model = regression(model, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

# Wrap the model in a network object
network = tflearn.DNN(model, tensorboard_verbose=0, checkpoint_path='bird-classifier.ckpt')

# Train the network
network.fit(X, y, n_epoch=100, shuffle=True, validation_set=(X_test, y_test), show_metric=True, batch_size=96, snapshot_epoch=True, run_id='bird-classifier')

# Save model when training is complete to a file

model.save("bird-classifier")
print("Network trained and saved as bird-classifier!")

