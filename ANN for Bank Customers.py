#!/usr/bin/env python
# coding: utf-8

# ### Pandas for data manipulation, NumPy for numerical operations, TensorFlow for machine learning

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[2]:


tf.__version__


# In[19]:


dataset = pd.read_csv('Churn_Modelling.csv')
print(dataset)


# ### X keep columns from 4th till the penultimate column
# ### y keep the last column

# In[20]:


X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:,-1].values


# In[23]:


print(X)
print(y)


# ### LabelEncoder is used to convert categorical labels into numerical format for machine learning models

# In[24]:


from sklearn.preprocessing import LabelEncoder


# In[25]:


le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:,2])
print(X)


# ### ColumnTransformer and OneHotEncoder are used for transforming specific columns in a dataset, especially for encoding categorical variables into a binary matrix format (one-hot encoding
# ### Binary matrix is a matrix with entries from the Boolean domain B = {0, 1}. Such a matrix can be used to represent a binary relation between a pair of finite sets

# In[26]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


#  ### Apply OneHotEncoder to the column at index 1, while keeping the other columns unchanged ("remainder='passthrough'")

# In[27]:


ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[28]:


print(X)


# ### 80% of the data is used for training (X_train and y_train), and 20% of the data is used for testing (X_test and y_test). This is specified by the test_size=0.2 parameter in the train_test_split function.
# ### random_state=0 ensures that the split is reproducible, meaning that if you run the code again with the same random_state value, you will get the same split.

# In[30]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.2 , random_state=0)


# ### The purpose of StandardScaler is to make the values comparable and easier for a computer to understand. The relative relationships (eg between ages and incomes) remain the same, but they are now on a standardized scale.
# 
# ### Example :
# 
# ##### Before scaling:
# 
# ##### Age: [8, 10, 12, 15]
# ##### Income: [5, 10, 15, 20]
# ##### After scaling with StandardScaler:
# 
# ##### Age: [-1.34, -0.45, 0.45, 1.34]
# ##### Income: [-1.34, -0.45, 0.45, 1.34]

# In[33]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ##### There are two ways to build Keras models: sequential and functional.
# 
# ##### The sequential API allows you to create models layer-by-layer for most problems. It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs.
# 
# ##### Alternatively, the functional API allows you to create models that have a lot more flexibility as you can easily define models where layers connect to more than just the previous and next layers. In fact, you can connect layers to (literally) any other layer. As a result, creating complex networks such as siamese networks and residual networks become possible.

# In[35]:


ann = tf.keras.models.Sequential()


# ### This code adds a new layer to the neural network (ann). The layer is a type called "Dense," which means every neuron in this layer is connected to every neuron in the previous layer. It has 6 neurons (units), and the activation function used in each neuron is 'relu' (Rectified Linear Unit).
# 
# ### Rectified Linear Unit, or ReLU, is an activation function commonly used in neural networks. It's a simple mathematical function that returns the input value if it's positive, and zero otherwise. Mathematically, the ReLU function is defined as: f(x)=max(0,x)

# In[36]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# In[37]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# #### Each neuron in a layer is responsible for learning and recognizing different patterns in the input data. The number of units in a layer is a hyperparameter that you can adjust based on the complexity of the problem you are trying to solve. More units generally allow the network to learn more complex representations but may also increase the computational cost. The choice of the number of units often involves experimentation and tuning to find a suitable balance for the specific task at hand.
# #### Setting activation='sigmoid' in a neural network layer means that the activation function used in each neuron of that layer is the sigmoid function. The sigmoid function is a mathematical function that compresses its input values between 0 and 1.

# In[38]:


ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ####  This line compiles the neural network (ann) with specific configurations:
# 
# #### Optimizer: 'adam' is the optimization algorithm used during training. Adam is a popular optimization algorithm that adapts the learning rates of each parameter during training.
# 
# #### Loss Function: 'binary_crossentropy' is the loss function used to measure how well the neural network is performing. It is commonly used for binary classification problems.
# 
# #### Metrics: 'accuracy' is chosen as the metric to evaluate the performance of the model. It represents the proportion of correctly classified samples out of the total.
# 
# #### In summary, this line prepares the neural network for training by specifying how it should optimize its weights (optimizer), what it should try to minimize (loss function), and what performance metric to monitor (accuracy).

# In[41]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### this line is telling the neural network to learn from the training data in 100 cycles (epochs), updating its weights after processing each batch of 32 examples

# In[42]:


ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


# ### Probability Prediction 

# In[43]:


print(ann.predict(sc.transform([[1,0,0,  600,  1,  40,  3,  60000,   2,  1,  1,   50000]])))


# ### Binary Prediction

# In[44]:


print(ann.predict(sc.transform([[1,0,0,  600,  1,  40,  3,  60000,   2,  1,  1,   50000]]))>0.5)


# #### Prints a concatenated array showing the predicted values (y_pred) alongside the actual values (y_test).

# In[45]:


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# ### This code uses scikit-learn's functions to evaluate the performance of your model on the test set:
# 
# ###  confusion_matrix: It computes a confusion matrix, which is a table showing the number of true positives, true negatives, false positives, and false negatives.
# 
# ###  accuracy_score: It calculates the accuracy of your model, which is the proportion of correctly predicted instances out of the total instances.

# In[46]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# #### The model made 1525 correct predictions of class 0.
# #### The model made 200 correct predictions of class 1.
# #### The model made 70 incorrect predictions, wrongly classifying instances as class 1.
# #### The model made 205 incorrect predictions, wrongly classifying instances as class 0.
# 
# #### Accuracy = (1525 + 200)/(1525 + 70 + 205 + 200) = 0.8625
