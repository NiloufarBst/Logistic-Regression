#!/usr/bin/env python
# coding: utf-8

# # G7 - Besharatzad and Agarwal

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
import random


# no_features --> number of features 
# exmpl_size --> number of training examples in data-set 
# x_train(and x_valid) --> input data matrix of shape (m x n) 
# y_train(and y_valid) --> true/ target value (can be 0 or 1 only) 
# x(i), y(i) --> ith training example weights (parameters) of shape (n x 1) 
# hypo --> hypothesis (outputs values between 0 and 1)

# In[2]:


def load_data(path):
    
    try:
        loaded_data = pd.read_csv(path)
    except:
        print('The required file does not exist!')    

    return loaded_data


# Data Preprocessing 

# In[3]:


# Part 0  : Data preprocessing.
def preprocess_data(data, normalize):
    
    columns_to_normalize = ['Annual_Premium','Vintage','Age']
    
    mean_and_std = {}
    
    if(normalize):
        for column in columns_to_normalize:

            mean = data[column].mean()
            std = data[column].std()
            data[column] = (data[column]-mean)/std
            mean_and_std[column] = {'mean':mean, 'std':std}
    
    preprocessed_data = data

    return preprocessed_data, mean_and_std


# In[4]:


loaded_data = load_data("IA2-train.csv")
preprocessed_data, mean_std = preprocess_data(loaded_data, True)


# In[5]:


labels = preprocessed_data['Response']
x_train= preprocessed_data.drop(['Response'] , axis=1)
y_train = labels.to_numpy()


# In[6]:


def normalizeValidationData(validation_data,mean_and_std):
    columns_to_normalize = ['Annual_Premium','Vintage','Age']
    
    for column in columns_to_normalize:
        mean = mean_and_std[column]['mean']
        std = mean_and_std[column]['std']
        validation_data[column] = (validation_data[column]-mean)/std
    
    return validation_data


# In[7]:


loaded_data = load_data("IA2-dev.csv")
validation_data = normalizeValidationData(loaded_data, mean_std)


# In[8]:


labels_valid = validation_data['Response']
x_valid= validation_data.drop(['Response'] , axis=1)
# x_valid = validation_data.to_numpy()
# labels_valid = validation_data['Response']
# del validation_data['Response']
y_valid = labels_valid.to_numpy()


# In[9]:


def sigmoid_z(sigma):
    return 1.0/(1 + np.exp(-sigma))


# In[10]:


def loss_func(y, hypo):
    loss = -np.mean(y*(np.log(hypo)) + (1-y)*np.log(1-hypo))
    
    return loss


# Note: There is a difference in batch gradient calculation. We have strong reason to suspect that performing difference of y and hypothesis changes agorithm from gradient descent to ascent. To achieve higher accuracy and contain loss, we perform "hypo - y"

# In[11]:


def batch_gradients(X, y, hypo):
    
    exmpl_size = X.shape[0]
    
    # For change in weights
    del_w = (1/exmpl_size)*np.dot(X.T, (hypo - y))
    
    return del_w


# In[12]:


def accuracy_func(weight, X_input, Y_input, Hypothesis):
    match = 0
    size = len(X_input)
    classifier = (Hypothesis > 0.5)
    test_y = Y_input
    match = (classifier == test_y)
    achieved_accuracy = (np.sum(match) / size)*100
    return achieved_accuracy


# # Part 1

# The below code was used for plotting and finding the best learning rate for individual lambda. It has been commented out for TA's convenience and to avoid pollution. 

# In[13]:


# learning_rates = [10**i for i in range(-2,1)]
# acc_lr = {rate:{"accuracy1":[],"accuracy2":[]}  for rate in learning_rates}
# def plot_learning_for_lamda(acc):
#     plt.figure(figsize=(8, 6), dpi=80)
    
#     labels = [rate for rate in acc]
#     colors = []
#     for i in range(len(acc)):
#         colors.append('#%06X' % randint(0, 0xFFFFFF))
    
#     for i,rate in enumerate(acc):
        
#         x = np.array([i for i in range(0,len(acc[rate]['accuracy1']))])
#         y = acc[rate]['accuracy1']
#         plt.plot(x,y,color=colors[i],label=labels[i])
        
#         x1 = np.array([i for i in range(0,len(acc[rate]['accuracy2']))])
#         y1 = acc[rate]['accuracy2']
#         plt.plot(x1,y1,color=colors[i], linestyle='dashed')
       
#         plt.plot(x,y,color=colors[i],label=labels[i])
#         plt.xlabel("Epoch")
#         plt.ylabel("Accuracy")
#         plt.legend(handles=[x], loc='upper right')
#         plt.legend(handles=[x], loc='upper right')
#         plt.title("Accuracy vs Epoch for different learning rates")
#     return


# Two lists are created one for lambdas and another for their learning rates. The selected learning rates converge fastest and reach higher accuracy for respective lambda value. They mapped/used through same indices. 

# In[14]:


lamda = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
learned_lr = [1, 1, 1, 1, 1, 0.1, 0.01]
sparse_weights_lm = {lamda[i]:{'sparse_weights':[]} for i in range(7)}
acc_lamda = {lamda[i]:{'lr_lambda':learned_lr[i], 'accuracy1':[], 'accuracy2':[]} for i in range(7)}
max_weights = {lamda[i]:{'weights':[]} for i in range(7)}


# In[15]:


def plot_accuracy(acc, lamda, label):
    
    # Your code here:
    number_of_colors = 4
    
    plt.figure(figsize=(8, 6), dpi=80)
    size = len(lamda)
    x = list(range(size))
    plt.plot(x, acc, label='Accuracy', marker='d')
    #plt.plot(x, acc, label='Valid Accuracy', marker='d')
    _=plt.xticks(x, lamda)
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.legend(['Accuracy'])
    plt.title(label)
       
    return


# Code for Ridge Regression 

# In[16]:


def LR_L2_train(x, y, weight, lamda, learning_rate, store_w):
         
    exmpl_size, no_features = x.shape
    iterations = 1000 
    
    # Initializing weights 
    weight = weight
               
    # Reshaping y
    y = y.reshape(exmpl_size,1)
    
    #list to store losses
    all_accuracies = []
    
    # Training loop.
    for iterations in range(iterations):
            
        # Calculating hypothesis
        hypo = sigmoid_z(np.dot(x, weight))
        
        # gradients of loss
        new_weight = batch_gradients(x, y, hypo) + weight*lamda
            
        # Updating weight 
        weight -= learning_rate*new_weight
        
        # Calculating loss 
        current_loss = loss_func(y, hypo) + (lamda * np.sum(np.square(weight)))
        
       
    # Get accuracy
    all_accuracies.append(accuracy_func(weight, x, y, hypo)) 
    no_of_sparseweights = np.where(weight<10**-6)[0].shape[0] 
   
    # return values
    if(store_w):
        return all_accuracies, weight, no_of_sparseweights, abs(weight) 
    else:
        return all_accuracies 


# In[17]:


def plot_sparse_weights(s_w, lamda):
    
    # Your code here:
    
    plt.figure(figsize=(8, 6), dpi=80)
    size = len(lamda)
    x = list(range(size))
    plt.plot(x, s_w, label='Trend', marker='d')
    _=plt.xticks(x, lamda)
    plt.xlabel("Lambda")
    plt.ylabel("Number of Sparse weights")
    plt.legend()
    plt.title("Sparse weights as per Lambda values")
       
    return


# Part 2 a&c. Generates plots 

# In[18]:


weight_tr = np.zeros((x_train.shape[1],1))
weight_val = np.zeros((x_train.shape[1],1))
train_acc_temp = []
valid_acc_temp = []
sw_temp = []
for i in range(7):
    lr = learned_lr[i]
    lm = lamda[i]
    weight = np.random.randn(x_train.shape[1],1)
    acc_lamda[lm]["accuracy1"], weight_tr, sparse_weights_lm[lm]["sparse_weights"], max_weights[lm]['weights'] = LR_L2_train(x_train, y_train, weight, lamda = lm, learning_rate=lr, store_w=True)
    train_acc_temp.append(acc_lamda[lm]["accuracy1"])
    sw_temp.append(sparse_weights_lm[lm]["sparse_weights"])
    acc_lamda[lm]["accuracy2"] = LR_L2_train(x_valid, y_valid, weight_tr, lamda = lm, learning_rate=lr, store_w=False)
    valid_acc_temp.append(acc_lamda[lm]["accuracy2"])
    
train_acc = np.array(train_acc_temp)
valid_acc = np.array(valid_acc_temp)
s_w = np.array(sw_temp)
plot_accuracy(train_acc, lamda, label="Training accuracy vs Lambda Using Batch Gradient Descent")
plot_accuracy(valid_acc, lamda, label="Validation accuracy vs Lambda Using Batch Gradient Descent")
plot_sparse_weights(s_w, lamda)


# Part 2-b. Prints the features of best performing lambdas.

# In[19]:


features = preprocessed_data.columns
for i in range(3):
    largest_weights = max_weights[10**-i]['weights']
    for w, columns in sorted(zip(largest_weights, features), reverse=True):
        print(columns, w)


# # Part 2

# In[20]:


loaded_data = load_data("IA2-train-noisy.csv")
preprocessed_data, mean_std2 = preprocess_data(loaded_data, True)


# In[21]:


labels = preprocessed_data['Response']
x_train= preprocessed_data.drop(['Response'] , axis=1)
y_train = labels.to_numpy()


# In[22]:


loaded_data = load_data("IA2-dev.csv")
validation_data = normalizeValidationData(loaded_data, mean_std2)
labels_valid = validation_data['Response']
x_valid= validation_data.drop(['Response'] , axis=1)


# In[23]:


lamda2 = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
learned_lr2 = [1, 1, 1, 1, 1, 0.1, 0.01]
sparse_weights_lm2 = {lamda2[i]:{'sparse_weights':[]} for i in range(7)}
acc_lamda2 = {lamda2[i]:{'lr_lambda':learned_lr2[i], 'accuracy1':[], 'accuracy2':[]} for i in range(7)}
max_weights2 = {lamda2[i]:{'weights':[]} for i in range(7)}


# In[24]:


weight_tr2 = np.zeros((x_train.shape[1],1))
weight_val2 = np.zeros((x_train.shape[1],1))
train_acc_temp2 = []
valid_acc_temp2 = []


# In[25]:


for i in range(7):
    lr = learned_lr2[i]
    lm = lamda2[i]
    weight2 = np.random.randn(x_train.shape[1],1)
    acc_lamda2[lm]["accuracy1"], weight_tr2, sparse_weights_lm2[lm]["sparse_weights"], max_weights2[lm]['weights'] = LR_L2_train(x_train, y_train, weight, lamda = lm, learning_rate=lr, store_w=True)
    train_acc_temp2.append(acc_lamda2[lm]["accuracy1"])
    acc_lamda2[lm]["accuracy2"] = LR_L2_train(x_valid, y_valid, weight_tr2, lamda = lm, learning_rate=lr, store_w=False)
    valid_acc_temp2.append(acc_lamda2[lm]["accuracy2"])
    
train_acc2 = np.array(train_acc_temp2)
valid_acc2 = np.array(valid_acc_temp2)
plot_accuracy(train_acc2, lamda2, label="Training accuracy vs Lambda Using Batch Gradient Descent")
plot_accuracy(valid_acc2, lamda2, label="Validation accuracy vs Lambda Using Batch Gradient Descent")


# # Part 3

# In[26]:


loaded_data = load_data("IA2-train.csv")
preprocessed_data, mean_std = preprocess_data(loaded_data, True)


# In[27]:


labels = preprocessed_data['Response']
x_train= preprocessed_data.drop(['Response'] , axis=1)
y_train = labels.to_numpy()


# In[28]:


loaded_data = load_data("IA2-dev.csv")
validation_data = normalizeValidationData(loaded_data, mean_std)


# In[29]:


labels_valid = validation_data['Response']
x_valid= validation_data.drop(['Response'] , axis=1)
y_valid = labels_valid.to_numpy()


# In[30]:


lamda3 = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
learned_lr3 = [1, 1, 1, 0.1, 1, 0.1, 10**-5]
sparse_weights_lm3 = {lamda3[i]:{'sparse_weights':[]} for i in range(7)}
acc_lamda3 = {lamda3[i]:{'lr_lambda':learned_lr3[i], 'accuracy1':[], 'accuracy2':[]} for i in range(7)}
max_weights3 = {lamda3[i]:{'weights':[]} for i in range(7)}


# In[31]:


def LR_L1_train(x, y, weight, lamda, learning_rate, store_w):
         
    exmpl_size, no_features = x.shape
    iterations = 1500 
    
    # Initializing weights 
    weight = weight
               
    # Reshaping y
    y = y.reshape(exmpl_size,1)
    
    #list to store losses
    all_accuracies = []
    
    # Training loop.
    for iterations in range(iterations):
            
        # Calculating hypothesis
        hypo = sigmoid_z(np.dot(x, weight))
        
        # gradients of loss
        if (abs(weight).all() > (learning_rate*lamda)):
            new_weight = batch_gradients(x, y, hypo) + np.sign(weight)*(np.abs(weight) - (learning_rate*lamda))
        else:
            new_weight = batch_gradients(x, y, hypo)
            
        # Updating weight 
        weight -= new_weight
        
        # Calculating loss 
        current_loss = loss_func(y, hypo) + (lamda * np.sum(np.abs(weight)))
        
       
    # Get accuracy
    all_accuracies.append(accuracy_func(weight, x, y, hypo)) 
    no_of_sparseweights = np.where(weight<10**-6)[0].shape[0] 
   
    # return values
    if(store_w):
        return all_accuracies, weight, no_of_sparseweights, abs(weight) 
    else:
        return all_accuracies 


# Part 3 - a & c

# In[32]:


weight_tr3 = np.zeros((x_train.shape[1],1))
weight_val3 = np.zeros((x_train.shape[1],1))
train_acc_temp3 = []
valid_acc_temp3 = []
sw_temp3 = []
for i in range(7):
    lr = learned_lr3[i]
    lm = lamda3[i]
    weight3 = np.random.randn(x_train.shape[1],1)
    acc_lamda3[lm]["accuracy1"], weight_tr3, sparse_weights_lm3[lm]["sparse_weights"], max_weights3[lm]['weights'] = LR_L1_train(x_train, y_train, weight, lamda = lm, learning_rate=lr, store_w=True)
    train_acc_temp3.append(acc_lamda3[lm]["accuracy1"])
    sw_temp3.append(sparse_weights_lm3[lm]["sparse_weights"])
    acc_lamda3[lm]["accuracy2"] = LR_L1_train(x_valid, y_valid, weight_tr, lamda = lm, learning_rate=lr, store_w=False)
    valid_acc_temp3.append(acc_lamda3[lm]["accuracy2"])
    
train_acc3 = np.array(train_acc_temp3)
valid_acc3 = np.array(valid_acc_temp3)
s_w3 = np.array(sw_temp3)
plot_accuracy(train_acc3, lamda3, label="Training accuracy vs Lambda Using Batch Gradient Descent")
plot_accuracy(valid_acc3, lamda3, label="Validation accuracy vs Lambda Using Batch Gradient Descent")
plot_sparse_weights(s_w3, lamda3)


# Part 3 - b. Prints the features of best performing lambdas.

# In[34]:


features3 = preprocessed_data.columns
for i in range(0,4):
    largest_weights = max_weights3[10**-i]['weights']
    for w, columns in sorted(zip(largest_weights, features3), reverse=True):
        print(columns, w)


# In[ ]:




