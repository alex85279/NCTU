#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

loss_record = []
epoch_record = []

# In[3]:


def show_result(x, y, pred_y):
    pred_y = np.round(pred_y)
    pt = pred_y[:,0]
    gt = y[:,0]
    total_element = len(gt)
    num_equal_element = 0
    for i in range(total_element):
        if pt[i] == gt[i]:
            num_equal_element = num_equal_element + 1
    
    print(f"Accuracy: {num_equal_element / total_element}")
    cm = LinearSegmentedColormap.from_list(
        'mymap', [(1, 0, 0), (0, 0, 1)], N=2)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    plt.scatter(x[:,0], x[:,1], c=y[:,0], cmap=cm)
    
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    plt.scatter(x[:,0], x[:,1], c=pred_y[:,0], cmap=cm)
    
    plt.show()


# In[4]:


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy(n=11):
    inputs = []
    labels = []
    step = 1/(n-1)
    for i in range(n):
        inputs.append([step*i, step*i])
        labels.append(0)
        
        if i == int((n-1)/2):
            continue
        
        inputs.append([step*i, 1 - step*i])
        labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(n*2 - 1,1)




# In[5]:

#activation funcitons
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# y = sigmoid(x)
def derivative_sigmoid(y):
    return np.multiply(y, 1.0 - y)
    
def sinh(x):    
    return 
    return (np.exp(x) - np.exp(-1 * x))/2
    
def cosh(x):
    return (np.exp(x) + np.exp(-1 * x))/2


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def derivative_tanh(y):
    return 1 - np.multiply(y,y)
    
def relu(x):
    return np.maximum(0,x)

def derivative_relu(x):
    x[x <= 0] = 0
    x[x >  0] = 1
    return x
    

def leaky_relu(x):
    return np.maximum(0,x) + np.minimum(0, 0.01 * x)

def derivative_leaky_relu(x):
    x[x <= 0] = 0.01
    x[x >  0] = 1
    return x
    
# In[6]:


def MSE(y, y_hat):
    return np.mean((y - y_hat)**2)
    #return np.square(y - y_hat).mean()

def D_MSE(y,y_hat):
    return 2 * (y - y_hat) / y.shape[0]


    
# In[7]:


class layer():
    def __init__(self, input_cell, output_cell):
        #input cell + 1 for bias
        layer.inSize = input_cell
        layer.outSize = output_cell
        self.weight = np.random.normal(0,1,(input_cell + 1, output_cell))
        self.momentum = 0
        self.n = 0
        self.mt = 0
        self.vt = 0
        self.b1 = 0.9
        self.b2 = 0.999
        self.t = 1
        
        
    def forward(self, layer_in):
        #add 1 to multiply with bias
        layer_in = np.append(layer_in, np.ones((layer_in.shape[0],1)), axis=1)
        #with activation sigmoid
        self.layer_out = sigmoid(np.matmul(layer_in, self.weight))
        
        #with activation tanh
        #self.layer_out = tanh(np.matmul(layer_in, self.weight))
        
        #with activation relu
        '''if layer.inSize == 4 and layer.outSize == 1:
            self.layer_out = sigmoid(np.matmul(layer_in, self.weight))
        else:
            self.layer_out = relu(np.matmul(layer_in, self.weight))'''
        
        #with activation leaky-relu
        '''if layer.inSize == 4 and layer.outSize == 1:
            self.layer_out = sigmoid(np.matmul(layer_in, self.weight))
        else:
            self.layer_out = leaky_relu(np.matmul(layer_in, self.weight))'''
       
        
        #without activation
        '''if layer.inSize == 4 and layer.outSize == 1:
            self.layer_out = sigmoid(np.matmul(layer_in, self.weight))
        else:
            self.layer_out = np.matmul(layer_in, self.weight)'''
        #self.layer_out = np.matmul(layer_in, self.weight)
        
        #store forward pass here for gradient
        self.forward_pass = layer_in
        return self.layer_out
    
    def backward(self, derivate_loss):
        #backward pass = (derivative Loss)dC/d(aL) * (derivative activation function)(f(L))'
        #loss_value = backward pass * weight_transpose(without bias)
        #with activation sigmoid
        
        self.backward_pass = derivate_loss * derivative_sigmoid(self.layer_out)
        
        #with activation tanh
        #self.backward_pass = derivate_loss * derivative_tanh(self.layer_out)
        #with activation leaky relu
        '''if layer.inSize == 4 and layer.outSize == 1:
            self.backward_pass = derivate_loss * derivative_sigmoid(self.layer_out)
            
        else:
            self.backward_pass = derivate_loss * derivative_leaky_relu(self.layer_out)'''
        
        
        #with activation relu
        '''if layer.inSize == 4 and layer.outSize == 1:
            self.backward_pass = derivate_loss * derivative_sigmoid(self.layer_out)
            
        else:
            self.backward_pass = derivate_loss * derivative_relu(self.layer_out)'''
        
        #without activation
        '''if layer.inSize == 4 and layer.outSize == 1:
            self.backward_pass = derivate_loss * derivative_sigmoid(self.layer_out)
            
        else:
            self.backward_pass = derivate_loss'''
        #self.backward_pass = derivate_loss
        return np.matmul(self.backward_pass, self.weight[:-1].transpose())
    def update(self, lr):
        #gradient = forward_pass_transpose * backward_pass
        #new_weight = weight - learning_rate * gradient
        self.gradient = np.matmul(self.forward_pass.transpose(), self.backward_pass)
        #gde
        #self.weight = self.weight - lr * self.gradient
        
        #momentum
        #self.momentum = 0.9 * self.momentum - lr * self.gradient
        #self.weight = self.weight + self.momentum
        
        #adagrad
        #self.n += np.square(self.gradient)
        #n_lr = np.divide(lr, np.sqrt(self.n + 1e-8))
        #self.weight = self.weight - n_lr * self.gradient
        
        #adam
        self.mt = self.b1 * self.mt + (1 - self.b1) * self.gradient
        self.vt = self.b1 * self.vt + (1 - self.b2) * np.square(self.gradient)
        mt_hat = np.divide(self.mt, 1 - self.b1 ** self.t)
        vt_hat = np.divide(self.vt, 1 - self.b2 ** self.t)
        self.t += 1
        self.weight = self.weight - lr * np.divide(mt_hat, np.sqrt(vt_hat) + 1e-8)

# In[8]:


#layer test block
'''l = layer(2,1)
y = l.forward(x1)
loss = MSE(y,y1)
back = l.backward(D_MSE(y,y1))
up = l.update(0.001);
y'''


# In[9]:


lay = [2,4,4,1]
for i in range(len(lay)-1):
    
    pair = [lay[i], lay[i+1]]
    print(pair)
    


# In[10]:


class net():
    def __init__(self, layers_unit, lr = 0.01):
        self.lr = lr
        self.total_layer = []
        for i in range(len(layers_unit)-1):
            self.total_layer += [layer(layers_unit[i], layers_unit[i+1])]
    
    def forward(self, network_input):
        network_output = network_input
        for lay in self.total_layer:
            network_output = lay.forward(network_output)
            
        return network_output
    
    def backward(self, derivative_loss):
        tmp_d_loss = derivative_loss
        for lay in self.total_layer[::-1]:
            tmp_d_loss = lay.backward(tmp_d_loss)
            
    def update(self):
        for lay in self.total_layer:
            lay.update(self.lr)
    
    def train(self, epoch, network_input, ground_truth):
        for i in range(epoch):
            loss_thresh = 0.01
            show_info_rate = 1;
            network_output = self.forward(network_input)
            loss = MSE(network_output, ground_truth)
            
            self.backward(D_MSE(network_output, ground_truth))
            self.update()
            global loss_record
            global epoch_record
            
            if(loss <= loss_thresh):
                break
            if i % show_info_rate == 0: 
                print(f"epoch: {i}  loss: {loss}")
                loss_record += [loss]
                epoch_record += [i]
            
        


# In[11]:
#generate data
x1, y1 = generate_linear()
x2, y2 = generate_XOR_easy()
#generate net
nn = net([2,4,4,1], lr = 0.1)


# In[12]:

#test the forward output before training
#ty = nn.forward(x1)



# In[13]:

#max epoch = 1000000, loss threshold = 0.01
input_x = x2
input_y = y2
nn.train(1000000,input_x,input_y)


# In[14]:


result = nn.forward(input_x)
show_result(input_x,input_y,result)
plt.plot(epoch_record, loss_record)
plt.show()


