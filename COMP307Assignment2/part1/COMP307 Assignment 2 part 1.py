#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as mlp

class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def sigmoid(self, x):
        output = 1/(1+np.exp(-x)) # Sigmoid function
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs, p=False):
        hidden_layer_outputs = self.sigmoid(np.dot(inputs, self.hidden_layer_weights))
        output_layer_outputs = self.sigmoid(np.dot(hidden_layer_outputs, self.output_layer_weights))
        if p == True:
            print("Hidden Layer Outputs: ", hidden_layer_outputs)
            print("Outer Layer Outputs: ", output_layer_outputs)
        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs, p=False):
        
        output_layer_betas = []
        for i in range(self.num_outputs):
            output_layer_betas.append(desired_outputs[i] - output_layer_outputs[i])
        hidden_layer_betas = []
        for i in range(self.num_hidden):
            hidden_layer_beta = 0
            for j in range(self.num_outputs):
                hidden_layer_beta = hidden_layer_beta + self.output_layer_weights[i][j] * output_layer_betas[j] * output_layer_outputs[j] * (1 - output_layer_outputs[j])
            hidden_layer_betas.append(hidden_layer_beta)
#             print(hidden_layer_beta)
        if p == True:
            print("Output betas: ", output_layer_betas)
            print("Hidden betas: ", hidden_layer_betas)
        # This is a HxO array (2 hidden nodes, 3 outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        # TODO! Calculate output layer weight changes.
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                delta_output_layer_weights[i][j] = self.learning_rate * hidden_layer_outputs[i] * \
                output_layer_outputs[j] * (1 - output_layer_outputs[j]) * output_layer_betas[j]
        # This is a IxH array (4 inputs, 2 hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        # TODO! Calculate hidden layer weight changes.
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                delta_hidden_layer_weights[i][j] = self.learning_rate * inputs[i] * \
                hidden_layer_outputs[j] * (1 - hidden_layer_outputs[j]) * hidden_layer_betas[j]
        
        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        self.hidden_layer_weights += delta_hidden_layer_weights
        self.output_layer_weights += delta_output_layer_weights

    def train(self, instances, desired_outputs, epochs, p):
        accArrays = []
        for epoch in range(epochs):
            print('epoch = ', epoch+1)
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance, p)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i], p)
                predicted_class = output_layer_outputs
                predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print new weights
            print('Hidden layer weights \n', self.hidden_layer_weights)
            print('Output layer weights  \n', self.output_layer_weights)

            acc = 0
            for i in range(len(predictions)-1):
                if len(predictions) != len(desired_outputs):
                    if np.argmax(desired_outputs[i]) == np.argmax(predictions):
                        acc+=1
                else:
                    if np.argmax(desired_outputs[i]) == np.argmax(predictions[i]):
                        acc+=1
            acc = acc/len(desired_outputs) * 100
            accArrays.append(acc)
            print('acc = ', acc)
        if p == True:
            print("Predictions are:", predictions)
        mlp.plot(range(epochs), accArrays)
            

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            predicted_class = np.argmax(output_layer_outputs)
            predictions.append(predicted_class)
        return predictions

class Neural_Network_Biases:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, biases, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights
        self.biases = biases
        self.hidden_biases = biases[:num_hidden]
        self.output_biases = biases[num_hidden:(num_hidden+num_outputs)]
        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def sigmoid(self, x):
        output = 1/(1+np.exp(-x)) # Sigmoid function
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = self.sigmoid(np.dot(inputs, self.hidden_layer_weights) + self.hidden_biases)
        output_layer_outputs = self.sigmoid(np.dot(hidden_layer_outputs, self.output_layer_weights) + self.output_biases)
        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):
        
        output_layer_betas = []
        for i in range(self.num_outputs):
            output_layer_betas.append(desired_outputs[i] - output_layer_outputs[i])
        hidden_layer_betas = []
        for i in range(self.num_hidden):
            hidden_layer_beta = 0
            for j in range(self.num_outputs):
                hidden_layer_beta = hidden_layer_beta + self.output_layer_weights[i][j] * output_layer_betas[j] * output_layer_outputs[j] * (1 - output_layer_outputs[j])
            hidden_layer_betas.append(hidden_layer_beta)
#             print(hidden_layer_beta)
        # This is a HxO array (2 hidden nodes, 3 outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        # TODO! Calculate output layer weight changes.
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                delta_output_layer_weights[i][j] = self.learning_rate * hidden_layer_outputs[i] * \
                output_layer_outputs[j] * (1 - output_layer_outputs[j]) * output_layer_betas[j]
        # This is a IxH array (4 inputs, 2 hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        # TODO! Calculate hidden layer weight changes.
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                delta_hidden_layer_weights[i][j] = self.learning_rate * inputs[i] * \
                hidden_layer_outputs[j] * (1 - hidden_layer_outputs[j]) * hidden_layer_betas[j]
        hiddenBoo = hidden_layer_outputs[1] > hidden_layer_betas[1]
        # if hiddenBoo == True:
        self.output_biases += self.learning_rate * (output_layer_betas * \
            output_layer_outputs * (1-output_layer_outputs))
        self.hidden_biases += self.learning_rate * (hidden_layer_betas * \
            hidden_layer_outputs * (1-hidden_layer_outputs))
        # else:
        #     self.hidden_biases = self.biases[:self.num_hidden]
        #     self.output_biases = self.biases[self.num_hidden:(self.num_hidden+self.num_outputs)]
        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        self.hidden_layer_weights += delta_hidden_layer_weights
        self.output_layer_weights += delta_output_layer_weights

    def train(self, instances, desired_outputs, epochs, p):
        accArrays = []
        for epoch in range(epochs):
            print('epoch = ', epoch+1)
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
                predicted_class = output_layer_outputs
                predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print new weights
            print('Hidden layer weights \n', self.hidden_layer_weights)
            print('Output layer weights  \n', self.output_layer_weights)

            acc = 0
            for i in range(len(predictions)-1):
                if len(predictions) != len(desired_outputs):
                    if np.argmax(desired_outputs[i]) == np.argmax(predictions):
                        acc+=1
                else:
                    if np.argmax(desired_outputs[i]) == np.argmax(predictions[i]):
                        acc+=1
            acc = acc/len(desired_outputs) * 100
            accArrays.append(acc)
            print('acc = ', acc)
        if p == True:
            print("Predictions are:", predictions)
        mlp.plot(range(epochs), accArrays)
        
            

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            predicted_class = np.argmax(output_layer_outputs)
            predictions.append(predicted_class)
        return predictions


# In[2]:


def encode_labels(labels):
    # encode 'Adelie' as 1, 'Chinstrap' as 2, 'Gentoo' as 3
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # don't worry about this
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    # encode 1 as [1, 0, 0], 2 as [0, 1, 0], and 3 as [0, 0, 1] (to fit with our network outputs!)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return label_encoder, integer_encoded, onehot_encoder, onehot_encoded


# Note: The skeleton code was indeed used

# In[3]:


if __name__ == '__main__':
    file = sys.argv[1]
    data = pd.read_csv(file)
#     the class label is last!
    labels = data.iloc[:, -1]
#     seperate the data from the labels
    instances = data.iloc[:, :-1]
#     scale features to [0,1] to improve training
    scaler = MinMaxScaler()
    instances = scaler.fit_transform(instances)
    # We can't use strings as labels directly in the network, so need to do some transformations
    label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(labels)
    labels = onehot_encoded

    # Parameters. As per the handout.
    n_in = 4
    n_hidden = 2
    n_out = 3
    learning_rate = 0.2

    initial_hidden_layer_weights = np.array([[-0.28, -0.22], [0.08, 0.20], [-0.30, 0.32], [0.10, 0.01]])
    initial_output_layer_weights = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36]])
    
    nn = Neural_Network(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                        learning_rate)

    print('First instance has label {}, which is {} as an integer, and {} as a list of outputs.\n'.format(
        labels[0], integer_encoded[0], onehot_encoded[0]))
    
    # need to wrap it into a 2D array
    instance1_prediction = nn.predict(np.array([instances[0]]))
    print(instance1_prediction)
    if instance1_prediction[0] is None:
        # This should never happen once you have implemented the feedforward.
        instance1_predicted_label = "???"
    else:
        instance1_predicted_label = label_encoder.inverse_transform(instance1_prediction)
    print('Predicted label for the first instance is: {}\n'.format(instance1_predicted_label))

#     TODO: Perform a single backpropagation pass using the first instance only. (In other words, train with 1
#      instance for 1 epoch!). Hint: you will need to first get the weights from a forward pass.
    nn.train([instances[0]], labels, 1, p=True)
    
    print('Weights after performing BP for first instance only:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)
    


# In[4]:


# TODO: Train for 100 epochs, on all instances.
nn = Neural_Network(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                        learning_rate)
nn.train(instances, labels, 100, False)
print('\nAfter training:')
print('Hidden layer weights:\n', nn.hidden_layer_weights)
print('Output layer weights:\n', nn.output_layer_weights)


# In[5]:

file1 = sys.argv[2]
pd_data_ts = pd.read_csv(file1)
test_labels = pd_data_ts.iloc[:, -1]
test_instances = pd_data_ts.iloc[:, :-1]
#scale the test according to our training data.
# TODO: Compute and print the test accuracy
test_instances = scaler.transform(test_instances)
label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(test_labels)
test_labels = onehot_encoded
acc= 0
predictions = nn.predict(test_instances) # using the same model as before
acc = sum(predictions == np.argmax(test_labels, axis=1)) / len(test_labels) * 100
print('Test Accuracy is: ', acc)


# In[6]:


n_in = 4
n_hidden = 2
n_out = 3
learning_rate = 0.2
biases = np.array([-0.02, -0.20, -0.33, 0.26, 0.06])
initial_hidden_layer_weights = np.array([[-0.28, -0.22], [0.08, 0.20], [-0.30, 0.32], [0.10, 0.01]])
initial_output_layer_weights = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36]])

nnBias = Neural_Network_Biases(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,biases,learning_rate)


# In[7]:


nnBias.train(instances, labels, 100, False)
print('\nAfter training:')
print('Hidden layer weights:\n', nnBias.hidden_layer_weights)
print('Output layer weights:\n', nnBias.output_layer_weights)


# In[8]:


predictions = nnBias.predict(test_instances) # using the same model as before
acc = sum(predictions == np.argmax(test_labels, axis=1)) / len(test_labels) * 100
print('Test Accuracy is: ', acc)


# In[9]:


nnBiasLr2 =  Neural_Network_Biases(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,biases,0.02)


# In[10]:


nnBiasLr2.train(instances, labels, 100, False)
print('\nAfter training:')
print('Hidden layer weights:\n', nnBias.hidden_layer_weights)
print('Output layer weights:\n', nnBias.output_layer_weights)


# In[11]:


predictions = nnBiasLr2.predict(test_instances) # using the same model as before
acc = sum(predictions == np.argmax(test_labels, axis=1)) / len(test_labels) * 100
print('Test Accuracy is: ', acc)


# In[12]:


nnBiasLr3 =  Neural_Network_Biases(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,biases,0.1)


# In[13]:


nnBiasLr3.train(instances, labels, 100, False)
print('\nAfter training:')
print('Hidden layer weights:\n', nnBias.hidden_layer_weights)
print('Output layer weights:\n', nnBias.output_layer_weights)


# In[14]:


predictions = nnBiasLr3.predict(test_instances) # using the same model as before
acc = sum(predictions == np.argmax(test_labels, axis=1)) / len(test_labels) * 100
print('Test Accuracy is: ', acc)


# In[15]:
nnBiasLr4 =  Neural_Network_Biases(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,biases,0.5)

# In[16]:
nnBiasLr4.train(instances, labels, 100, False)
print('\nAfter training:')
print('Hidden layer weights:\n', nnBias.hidden_layer_weights)
print('Output layer weights:\n', nnBias.output_layer_weights)

# In[17]:
predictions = nnBiasLr4.predict(test_instances) # using the same model as before
acc = sum(predictions == np.argmax(test_labels, axis=1)) / len(test_labels) * 100
print('Test Accuracy is: ', acc)

# In[18]:
nnBiasLr5 =  Neural_Network_Biases(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,biases,0.001)

# In[19]:

nnBiasLr5.train(instances, labels, 100, False)
print('\nAfter training:')
print('Hidden layer weights:\n', nnBias.hidden_layer_weights)
print('Output layer weights:\n', nnBias.output_layer_weights)

# In[20]:

predictions = nnBiasLr5.predict(test_instances) # using the same model as before
acc = sum(predictions == np.argmax(test_labels, axis=1)) / len(test_labels) * 100
print('Test Accuracy is: ', acc)