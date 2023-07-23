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