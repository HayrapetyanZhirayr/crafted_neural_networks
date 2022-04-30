import numpy as np

class Layer:

    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        return input

class ReLU(Layer):

    def __init__(self, ):
        pass

    def forward(self, input):
        output = np.maximum(input, 0)
        return output

    def backward(self, input, grad_output):
        relu_grad = input > 0  #  dinput / doutput
        return grad_output * relu_grad  # dLoss / dinput


class Dense(Layer):

    def __init__(self, input_units, output_units, lr=.1):
        self.lr = lr

        self.weights = np.random.randn(input_units, output_units)
        self.biases = np.zeros(output_units)

    def update_weights(self, grad_weights, grad_biases):
        self.weights = self.weights - self.lr*grad_weights
        self.biases = self.biases - self.lr*grad_biases

    def forward(self, input):
        output = input@self.weights + self.biases
        return output

    def backward(self, input, grad_output):
        grad_input = grad_output@self.weights.T  # dLoss / dinput
        grad_weights = input.T@grad_output  # dLoss /dweights
        grad_biases = np.sum(grad_output, axis=0)  # dLoss / dbias  # summation over batch ax

        self.update_weights(grad_weights, grad_biases)

        return grad_input



class LogSoftMax(Layer):

    def __init__(self, ):
        pass

    def forward(self, logits, labels):
        true_label_logits = logits[np.arange(len(labels)), labels]
        log_sum_logits = np.log(np.sum(np.exp(logits), axis=-1))  # summation over logits axis
        logsoftmax = - true_label_logits + log_sum_logits
        return np.mean(logsoftmax, axis=0)  # mean over batch axis


    def backward(self, logits, labels):
        softmax = np.exp(logits) / np.sum(np.exp(logits),axis=-1, keepdims=True) # summation over logits axis
        ones_for_labels = np.zeros_like(logits)
        ones_for_labels[np.arange(len(labels)), labels] = 1

        grad_logits = -ones_for_labels + softmax

        return grad_logits  # dLoss / dlogits


class NNetwork:

    def __init__(self, layers, loss):
        # layers :: list<Layer>
        # loss :: layer
        self.layers = layers
        self.loss = loss
        self.activations = []

    def forward(self, X):

        input = X
        layer_inputs = [input]
        for layer in self.layers:
            input = layer.forward(input)
            layer_inputs.append(input)

        logits = layer_inputs.pop(-1)

        return layer_inputs, logits

    def compute_loss(self, logits, labels):
        loss_value = self.loss.forward(logits, labels)  # END OF FORWARD
        grad_logits = self.loss.backward(logits, labels)  # START OF BACKWARD
        return loss_value, grad_logits

    def backward(self, layer_inputs, grad_logits):

        grad_output = grad_logits
        for input, layer in reversed(list(zip(layer_inputs, self.layers))):
            grad_output = layer.backward(input, grad_output)


    def predict(self, X):
        _, logits = self.forward(X)
        return np.argmax(logits, axis=-1)
