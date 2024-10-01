import numpy as np
import matplotlib.pyplot as plt

# load data
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

# merge data
X = np.vstack((X1, X2))
Y = np.hstack((np.zeros(len(X1)), np.ones(len(X2))))


class MLP:
    def __init__(self):
        # initialize weights and biases
        self.W1 = np.random.randn(2, 3)  # weights from input layer to hidden layer
        self.b1 = np.random.randn(3)     # biases of hidden layer
        self.W2 = np.random.randn(3, 1)  # weights from hidden layer to output layer
        self.b2 = np.random.randn(1)     # biases of output layer

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # derivative of sigmoid function

    # forward propagation
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # hidden layer activation(with ReLU)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # Output layer activation(with sigmoid)
        return self.a2
    
    # backward propagation
    def backward(self, X, Y, learning_rate=0.001):
        
        m = Y.size  # number of samples

        # calculate output layer error
        output_error = self.a2 - Y.reshape(-1, 1)  # difference between prediction and true value
        output_delta = output_error * self.sigmoid_derivative(self.a2)  # gradient of output layer

        # update output layer weights and biases
        self.W2 -= learning_rate * np.dot(self.a1.T, output_delta) / m
        self.b2 -= learning_rate * np.sum(output_delta, axis=0) / m

        # calculate hidden layer error
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.relu_derivative(self.a1)  # gradient of hidden layer

        # update hidden layer weights and biases
        self.W1 -= learning_rate * np.dot(X.T, hidden_delta) / m
        self.b1 -= learning_rate * np.sum(hidden_delta, axis=0) / m

    # training
    def train(self, X, Y, epochs=1000):
        for _ in range(epochs):
            self.forward(X)  # forward propagation
            self.backward(X, Y)  # backward propagation

    # predict
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)
    
    # calculate accuracy
    def accuracy(self, X, Y):
        predictions = self.predict(X)  # get prediction results
        correct_predictions = np.sum(predictions.flatten() == Y)  # calculate the number of correct predictions
        accuracy = correct_predictions / len(Y)  # calculate accuracy
        return accuracy

# MLP instance
mlp = MLP()

# print initial weights and biases
print("Initial weights and biases:")
print("W1:", mlp.W1)
print("b1:", mlp.b1)
print("W2:", mlp.W2)
print("b2:", mlp.b2)

# train
mlp.train(X, Y, epochs=10000)

# print trained weights and biases
print("Trained weights and biases:")
print("W1:", mlp.W1)
print("b1:", mlp.b1)
print("W2:", mlp.W2)
print("b2:", mlp.b2)

# calculate accuracy after training
accuracy = mlp.accuracy(X, Y)
print(f"Classification accuracy after training: {accuracy:.4f}")

# plot decision boundary
def plot_decision_boundary(X, Y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.8, cmap='bwr', edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary of Trained MLP")
    plt.show()

plot_decision_boundary(X, Y, mlp)