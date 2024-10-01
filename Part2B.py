import numpy as np
import matplotlib.pyplot as plt

# load data
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

# merge data
X = np.vstack((X1, X2))
Y = np.hstack((np.zeros(len(X1)), np.ones(len(X2))))

class RBFNetwork:
    def __init__(self, n_hidden=6, sigma=1):
        self.n_hidden = n_hidden
        self.sigma = sigma
        # random select RBF centers
        self.centers = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (n_hidden, X.shape[1]))
        self.weights = np.zeros((n_hidden, 1))  # initialize output layer weights

    def gaussian(self, x, center):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * self.sigma ** 2))

    def forward(self, X):
        # calculate hidden layer output
        self.hidden_output = np.array([[self.gaussian(x, center) for center in self.centers] for x in X])
        return self.hidden_output

    def fit(self, X, Y):
        # calculate hidden layer output
        hidden_output = self.forward(X)
        # calculate weights using least squares method
        self.weights = np.linalg.pinv(hidden_output).dot(Y.reshape(-1, 1))
        print("Calculated weights from hidden layer to output layer:")
        print(self.weights)  # print weights

    def predict(self, X):
        hidden_output = self.forward(X)
        output = hidden_output.dot(self.weights)
        return (output > 0.5).astype(int)  

    def accuracy(self, X, Y):
        predictions = self.predict(X)
        return np.mean(predictions.flatten() == Y)

# RBF network instance
rbf_network = RBFNetwork(n_hidden=6, sigma=1)

# train model
rbf_network.fit(X, Y)

# print RBF centers
print("RBF Centers:")
print(rbf_network.centers)

# calculate accuracy after training
accuracy = rbf_network.accuracy(X, Y)
print(f"Classification accuracy after training: {accuracy:.4f}")

# plot RBF centers
plt.scatter(rbf_network.centers[:, 0], rbf_network.centers[:, 1], c='red', marker='x', s=100, label='RBF Centers')
plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.5, cmap='bwr')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("RBF Centers and Data Points")
plt.legend()
plt.show()

# plot decision boundary
def plot_decision_boundary(X, Y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.8, cmap='bwr')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary of RBF Network")
    plt.show()

plot_decision_boundary(X, Y, rbf_network)