import numpy as np
import matplotlib.pyplot as plt

# 加载数据
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

# 合并数据
X = np.vstack((X1, X2))
Y = np.hstack((np.zeros(len(X1)), np.ones(len(X2))))


class MLP:
    def __init__(self):
        # 初始化权重和偏置
        self.W1 = np.random.randn(2, 3)  # 输入层到隐藏层的权重
        self.b1 = np.random.randn(3)     # 隐藏层的偏置
        self.W2 = np.random.randn(3, 1)  # 隐藏层到输出层的权重
        self.b2 = np.random.randn(1)     # 输出层的偏置

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # 对于sigmoid，导数可以用输出值计算

    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # 隐藏层使用ReLU激活
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # 输出层使用sigmoid激活
        return self.a2
    
    def backward(self, X, Y, learning_rate=0.001):
        # 反向传播
        m = Y.size  # 样本数量

        # 计算输出层的误差
        output_error = self.a2 - Y.reshape(-1, 1)  # 预测值与真实值的差
        output_delta = output_error * self.sigmoid_derivative(self.a2)  # 输出层的梯度

        # 更新输出层权重和偏置
        self.W2 -= learning_rate * np.dot(self.a1.T, output_delta) / m
        self.b2 -= learning_rate * np.sum(output_delta, axis=0) / m

        # 计算隐藏层的误差
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.relu_derivative(self.a1)  # 隐藏层的梯度

        # 更新隐藏层权重和偏置
        self.W1 -= learning_rate * np.dot(X.T, hidden_delta) / m
        self.b1 -= learning_rate * np.sum(hidden_delta, axis=0) / m

    def train(self, X, Y, epochs=1000):
        for _ in range(epochs):
            self.forward(X)  # 前向传播
            self.backward(X, Y)  # 反向传播

    def predict(self, X):
        # 预测
        return (self.forward(X) > 0.5).astype(int)
    
    def accuracy(self, X, Y):
        # 计算准确率
        predictions = self.predict(X)  # 获取预测结果
        correct_predictions = np.sum(predictions.flatten() == Y)  # 计算正确预测的数量
        accuracy = correct_predictions / len(Y)  # 计算准确率
        return accuracy

# 创建MLP实例
mlp = MLP()

# 打印初始权重和偏置
print("Initial weights and biases:")
print("W1:", mlp.W1)
print("b1:", mlp.b1)
print("W2:", mlp.W2)
print("b2:", mlp.b2)

# train
mlp.train(X, Y, epochs=10000)

# 打印训练后的权重和偏置
print("Trained weights and biases:")
print("W1:", mlp.W1)
print("b1:", mlp.b1)
print("W2:", mlp.W2)
print("b2:", mlp.b2)

# 计算训练后的准确率
accuracy = mlp.accuracy(X, Y)
print(f"Classification accuracy after training: {accuracy:.4f}")

# 绘制决策边界
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