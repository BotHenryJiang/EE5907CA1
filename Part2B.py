import numpy as np
import matplotlib.pyplot as plt

# 加载数据
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

# 合并数据
X = np.vstack((X1, X2))
Y = np.hstack((np.zeros(len(X1)), np.ones(len(X2))))

class RBFNetwork:
    def __init__(self, n_hidden=6, sigma=1):
        self.n_hidden = n_hidden
        self.sigma = sigma
        # 随机选择RBF中心
        self.centers = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (n_hidden, X.shape[1]))
        self.weights = np.zeros((n_hidden, 1))  # 初始化输出层权重

    def gaussian(self, x, center):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * self.sigma ** 2))

    def forward(self, X):
        # 计算隐藏层输出
        self.hidden_output = np.array([[self.gaussian(x, center) for center in self.centers] for x in X])
        return self.hidden_output

    def fit(self, X, Y):
        # 计算隐藏层输出
        hidden_output = self.forward(X)
        # 使用最小二乘法计算权重
        self.weights = np.linalg.pinv(hidden_output).dot(Y.reshape(-1, 1))
        print("Calculated weights from hidden layer to output layer:")
        print(self.weights)  # 打印权重

    def predict(self, X):
        hidden_output = self.forward(X)
        output = hidden_output.dot(self.weights)
        return (output > 0.5).astype(int)  # 使用0.5作为阈值进行分类

    def accuracy(self, X, Y):
        predictions = self.predict(X)
        return np.mean(predictions.flatten() == Y)

# 创建RBF网络实例
rbf_network = RBFNetwork(n_hidden=6, sigma=1)

# 训练模型
rbf_network.fit(X, Y)

# 打印RBF中心
print("RBF Centers:")
print(rbf_network.centers)

# 计算训练后的准确率
accuracy = rbf_network.accuracy(X, Y)
print(f"Classification accuracy after training: {accuracy:.4f}")

# 绘制RBF中心
plt.scatter(rbf_network.centers[:, 0], rbf_network.centers[:, 1], c='red', marker='x', s=100, label='RBF Centers')
plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.5, cmap='bwr')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("RBF Centers and Data Points")
plt.legend()
plt.show()

# 绘制决策边界
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