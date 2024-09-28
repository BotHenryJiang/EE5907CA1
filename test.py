# Part1C.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 生成数据集
X, Y = make_moons(n_samples=100, noise=0.1, random_state=42)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 创建MLPClassifier实例
mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=10000, random_state=42)

# 训练模型
mlp.fit(X_train, Y_train)

# 打印训练后的权重和偏置
print("Trained weights and biases:")
print("Weights:", mlp.coefs_)
print("Biases:", mlp.intercepts_)

# 计算训练后的准确率
accuracy = mlp.score(X_test, Y_test)
print(f"Classification accuracy after training: {accuracy:.4f}")

# 绘制决策边界
def plot_decision_boundary(model, X, Y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', marker='o')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# 绘制决策边界
plot_decision_boundary(mlp, X_test, Y_test)