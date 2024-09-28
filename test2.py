import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # 使用支持向量机实现RBF

# 加载数据
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

# 合并数据
X = np.vstack((X1, X2))
Y = np.hstack((np.zeros(len(X1)), np.ones(len(X2))))

# 创建RBF SVM模型
rbf_svm = SVC(kernel='rbf', gamma='scale')  # gamma可以调整以优化模型

# 训练模型
rbf_svm.fit(X, Y)

# 计算训练后的准确率
accuracy = rbf_svm.score(X, Y)
print(f"Classification accuracy after training: {accuracy:.4f}")

# 绘制RBF中心（支持向量）
support_vectors = rbf_svm.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='red', marker='x', s=100, label='Support Vectors')
plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.5, cmap='bwr')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Support Vectors and Data Points")
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
    plt.title("Decision Boundary of RBF SVM")
    plt.show()

plot_decision_boundary(X, Y, rbf_svm)