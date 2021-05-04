import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset
# 超参数
learning_rate = 0.005
iterations = 1500

class lr():
    def __init__(self, dim):
        self.w = np.zeros(shape=(dim, 1))
        self.b = 0

    def sigmoid(self, z):
        return .5 * (1 + np.tanh(.5 * z))#使用tanh的形式防止溢出

    def forward(self, X, Y):
        m = X.shape[1]  # 样本数
        A = self.sigmoid(np.dot(self.w.T, X)+self.b)  # 激活向量
        cost = (-1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
        return A, cost  # 缓存中间值A以反向传播

    def backward(self, X, Y, A):
        m = X.shape[1]  # 样本数
        db = (1/m)*np.sum(A-Y)
        dw = (1/m)*np.dot(X, (A-Y).T)
        assert(dw.shape == self.w.shape)
        grad = {
            "dw": dw,
            "db": db
        }
        return grad

    def GD(self, grad):
        self.w = self.w-learning_rate*grad['dw']
        self.b = self.b-learning_rate*grad['db']

    def predict(self,X):
        m = X.shape[1]
        Y = np.zeros(shape=(1,m))
        A = self.sigmoid(np.dot(self.w.T, X)+self.b)
        for i in range(A.shape[1]):
            if A[0][i]>=0.5:
                Y[0][i]=1
            else:
                Y[0][i]=0
        assert(Y.shape==(1,m))
        return Y

def main():
    # 读取数据
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
    # 将图片展平后按列组合
    #train_set_x_flatten = train_set_x_orig.reshape(-1, train_set_x_orig.shape[0])这行代码是错的，样本并不是按列分的
    train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T#先确定行数再转置
    # 将图片展平后按列组合
    #test_set_x_flatten = test_set_x_orig.reshape(-1, test_set_x_orig.shape[0])这行代码是错的，样本并不是按列分的
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T#先确定行数再转置
    # 图像预处理、归一化
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255
    #开始训练
    LR=lr(train_set_x.shape[0])
    costs=[]
    for i in range(iterations):
        A,cost=LR.forward(train_set_x,train_set_y_orig)#前向传播计算损失，batchsize取整个训练集
        if i%100==0:
            costs.append(cost)
        grad=LR.backward(train_set_x,train_set_y_orig,A)#反向传播计算梯度
        LR.GD(grad)#梯度下降
    #在训练集、测试集上测试
    Y_train=LR.predict(train_set_x)
    Y_test=LR.predict(test_set_x)
    print("train set accuracy："  , format(100 - np.mean(np.abs(Y_train - train_set_y_orig)) * 100) ,"%")
    print("test set accuracy："  , format(100 - np.mean(np.abs(Y_test - test_set_y_orig)) * 100) ,"%")
    #绘制cost下降图
    x=list(range(len(costs)))
    plt.plot(x,costs)
    plt.show()



if __name__ == "__main__":
    main()

