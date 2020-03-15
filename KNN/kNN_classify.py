import numpy as np
from math import sqrt
from collections import Counter
# k       : 取k个最邻近的值的
# X_train : 样品的参数(比如在n*2矩阵里 ,有n个样品量 , 2是指样品的特征量,比如鸢尾花的花瓣大小和茎的长短
# y_train : 二分法下,只有0和1,0表示是哪一类鸢尾花,1表示另一类鸢尾花   (就是标记)
# x       : 要判断的样品(可以是个1*2的矩阵)
def kNN_classify(k,X_train ,y_train,x):
    assert 1<= k <= X_train.shape[0], "k must be valid."
    assert X_train.shape[0] == y_train.shape[0] , \
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0] , \ 
        "the feature number of x must be equal to X_train"

    # 算出样本点(已知其鸢尾花种类)相对于x的距离
    distances = [sqrt(np.sum((x_train -x )** 2)) for x_train in X_train]
    # distances = []
    # for x_train in X_train:
    #     d = sqrt(np.sum((x_train -x )** 2)) # 求差值平方的和,再开平方
    #     distances.append(d)

    # 对距离排序
    nearest = np.argsort(distances)

    # 得到最近的k个点的标记(0 OR 1)
    topK_y = [y_train[i] for i in nearest[:k]]

    # 统计 0 和 1 的个数
    votes = Counter(topK_y)

    # 返回最多的元素是哪个
    return votes.most_common(1)[0][0]