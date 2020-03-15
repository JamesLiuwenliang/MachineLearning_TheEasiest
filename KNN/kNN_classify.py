import numpy as np
from math import sqrt
from collections import Counter
# k       : 取k个最邻近的值的
# X_train : 样品的参数(比如在n*2矩阵里 ,有n个样品量 , 2是指样品的特征量,比如鸢尾花的花瓣大小和茎的长短
# y_train : 二分法下,只有0和1,0表示是哪一类鸢尾花,1表示另一类鸢尾花
# x       :
def kNN_classify(k,X_train ,y_train,x):
    assert 1<= k <= X_train.shape[0], "k must be valid."
    assert X_train.shape[0] == y_train.shape[0] , \
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0] , \ 
        "the feature number of x must be equal to X_train"
    
    distances = [sqrt(np.sum((x_train -x )** 2)) for x_train in X_train]

    nearest = np.argsort(distances)

    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)
    
    return votes.most_common(1)[0][0]