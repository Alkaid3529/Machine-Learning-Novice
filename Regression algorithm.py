from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

"""
回归算法：
    线性回归
        欠拟合 与 过拟合
        
    岭回归
    
分类算法：
    逻辑回归
    
模型保存与加载
"""

"""
线性回归: Linear regression
    回归问题：目标值连续
        房价预测、销售额度预测
    
    原理：
        利用回归方程对一个或多个自变量（特征值）和因变量（目标值）进行建模的一种分析方式
        
        函数关系：线性模型
        
        h(w) = w1*x1 + w2*x2 + w3*x3 + ... + b = wT*x + b
        wn: 权重值
        b : 偏置值
        
        两个二维矩阵相乘
        
        广义线性模型：
            线性关系  ：空间内直线
            非线性关系：空间内曲线
            
            线性模型：
                自变量为一次幂
                参数为一次幂
    
    损失函数与优化：
        目标：求解模型参数并且使得预测结果准确
        
        真实值和预测值之间的差距，用损失函数衡量 （cost/成本函数/目标函数）
        
        损失函数定义：
            J(Θ) = ∑ (hw(xi) - yi)^2
            最小二乘法
        
        优化损失的方法：
            正规方程
                懒惰的天才：直接求解
                w = (XT*X)^-1 * XT * y
                
            梯度下降
                勤奋的普人：试错改进
                
                w1 = w1 - α * ∂ cost(w0 + w1*x1) / ∂ w1
                w0 = w0 - α * ∂ cost(w0 + w1*x1) / ∂ w0
        
    API:
        sklearn.linear_model.LinearRegression(fit_intercept=True)
        正规方程优化
        fit_intercept：计算偏置
        LinearRegression.coef_:回归系数
        LinearRegression.intercept_：偏置
        
        sklearn.linear_model.SGDRegressor(loss="squared_loss",fit_intercept=True,learning_rate='invscaling',eta0=0.01)
        梯度下降，支持不同的loss函数和正则化惩罚项拟合线性回归模型
        loss:损失类型
        learning_rate：学习率填充
        SGDRegressor.coef_:回归系数
        SGDRegressor.intercept_：偏置
    
    案例：波士顿房价预测
        获取数据
        划分数据集
        特征工程
            无量纲化 标准化
        预估器流程
            fit
            coef_   intercept_
            
    回归性能评估：
        MSE = ∑ (yi - y_)^2 / m
        均方误差
        
        API:
            sklearn.mean_squared_error
            
"""


def linear_demo():
    """
    正规方程 波士顿房价预测
    :return:
    """
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    print('coef = ', estimator.coef_)
    print('intercept_ = ', estimator.intercept_)

    y_predict = estimator.predict(x_test)
    print('y_predict = ', y_predict)

    error = mean_squared_error(y_test, y_predict)
    print('error = ', error)

    return None


def sgd_demo():
    """
    梯度下降 波士顿房价预测
    :return:
    """
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=10000,penalty="l1")
    estimator.fit(x_train, y_train)

    print('coef = ', estimator.coef_)
    print('intercept_ = ', estimator.intercept_)

    y_predict = estimator.predict(x_test)
    print('y_predict = ', y_predict)

    error = mean_squared_error(y_test, y_predict)
    print('error = ', error)

    return None


"""
正规方程：
    不需选择学习率，一次运算得出，需要计算方程，时间复杂度高
梯度下降：
    需要选择学习率，迭代求解，适合特征量较大时使用
    
    小规模数据：
        正规方程（无法拟合） num < 100K
        岭回归
    
    大规模数据：
        SGD
"""

"""
梯度下降改进：
    GD: Gradient Descent
        计算所有样本值得出梯度计算量大
        
    SGD: Stochastic Gradient Descent
        随机梯度下降：
            在一次迭代时只考虑一个训练样本
        
        高效且易实现
        需要超参数：正则参数，迭代数
        对特征标准化较敏感
        
    SAG:
        随机平均梯度法
            收敛速度慢
        最好的一种
        
"""

"""
欠拟合 与 过拟合：
    在训练集上表现很好，但是在测试集上表现不好 - 过拟合
    
    欠拟合：学习到的特征过少
        模型过于简单
        
        增加数据特征
    
    过拟合：学习到的特征过多
        模型过于复杂
        
        减少数据特征
        
    刚好拟合 just right
    
    解决方法：正则化 减少 高次幂项 影响
        L1正则化：
            损失函数 + 惩罚系数（λ）（超参数） * 惩罚项（权重的绝对值之和）
            直接置零，较为激烈
            LASSO
        
        L2正则化，更常用：
            损失函数 + 惩罚系数（λ）（超参数） * 惩罚项（权重的平方和）
            削弱权重，较为缓和
            Ridge - 岭回归
    
"""

"""
岭回归：带L2正则化的线性回归
    在算法建立回归方程时加上正则化限制
    
    API：
        klearn.linear_model.Ridge(alpha=1.0,fit_intercept=True,solver="auto",normalize=False)
        alpha: 正则化力度，惩罚项系数  可调超参数
            0~1 1~10
        solver：自动选择优化方法
            sag: 若数据量较大，自动选择随机梯度下降
        normalize: 标准化数据
        
        Ridge.coef_:
        Ridge.intercept_
        
"""


def ridge_demo():
    """
    岭回归 波士顿房价预测
    :return:
    """
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = Ridge(alpha=0.5, max_iter=10000)
    estimator.fit(x_train, y_train)

    print('coef = ', estimator.coef_)
    print('intercept_ = ', estimator.intercept_)

    y_predict = estimator.predict(x_test)
    print('y_predict = ', y_predict)

    error = mean_squared_error(y_test, y_predict)
    print('error = ', error)

    return None


if __name__ == '__main__':
    print('this is Regression algorithm')

    # 线性模型
    linear_demo()
    print('')
    sgd_demo()

    # 岭回归
    ridge_demo()
