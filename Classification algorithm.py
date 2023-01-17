import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

print('Classification algorithms')

"""
分类算法： 目标值离散，类别
    
    sklearn 转换器和预估器
    KNN 算法
    模型选择与调优
    
    朴素贝叶斯算法
    
    决策树
    随机森林
    
"""

"""
sklearn 转换器和预估器
    
    sklearn 转换器：
        实例化
        调用 fit_transform:
            
            例：标准化
                (x - mean) / std
                fit(): 计算每一列的数据的平均值、标准差
                transform(): 转换为最终形式
                
    sklearn 预估器： estimator
        所有算法都被封装在预估器当中
        
        实例化一个 estimator类
        调用 fit 方法： estimator.fit(x_train, y_train) 计算
                       计算完毕，生成模型
        模型评估：
            直接比较预测值与真实值：
                y_predict = estimator.predict(x_test)
                y_test == y_predict
            
            计算准确率：
                accuracy =  estimator.score(x_test, y_test)
        
"""

"""
KNN 算法：K 近邻算法
    k nearest neighbors
    根据邻居推断类别
    
    K - 近邻算法原理
        如果一个样本在特征空间中的 K 个最相似的样本中的大多数属于某个类别，则该样本也属于这个类别
        
        k = 1 : 容易受到异常值干扰
        
        距离公式：A (a1, a2, a3)  B(b1, b2, b3)
            欧氏距离: sqrt((a1 - b1)^2 + )
            曼哈顿距离: |a1 - b1| + 
            闵可夫斯基距离: 
            
    实例：电影类型分析
        根据各种镜头数量抽取特征
        
    K 取值
        k 取值过小：容易受异常数据影响
        k 取值过大：容易受样本不均衡影响
        
    数据处理：无量纲化 标准化
    
    API:
        sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto)
            n_neighbors: 邻居数量
            algorithm: 查找算法，默认即可
        
        实例化，调用，
        
    算法简单，易于理解
    K 取值不易确定，选择不当导致误差；懒惰算法，计算量大，内存开销大，适用于小数据量
        
"""


def knn_iris_demo():
    """
    利用KNN算法对鸢尾花进行分类
    :return:
    """
    # 获取数据
    iris = load_iris()

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 特征工程 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 模型评估

    # 比对
    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('compare;\n', y_test == y_predict)

    # 计算
    score = estimator.score(x_test, y_test)
    print('accuracy = \n', score)
    return None


"""
模型选择与调优：
    在 KNN 中，如何选择最合适的 K 值？
    
    交叉验证： cross validation
        将训练集的数据分为训练和验证集，让模型更加准确
        
        训练集：训练 验证
        测试集：测试
        
        四折交叉验证
        
    超参数搜索 - 网格搜索： Grid search
        暴力搜索： for k in range()
        
    API:
        sklearn.model_selection.GridSearchCV(estimator,param_grid=None,cv=None)
            对预估器的指定参数进行详尽搜索
            estimator: 预估器对象
            param_grid: 估计器参数(dict)
            cv: 指定几折交叉验证
            fit(): 输入训练数据
            score(): 准确率
            结果分析：
                最佳参数：
                最佳结果：
                最佳预估器：
                交叉验证结果：
"""


def knn_iris_gscv_demo():
    """
    超参数搜索、交叉验证优化
    :return:
    """

    iris = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

    transfer = StandardScaler()

    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = KNeighborsClassifier(n_neighbors=3)

    # 模型选择与调优
    param_dict = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)

    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('compare;\n', y_test == y_predict)

    score = estimator.score(x_test, y_test)
    print('accuracy = \n', score)

    # 最佳参数：
    print('best_params_:\n', estimator.best_params_)
    # 最佳结果：
    print('best_score_:\n', estimator.best_score_)
    # 最佳预估器：
    print('best_estimator_:\n', estimator.best_estimator_)
    # 交叉验证结果：
    print('cv_results_:\n', estimator.cv_results_)

    return None


def facebook_demo():
    """
    案例
        流程分析：
            获取数据
            数据处理：
                缩小数据范围
                    x y 一定范围
                    time 具体时间
                    place_id 剔除低频率地址
            特征工程
            KNN算法预估流程
            模型选择与调优
            模型评估

        未完成
    :return:
    """

    data = pd.read_csv("./FBlocation/train.csv")

    # 缩小数据规模
    data = data.query("x<2.5 & x>2.0 &y<1.5&y>1.0")

    # 处理时间戳
    time_value = pd.to_datetime(data["time"], unit="s")

    date = pd.DatetimeIndex(time_value)

    data["day"] = date.day
    data["weekday"] = date.weekday
    data["hour"] = date.hour

    # 过滤频率过低的地点
    place_count = data.groupby("place_id").count()["row_id"]

    data_final = data[data["place_id"].isin(place_count[place_count > 3]).index.values]

    return None


"""
朴素贝叶斯算法：得到不同的概率分布

    概率基础：
        样本量不足时，结果并不准确
        
        联合概率：同时满足多个条件
            P(A,B)
            
        条件概率：在一个 B 条件下另一个 A 事件发生的概率
            P(A|B)
            
        相互独立：P(A,B) = P(A) * P(B)
        
    贝叶斯公式：
        P(C|W) = P(W|C) * p(C) / P(W)
        
    朴素：假设特征与特征之间相互独立
    
    应用场景：
        文本分类
            以单词作为特征，单词之间独立性更高
            
            拉普拉斯平滑系数
            
    API:
        sklearn.naive_bayes.MultinomialNB(alpha = 1.0)
        朴素贝叶斯分类
        alpha: 拉普拉斯平滑系数
        
    案例：新闻分类
        获取数据
        划分数据集
        特征工程
            文本特征抽取
        朴素贝叶斯算法预估器
        
    对缺失数据不敏感
    分类稳定
    准确度高，速度快
    
    处理特征关联性较强的数据时效果不好
"""


def nb_news_demo():
    """
    朴素贝叶斯算法
    :return:
    """
    news = fetch_20newsgroups(subset='all')

    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    transfer = TfidfVectorizer()

    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = MultinomialNB()

    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('compare;\n', y_test == y_predict)

    score = estimator.score(x_test, y_test)
    print('accuracy = \n', score)

    return None


"""
决策树：
    认识决策树：
        利用 if-else 结构分类
        
        高效进行决策，特征先后顺序
        
    决策树分类原理：
        如何找到最高效的决策顺序？
        信息熵，信息增益
        
        信息论基础：
            信息
                香农：消除随机不定性的东西
                
            信息量：
                信息熵: H(X) = -∑ P(xi) * ( log b P(xi) )
            
            信息增益：
                条件熵：H(D|A)
                
                g(D,A) = H(D) - H(D|A)
    
    API:
        sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=None)
        criterion: 默认基尼系数
        max_depth：树的深度大小
        random_state：随机数种子
        
    可视化：
        sklearn.tree.export_graphviz(estimator,out_file='tree.dot,feature_names=['','']
        
    可解释能力强，树木可视化
    
    无法推广至过于复杂的树，容易产生过拟合
        剪枝cart算法
        随机森林
"""


def tree_iris_demo():
    """
    利用决策树对鸢尾花分类
    :return:
    """
    iris = load_iris()

    mineral_train = 0

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    print(x_train)

    estimator = DecisionTreeClassifier(criterion="entropy")

    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('compare;\n', y_test == y_predict)

    score = estimator.score(x_test, y_test)
    print('accuracy = \n', score)

    export_graphviz(estimator, out_file='tree_iris_demo tree.dot', feature_names=iris.feature_names)

    return None


"""
没有免费的午餐
勇于尝试
"""

"""
集成学习方法之随机森林：
    集成学习方法：
        生成多个分类器，分别单独预测，组成最终预测
        优于任何一种单分类做出的预测
        
    随机森林：
        森林：包含多个决策树的分类器，最终结果由多棵树的众数决定
        
        随机：
            训练集随机：
                bootstrap 随机有放回抽样：从原有的 N 个样本中随机有放回抽样抽取 N 个
                    [1, 2, 3, 4, 5]
                    新的树的训练集
                    [2, 4, 5, 1, 2]
                    [3, 2, 3, 5, 1]
                    ......
            特征值随机:
                从 M 个特征中随机抽取 m 个特征
                M >> m
                降维
                
    API: sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, bootstrap=True, 
    random_state=None, mini_sample_split=2)
    max_features:
        auto: m = sqrt(M)
        sqrt: m = sqrt(M)
        log2: m = log2(M)
        None: m = M 
    
"""


def random_forest_demo():
    """
    随机森林预测鸢尾花树
    :return:
    """
    iris = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

    estimator = RandomForestClassifier()
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('compare;\n', y_test == y_predict)

    score = estimator.score(x_test, y_test)
    print('accuracy = \n', score)

    return None


"""
总结：
    以上算法可以有效运行在大数据集，处理高维数据，无需降维
    
    转换器和预估器
        转换器：特征工程
        预估器：机器学习
    
    KNN算法
        找到距离最近的几个邻居
        适合数据量较少的情况
    
    朴素贝叶斯
        特征之间相互独立
        拉普拉斯平滑系数
        不适用于特征间关系较强的情况
        
    决策树
        信息增益 log P(Xi) 的期望
        可视化，可解释能力强
    
    随机森林
        高维度，大数据集
"""

if __name__ == "__main__":
    # 利用KNN算法对鸢尾花进行分类
    # knn_iris_demo()

    # 超参数搜索、交叉验证优化
    # knn_iris_gscv_demo()

    # 案例 未完成
    # facebook_demo()

    # 朴素贝叶斯算法
    # nb_news_demo()

    # 决策树
    tree_iris_demo()

    # 随机森林
    # random_forest_demo()
