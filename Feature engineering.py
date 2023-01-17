print('\nThis is sklearn datasets')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from matplotlib import pyplot
from sklearn.decomposition import PCA

import jieba
import pandas as pd

'''
load_*() 小规模数据集
fetch_*() 大规模数据集
'''

# sklearn.datasets.load_iris()
# 小规模数据集

# sklearn.datasets.fetch_20newsgroups(data_home=None, subset='all')
# subset: train test all
# 大规模数据集

# 返回值数据类型为 Bunch
# Bunch 继承自字典，存在键值对
# 利用字典属性访问
# Bunch.key = values


'''
Bunch
data: 特征数据数组，二维numpy.ndarray数组
target: 标签数组，一维numpy.ndarray数组
DESCR: 数据描述
feature_names: 特征名字
target_names: 标签名字
'''

'''
不可以将全部数据均用于训练，否则模型评估就没有数据可以使用了
训练集：0.7 0.8 0.75
测试集：0.3 0.2 0.3
'''

'''
数据集划分：
sklearn.model_selection.train_test_split(arrays,option)
x: 数据集特征值
y: 数据集标签值
test_size: 测试集大小
random_state: 随机数种子
return: 训练集特征值 x_train ，测试集特征值 x_test ，训练集目标值 y_train ，测试集目标值 y_test
'''


def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """

    # 获取数据集
    iris = load_iris()

    # 打印整个数据集
    print("\n鸢尾花数据集：")
    print(iris)

    # 数据集描述
    print("\n数据集描述：")
    print(iris["DESCR"])

    # 特征值名字
    print("\n特征值名字：")
    print(iris.feature_names)

    # 特征值
    print("\n特征值：")
    print(iris.data, iris.data.shape)

    # 标签名
    print("\n标签名：")
    print(iris.target_names)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    print(x_train, x_train.shape)
    print(x_test, x_test.shape)
    print(y_train, y_train.shape)
    print(y_test, y_test.shape)

    return None


'''
Feature Engineering: data process

处理数据的技巧，使算法的性能更好地发挥
pandas: 数据清洗和处理
sklearn: feature learning

1. 特征抽取
    机器学习算法：统计方法，数学公式
    文本类型 -> 数值
    类型 -> 数值

    将任意数据转换为可用于机器学习的数字特征
        字典特征提取
        文本特征提取
        图像特征提取

    sklearn.feature_extraction()

    字典特征提取

    文本特征提取
        

2. 特征预处理
    无量纲化：
        归一化
        
        标准化

3. 特征降维
    降维：降低维度
    
    ndarray
    
    零维：标量
    一维：向量
    二维：矩阵
    三维
    N 维
    
    降低维度，不是嵌套层数
    
    二维数组：
        降维：降低特征个数，减少列数，得到一组不相关的主变量的过程
            不相关主变量：特征与特征之间不相关
            
    降维方式：
        特征选择
        主成分分析

数据和特征决定机器学习的上限，模型和算法只是逼近这个上限而已
'''


def dict_demo():
    """
    字典特征提取
    类别 -> one-hot编码
    sklearn.feature_extraction.DictVectorizer(sparse=True)
    Vector: 矢量 一维数组存储
    矩阵：matrix 二维数组

    实例化了一个转换器类
    DictVectorizer.fit_transform(X) X: 字典或者包含字典的迭代器  返回值：sparse矩阵（稀疏矩阵）
    稀疏矩阵：记录非零值的位置与值

    应用场景: 数据集中类别特征较多
             将数据集特征转化为数据类型
    :return:
    """
    data = [{'city': 'Beijing', 'tem': 30}, {'city': 'Shanghai', 'tem': 35}, {'city': 'Shenzhen', 'tem': 32}]
    # 1. 实例化一个转换器类
    transfer = DictVectorizer(sparse=True)

    # 2. 调用 fit_transform()
    data_new = transfer.fit_transform(data)

    # data_new本质为特殊的数据形式，转换为二维数组有专门的方法

    print('data_new : \n', data_new)
    print('data_new : \n', data_new.toarray())
    print('')
    print('feature_names : \n', transfer.get_feature_names_out())

    return None


def count_demo():
    """
    文本特征提取
        单词作为特征
        句子、短句、字母  单词更合适，句子太复杂

        特征：特征词（单词）

        方法：
            sklearn.feature_extraction.text.CountVectorizer(stop_words=[])
            通计样本特征值出现次数，默认空格为分词界
    :return:
    """

    data = ["life is short,i like like python", "life is too long,i dislike python"]

    transfer = CountVectorizer(stop_words=['is', 'too'])

    # stop_words 停用词 停用词表

    data_new = transfer.fit_transform(data)

    print('data_new : \n', data_new)

    # 利用专门方法转换为二维数组
    print('data_new : \n', data_new.toarray())

    print('')
    print('feature_names : \n', transfer.get_feature_names_out())

    return None


def count_chinese_demo1():
    """
    中文文本特征提取
    :return:
    """
    data = ["我 爱 北京 天安门", "天安门 上 太阳 升"]

    transfer = CountVectorizer()

    data_new = transfer.fit_transform(data)

    print('data_new : \n', data_new)

    # 利用专门方法转换为二维数组
    print('data_new : \n', data_new.toarray())

    print('')
    print('feature_names : \n', transfer.get_feature_names_out())

    return None


def cut_word(text):
    """
    中文分词
    :param text: 传入文本
    :return: 返回处理好的文本
    """
    text = " ".join(list(jieba.cut(text)))

    return text


def count_chinese_demo2():
    """
    中文文本特征提取 自动分词
    :return:
    """

    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。", "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    # 仅根据词语出现的次数进行分类，并不完全合理，部分词语经常出现在各种文章中
    # 需要找到在某类别文章中出现频率高，而在其他类别文章中很少出现的词语，称为关键词

    data_new = []

    # 将中文文本分词
    for sent in data:
        data_new.append(cut_word(sent))

    transfer = CountVectorizer(stop_words=['一种', '不要', '只用'])

    data_final = transfer.fit_transform(data_new)

    print("data_final:\n", data_final.toarray())
    print("feature_names:\n", transfer.get_feature_names_out())

    return None


"""
Tf-idf 文本特征提取
    Tf-idfVectorizer()
    
    TF-IDF - 重要程度 = TF * IDF
        Tf: term frequency词频
        IDF: inverse document frequency 逆文档频率 总文件数除以包含该词语的文件数目取十的对数
        
"""


def tfidf_demo():
    """
    Tf - Idf 文本特征抽取
    :return:
    """

    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。", "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    transfer = TfidfVectorizer()

    data_new = []

    for sent in data:
        data_new.append(cut_word(sent))

    data_final = transfer.fit_transform(data_new)

    print("data_final:\n", data_final.toarray())
    print("feature_names:\n", transfer.get_feature_names_out())

    return None


"""
特征预处理：
通过一些转换函数，将特征数据更加适合算法模型的特征数据的过程

数值型处理无量纲化：避免因量纲不统一导致的误差，提高预测准确度

    归一化：将原始数据进行变换，限制在[0 - 1]范围内
        X` = (X - min) / (max - min);
        X`` = X` * (mx - mi) + min;
        
        sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)...)
        MinMaxScaler.fit_transform(X) X: nd.array类型
        
        如果存在异常值：最大值、最小值，会导致归一化结果偏离过大,鲁棒性较差
        
    标准化：将原始数据变换到均值为0，标准差为1的范围内
        X` = (X - mean) / σ 
        mean: 均值变化不会太大
        σ: 标准差 集中程度 变化不会很大
        
        因此利用标准化处理数据，可以避免异常值影响，更加稳定，更加适合大多数场景
        
特征预处理：
    sklearn.preprocessing
"""


def minmax_demo():
    """
    归一化
    :return:
    """
    data = [[10, 30, 50],
            [50, 20, 40],
            [30, 25, 50]]

    transfer = MinMaxScaler()

    data_new = transfer.fit_transform(data)

    print("data:\n", data)
    print("data_new:\n", data_new)
    print("feature_names:\n", transfer.get_feature_names_out())

    return None


def standard_demo():
    """
    标准化处理数据
    :return:
    """

    data = [[10, 30, 50],
            [50, 20, 40],
            [30, 25, 50]]

    print('data:\n', data)

    transfer = StandardScaler()

    data_new = transfer.fit_transform(data)

    print('data_new:\n', data_new)

    print('feature_names:\n', transfer.get_feature_names_out())

    return None


"""
特征选择：剔除冗余特征，从原有特征中找到主要特征
    sklearn.feature_selection

    过滤式：Filter
        方差选择法：低方差特征过滤
            某特征，大多样本值较接近
            某特征，很多样本值有区别
            
            sklearn.feature_selection.VarianceThreshold()
            删除方差不符合的特征，过滤不太重要的特征
        
        相关系数：特征之间的相关程度，当某几个特征相关性过强的话，表明有冗余特征
            皮尔逊相关系数：[-1, 1]
                r > 0 : 正相关
                r < 0 : 负相关
                abs(r) < 0.4          低相关
                0.4 < abs(r) < 0.7    显著相关
                0.7 < abs(r) < 1      高度相关
                
            相关性很高：
                选取其中一个
                加权求和
                主成分分析
                
    
    嵌入式：Embedded 算法自动选择特征
        决策树：信息熵，信息增益
        
        正则化：L1,L2
        
        深度学习：卷积
        
"""


def Variance_demo():
    """
    低方差特征过滤
    :return:
    """
    data = [[10, 30, 50, 5],
            [50, 20, 40, 5],
            [30, 25, 50, 5],
            [20, 80, 40, 5]]

    print('data:\n', data)

    transfer = VarianceThreshold(threshold=0)

    data_new = transfer.fit_transform(data)

    print('data_new:\n', data_new)

    print('feature_names:\n', transfer.get_feature_names_out())

    # 计算某两特征之间的相关系数
    print(pearsonr(data[0], data[2]))

    # 绘图
    pyplot.figure()
    pyplot.scatter(data[2], data[1])
    pyplot.show()

    return None


"""
主成分分析：PCA降维
    将高维数据转化为低维数据，压缩数据维数，尽可能损失少量信息，应用与回归和聚类
    
    例：拍摄水壶
        将三维水壶拍摄为二维照片
        
        衡量标准：能否通过二维数据还原为三维
        
    例：用一条直线拟合五个点，同时最大化减少误差
    
        计算方式：找到一条合适直线，通过一个矩阵运算，得到主成分分析结果
        
    sklearn.decomposition.PCA(n_components=None)
        n_components: float 降维后保留多少比例的信息  0.6 -> 60%
        n_components: int   降维后保留多少特征  6 -> 6 个
    
"""


def pca_demo():
    """
    PCA降维
    :return:
    """

    data = [[10, 30, 50, 5],
            [50, 20, 40, 5],
            [30, 25, 50, 5],
            [20, 80, 40, 5]]

    print('data:\n', data)

    transfer = PCA(n_components=3)

    data_new = transfer.fit_transform(data)

    print('data_new:\n', data_new)

    return None


"""
案例：
    推测用户喜好的产品：
        user_id   aisle_id
        将用户和商品类别放在同一张表中 - 合并
        找到两者之间的关系 - 交叉表、透视表
        减少冗余特征
"""


def instance_demo():
    """
    实践
    :return:
    """
    # 读取数据
    order_products = pd.read_csv("./instacart/order_products__prior.csv")
    products = pd.read_csv("./instacart/products.csv")
    orders = pd.read_csv("./instacart/orders.csv")
    aisles = pd.read_csv("./instacart/aisles.csv")

    # 合并表

    tab1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])

    tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])

    tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])

    table = pd.crosstab(tab3["user_id"], tab3["aisle"])

    # 找到数据关系

    # PCA降维

    data = table[:100]

    transfer = PCA(n_components=0.95)

    data_new = transfer.fit_transform(data)

    print(data_new)

    return None


""""
机器学习 phase First
    机器学习概述：
        人工智能
        
        什么是机器学习
        
        机器学习算法分类
        
        机器学习开发流程
    
    特征工程
"""

if __name__ == "__main__":
    print('')
    # sklearn数据集使用
    # datasets_demo()

    # 字典特征提取
    # dict_demo()

    # 文本特征提取
    # count_demo()
    # print('')

    # 中文文本特征提取
    # count_chinese_demo1()
    # print('')

    # 中文文本特征处理 自动分词
    # count_chinese_demo2()

    # Tf - Idf 文本特征抽取
    # tfidf_demo()

    # 归一化
    # minmax_demo()

    # 标准化
    # standard_demo()

    # 低方差特征过滤
    # Variance_demo()

    # 主成分分析
    # pca_demo()

    # 案例
    # instance_demo()
