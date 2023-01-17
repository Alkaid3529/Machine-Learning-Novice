"""
无监督学习：
    没有目标值的分类问题 —— 无监督学习

    聚类：
        k-means:
            1> 随机找 k 个点作为初始的聚类中心，作为超参数
            2> 对于其他点，找到距离最近的聚类中心点，标记为一类
            3> 对每一类求一个中心点，作为新的聚类中心
            4> 若新旧聚类中心重合，则结束，否则，重新计算

            中心点求解：各个坐标求平均值

        API：
            sklearn.cluster_KMeans(n_clusters=8,init='k-means++)
            n_clusters: 聚类中心数量
            init: 初始化方法
            labels: 默认标记类型，可以和真实值比较

        评估：轮廓系数
            sci = (bi - ai) / max(bi, ai)
                bi: 到类外点的最短距离
                ai: 到类内点的平均距离

            sci ∈ [-1， 1]:
                趋于 1 性能越好

            高内聚，低耦合
            类外稀疏，类内密集

            API：
                sklearn.metrics.silhouette_score(X,labels)

            迭代式算法，易于理解
            易陷入局部最优

            分类

    降维：
        PCA

"""


def kmeans_demo():
    """
    k-means
    :return:
    """

    return None


if __name__ == "__main__":
    print('this is Clustering algorithm')

    kmeans_demo()
