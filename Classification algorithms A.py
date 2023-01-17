from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

"""
逻辑回归：
    应用场景：二分类（属于或者不属于）
        广告点击率：是否会被点击
        是否为垃圾邮件
        是否患病
        金融诈骗
        虚假账号

    原理：
        输入：
            h(w) = w1*x1 + w2*x2 + w3*x3 + ... + b = wT*x + b
            （线性模型的输出）
            映射到激活函数上，得到一个数（视作概率值）

        激活函数：
            sigmoid函数：1 / (1 + e^(-x))
            回归的结果输入到sigmoid函数中
            输出结果为 0~1 的一个概率值，多以 0.5 为阈值
            将回归问题转换为了分类问题

        假设函数：
            1 / (1 + e^( w1*x1 + w2*x2 + w3*x3 + ... + b = wT*x + b ))

        损失函数：最小二乘法 / 均方误差法
            ∑ ( y_predict - y_true )^2 / M
            逻辑回归真实值：类别

        损失与优化：
            损失：对数似然损失
                cost(h(x), y) = { -log (h(x))    y == 1
                                { -log (1-h(x))  y == 0

                cost(h(x), y) = ∑ -yi*log(h(x)) - (1 - yi)*log(1 - h(x))

                样本特征输入 -> 回归计算 -> 回归 -> 逻辑回归结果 -> 真实结果

            优化：梯度下降

    API:
        sklearn.linear_model.LogisticRegression(solver='liblinear',penalty='l2',C=1.0)
        solver: 优化求解方式
        penalty: 正则化方式
        C: 正则化力度

    案例：癌症分类
        流程：
            获取数据：
                添加名称
            数据处理：
                处理缺失值
            数据集划分
            特征工程
                无量纲化 标准化
            逻辑回归预估器
            模型评估
            
        未实现
"""

"""
分类评估: 确实患癌症被检查出的概率
    精确率和召回率：
        混淆矩阵：
            预测结果和正确标记的组合构成混淆矩阵，适用于多分类
            
            TP = True Positive
            FN = False Negative
            
            精确率：
                预测为正例样本中真实正例的比例
                TP / (TP + FP)
            召回率：检查是否全面
                真实正例中预测为正例样本的比例
                TP / (TP + FN)
            F1-score：
                反应模型稳健性
                F1 = 2 * TP / (2 * TP + FN + FP)
                
    API:
        sklearn.metrics.classification_report(y_true,y_predict,labels=[],target_names=None)
    
"""

"""
ROC曲线和AUC指标：衡量样本不均衡下模型的准确率
    AUC = 1
        性能好
    
    AUC = 0.5
        性能差
        
    ROC曲线：
        TPR：
            TPR = TP / (TP + FN)
            所有真实类别为1的样本中，预测类别为1的比例
        FPR:
            FPR = FP / (TN + FP)
            所有真实类别为0的样本中，预测类别为1的比例
            
    API:
        sklearn.metrics.roc_auc_score()
        计算 ROC 曲线面积，即 AUC 值
"""

if __name__ == "__main__":
    print('this is Classification algorithms')
