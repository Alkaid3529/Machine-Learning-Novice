from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import joblib

"""
模型保存和加载:
    from sklearn.externals import joblib
    
    joblib.dump(estimator, 'test.pkl)
    
    estimator = joblib.load('test.pkl)
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

    # estimator = Ridge(alpha=0.5, max_iter=10000)
    # estimator.fit(x_train, y_train)

    estimator = joblib.load("Ridge_A.pkl")

    print('coef = ', estimator.coef_)
    print('intercept_ = ', estimator.intercept_)

    y_predict = estimator.predict(x_test)
    print('y_predict = ', y_predict)

    error = mean_squared_error(y_test, y_predict)
    print('error = ', error)

    # joblib.dump(estimator, "Ridge_A.pkl")

    return None


if __name__ == "__main__":
    print('this is Model operations')

    ridge_demo()
