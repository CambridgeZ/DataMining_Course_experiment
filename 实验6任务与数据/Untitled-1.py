# %%
# system lib
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import model_selection

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier  #随机森林
from sklearn import tree

#用于参数搜索
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve, auc #绘制ROC曲线
import pylab as pl

from time import time
import datetime
import numpy as np

# %%
import pickle
from sklearn.model_selection import cross_validate
import pandas as pd

# %%
def load_data(filename):
    """根据数据格式，读取数据中的X和分类标签y
    """

    return x_data, ylabel

def evaluate_classifier( real_label_list,predict_label_list):
    """
       return Precision, Recall and ConfusionMatrix
       Input : predict_label_list,real_label_list
    """
    msg=''
    Confusion_matrix = confusion_matrix( real_label_list,predict_label_list)
    msg += '\n Confusion Matrix\n ' + str(Confusion_matrix)
    precision = precision_score(real_label_list,predict_label_list, average=None)
    recall = recall_score(real_label_list,predict_label_list, average=None)
    msg += '\n Precision of tag 0 and 1 =%s' %str(precision)
    msg += '\n Recall of tag 0 and 1 =%s' %str(recall)

    return msg

def test_svm(train_file, test_file):
    """用SVM分类 """
    # use SVM directly

    train_xdata, train_ylabel = load_data(train_file)

    test_xdata, test_ylabel = load_data(test_file)

    print('\nuse SVM directly')

    #classifier1 = SVC(kernel='linear')
    #classifier1 = SVC(kernel='linear',probability=True, C=200, cache_size=500)
    classifier1 = SVC(kernel='linear',probability=True,C=10, cache_size=500)

    classifier1.fit(train_xdata, train_ylabel)

    predict_labels = classifier1.predict(test_xdata)
    accuracy = accuracy_score(test_ylabel, predict_labels)
    print("\n The Classifier's Accuracy is : %f" %accuracy)
    #
    eval_msg = evaluate_classifier(test_ylabel,predict_labels)
    print(eval_msg)
    #
    #GridSearchCV搜索最优参数示例
    print("GridSearchCV搜索最优参数......")
    t0 = time()
    param_grid = {
        "C": [1e3, 5e3, 1e4, 5e4, 1e5],
        "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
    }
    classifier1 = GridSearchCV(SVC(kernel="rbf", class_weight="balanced",probability=True), param_grid)
    classifier1 = classifier1.fit(train_xdata, train_ylabel)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(classifier1.best_estimator_)


    #对于SVM来说，概率是通过交叉验证得到的，与其预测的结果未必一致，对小数据集来说，此概率没什么意义
    probas_ = classifier1.predict_proba(test_xdata)

    #对于二分类问题，可为分类器绘制ROC曲线，计算AUC
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_ylabel, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('%s SVM ROC' %train_file)
    pl.legend(loc="lower right")
    pl.show()


# %%
data = pd.read_csv('/kaggle/input/preprocess-train/preprocess_train.csv')

# %%
# 使用平均数填充缺失值
data = data.fillna(data.mean())

# %%
print(data.describe())

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
# 分割特征和标签
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 可根据需求设置测试集比例和随机种子


# %%
# 求出各个列的方差
variances = X_train.var(axis=0)
print(variances)

# %%
# 展示方差大于0.1的特征
print(variances[variances > 0.1])
#  输出个数
print(len(variances[variances > 0.1]))

# %%
# 选择方差大于0.1的特征
X_train = X_train.loc[:, variances > 0.1]

# %%
# 对于test集选择相同的特征
X_test = X_test.loc[:, variances > 0.1]

# %%
# 特征归一化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # 注意这里是fit_transform
X_test = scaler.transform(X_test) # 注意这里是transform

print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)


# %%
# 方差选择法数据预处理
from sklearn.feature_selection import VarianceThreshold
# 创建VarianceThreshold对象
selector = VarianceThreshold(threshold=0.01)

# 在训练集上拟合并应用特征选择
X_train = selector.fit_transform(X_train)

# 在测试集上应用相同的特征选择
X_test = selector.transform(X_test)

print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)

# %%
classifier1 = SVC(kernel='linear',probability=True,C=10, cache_size=10000)
classifier1.fit(X_train, y_train)

# %%
from sklearn.metrics import f1_score

predict_labels = classifier1.predict(X_test)
accuracy = accuracy_score(y_test, predict_labels)
print("\n The Classifier's Accuracy is : %f" %accuracy)
# 计算f1score
f1score = f1_score(y_test, predict_labels, average='macro')
print("\n The Classifier's f1score is : %f" %f1score)

# %%
classifier1 = SVC(kernel='linear',probability=True,C=10, cache_size=5000)
classifier1.fit(X_train, y_train)

# %%
eval_msg = evaluate_classifier(y_test,predict_labels) # 评估分类器
print(eval_msg) # 打印评估结果

# %%
print("GridSearchCV搜索最优参数......")
t0 = time()
param_grid = {
    "C": [1e3, 5e3, 1e4, 5e4, 1e5],
    "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}
classifier1 = GridSearchCV(SVC(kernel="rbf",probability=True), param_grid) #balance不需要 
classifier1 = classifier1.fit(X_train, y_train)

# %%
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:") # 打印最优参数
print(classifier1.best_estimator_) # 打印最优参数

# %%
probas_ = classifier1.predict_proba(X_test) # 对测试集进行预测
print(probas_)# 打印预测结果

# %%
# 持久化保存获得的最优svm模型。
import joblib


joblib.dump(classifier1, 'svm_model.pkl')

# %%
# 采用K-means进行分类

from sklearn.cluster import KMeans
from sklearn import metrics

# %%
# 选择最优的K值

# 评估不同K值的聚类效果
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


K = range(2, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_train)
    meandistortions.append(sum(np.min(cdist(X_train, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0])

# 绘制K值与误差平方和的关系图
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()

# %%
# 采用逻辑回归进行分类

from sklearn.linear_model import LogisticRegression

# 采用逻辑回归进行分类
classifier2 = LogisticRegression()
classifier2.fit(X_train, y_train)

# 评估分类器
from sklearn.metrics import accuracy_score
predict_labels = classifier2.predict(X_test)
accuracy = accuracy_score(y_test, predict_labels)
print("\n The Classifier's Accuracy is : %f" %accuracy)

# 计算f1score
from sklearn.metrics import f1_score
f1score = f1_score(y_test, predict_labels, average='macro')
print("\n The Classifier's f1score is : %f" %f1score)

# 计算recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, predict_labels)

# %%
# 采用决策树进行分类

from sklearn.tree import DecisionTreeClassifier


# 采用决策树进行分类
classifier3 = DecisionTreeClassifier()
classifier3.fit(X_train, y_train)

# 评估分类器
from sklearn.metrics import accuracy_score
predict_labels = classifier3.predict(X_test)
accuracy = accuracy_score(y_test, predict_labels)
print("\n The Classifier's Accuracy is : %f" %accuracy)

# 计算f1score
from sklearn.metrics import f1_score
f1score = f1_score(y_test, predict_labels, average='macro')
print("\n The Classifier's f1score is : %f" %f1score)

# 计算recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, predict_labels)

# %%
# 采用随机森林进行分类

from sklearn.ensemble import RandomForestClassifier

# 采用随机森林进行分类
classifier4 = RandomForestClassifier()
classifier4.fit(X_train, y_train)

# 评估分类器
from sklearn.metrics import accuracy_score
predict_labels = classifier4.predict(X_test)
accuracy = accuracy_score(y_test, predict_labels)
print("\n The Classifier's Accuracy is : %f" %accuracy)

# 计算f1score
from sklearn.metrics import f1_score
f1score = f1_score(y_test, predict_labels, average='macro')
print("\n The Classifier's f1score is : %f" %f1score)

# 计算recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, predict_labels)

# %%
# 多种分类器进行投票得到最终结果

from sklearn.ensemble import VotingClassifier

# 采用投票法进行分类
classifier5 = VotingClassifier(estimators=[('svm', classifier1), ('lr', classifier2), ('dt', classifier3), ('rf', classifier4)], voting='soft')
classifier5.fit(X_train, y_train)

# 评估分类器
from sklearn.metrics import accuracy_score
predict_labels = classifier5.predict(X_test)
accuracy = accuracy_score(y_test, predict_labels)
print("\n The Classifier's Accuracy is : %f" %accuracy)

# 计算f1score
from sklearn.metrics import f1_score
f1score = f1_score(y_test, predict_labels, average='macro')
print("\n The Classifier's f1score is : %f" %f1score)

# 计算recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, predict_labels)

# %%

# 绘制各模型的ROC曲线，输出AUC。建议，尝试将多个模型的ROC绘制在一幅图中。

# 绘制ROC曲线

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 绘制ROC曲线
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    plt.axis([0, 1, 0, 1]) # 范围
    plt.xlabel('False Positive Rate') # x轴标签
    plt.ylabel('True Positive Rate') # y轴标签
    plt.legend(loc="lower right") # 图例位置
# 将多个模型的ROC绘制在一幅图中
plt.figure(figsize=(8, 6)) # 设置画布大小
for clf in (classifier1, classifier2, classifier3, classifier4, classifier5):
    y_scores = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
    plot_roc_curve(fpr, tpr, clf.__class__.__name__)
plt.show()


