# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %%
# Apriori 算法
# 1. 支持度计算
def loadDataSet():
        # 读取数据txt中的数据
        data = pd.read_csv('Transactions.csv')
        return data

# %%
def createC1(dataSet):# 创建候选项集
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))# 使用frozenset是为了后面可以将这些值作为字典的键

def scanD(D, Ck, minSupport):# 从候选项集中筛选出满足最小支持度的项集
    # D: 数据集 Ck: 候选项集  minSupport: 最小支持度
    ssCnt = {}

    for tid in D: # 遍历数据集中的每一条交易记录
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: # 如果不在字典中，就添加进去
                    ssCnt[can] = 1
                else: # 如果在字典中，就加1
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData # 返回满足最小支持度的项集和对应的支持度



# %%
# 2. 生成候选项集
def aprioriGen(Lk, k):# 生成候选项集
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport = 0.0): # 生成所有满足最小支持度的项集
    # dataSet: 数据集  minSupport: 最小支持度
    C1 = createC1(dataSet)
    # D = list(map(set, dataSet))
    D = []
    # 遍历每一行
    for index, row in dataSet.iterrows():
        # 创建一个空集合来存储这一行的列名
        s = set()
        # 遍历除了views之外的所有列
        for col in dataSet.columns[:-1]:
        # 如果这个数据是1，就把这列的名称加入到集合中
            if row[col] == 1:
                s.add(col)
        # 将s转为list
        s = list(s)
        # 把这一行对应的集合加入到列表中
        D.append(s)

    # print (D)
    L1, supportData = scanD(D, C1, minSupport)
    print (L1)
    L = [L1]
    k = 2 # k表示项集中元素的个数
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData # 返回所有满足最小支持度的项集和对应的支持度


# %%
# 2. 生成候选项集
def aprioriGen(Lk, k):# 生成候选项集
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport = 0.05): # 生成所有满足最小支持度的项集
    # dataSet: 数据集  minSupport: 最小支持度
    C1 = createC1(dataSet)
    # D = list(map(set, dataSet))
    D = []
    # 遍历每一行
    for index, row in dataSet.iterrows():
        # 创建一个空集合来存储这一行的列名
        s = set()
        # 遍历除了views之外的所有列
        for col in dataSet.columns[:-1]:
        # 如果这个数据是1，就把这列的名称加入到集合中
            if row[col] == 1:
                s.add(col)
        # 将s转为list
        s = list(s)
        # 把这一行对应的集合加入到列表中
        D.append(s)

    print (D)
    L1, supportData = scanD(D, C1, minSupport)
    print (L1)
    L = [L1]
    k = 2 # k表示项集中元素的个数
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData # 返回所有满足最小支持度的项集和对应的支持度


# %%
# 3. 生成关联规则
def generateRules(L, supportData, minConf=0.1):# 生成关联规则
    bigRuleList = []
    
    for i in range(1, len(L)):
        for freqSet in L[i]: # 遍历频繁项集中的每一个项集

            print(freqSet)

            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1): # 如果频繁项集中的元素超过2个，就需要进一步合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):# 计算置信度
    # freqSet:频繁项集 H:频繁项集中所有的元素的集合  supportData:支持度数据  brl:关联规则列表  minConf:最小置信度
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq] # 计算置信度
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):# 生成候选规则集:
    # freqSet:频繁项集 H:频繁项集中所有的元素的集合  supportData:支持度数据  brl:关联规则列表  minConf:最小置信度
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

    

# %%
# test
dataSet = loadDataSet()
dataSet.head()

# %%
dataSet.drop(['COD'], axis=1, inplace=True)
dataSet.head()

# %%
# 将数据集转换为集合,如果对应项为0，则将这一列的名称不添加到集合中，如果为不为0，则添加到集合中
# 假设df是Pandas读入的数据
# 创建一个空列表来存储每一行对应的集合
D = []
# 遍历每一行
for index, row in dataSet.iterrows():
    # 创建一个空集合来存储这一行的列名
    s = set()
    # 遍历除了views之外的所有列
    for col in dataSet.columns[:-1]:
    # 如果这个数据是1，就把这列的名称加入到集合中
        if row[col] == 1:
            s.add(col)
    # 将s转为list
    s = list(s)
    # 把这一行对应的集合加入到列表中
    D.append(s)

# %%
D

# %%
C1 = createC1(D) # 创建候选项集
C1

# %%
L1, supportData0 = scanD(D, C1, 0.0) # 从候选项集中筛选出满足最小支持度的项集
print(L1) # 打印出满足最小支持度的项集
print(supportData0) # 打印出每个项集的支持度

# %%
L, supportData = apriori(D) # 生成所有满足最小支持度的项集
print(L) # 打印出所有满足最小支持度的项集
print(supportData) # 打印出每个项集的支持度

# %%
rules = generateRules(L, supportData, minConf=0.7) # 生成关联规则
print(rules) # 打印出关联规则

# %%
# test 
dataSet = loadDataSet()
dataSet.head()

# %%
dataSet.drop(['COD'], axis=1, inplace=True)
dataSet.head()

# %% [markdown]
# ### 在下面是成功的调试

# %%

# 将数据集转换为集合,如果对应项为0，则将这一列的名称不添加到集合中，如果为不为0，则添加到集合中
# 假设df是Pandas读入的数据
# 创建一个空列表来存储每一行对应的集合

D = []
# 遍历每一行
for index, row in dataSet.iterrows():
    # 创建一个空集合来存储这一行的列名
    s = []
    # 遍历除了views之外的所有列
    for col in dataSet.columns[:-1]:
    # 如果这个数据是1，就把这列的序号加入到集合中
        if row[col] == 1:
            # 查看现在的是第几列 
            col = dataSet.columns.get_loc(col)
            s.append(col)
    # 把这一行对应的集合加入到列表中
    D.append(s)

D

# %%
def apriori_test(dataSet, minSupport = 0.2):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

# %%
print(type(D))
print(type(D[0]))

# %%
dataSet.head()

# %%
L, supportData = apriori_test(D,0.07) # 生成所有满足最小支持度的项集
print(L) # 打印出所有满足最小支持度的项集
print(supportData) # 打印出每个项集的支持度

# %%
rules = generateRules(L, supportData, minConf=0.7) # 生成关联规则
print(rules) # 打印出关联规则

# %% [markdown]
# ### 上面是成功的

# %%

C1 = createC1(D) # 创建候选项集
C1

# %%
L1, supportData0 = scanD(D, C1, 0.2) # 从候选项集中筛选出满足最小支持度的项集
print(L1) # 打印出满足最小支持度的项集

# %%
print(supportData0) # 打印出每个项集的支持度

# %%
L = [L1]
print(L)
print(len(L))
k = 2

while (len(L[k-2]) > 0): # 在L[k-2]中的项集的元素个数大于0时，继续循环
    Ck = aprioriGen(L[k-2], k)
    Lk, supK = scanD(D, Ck, 0.05)
    supportData.update(supK)
    L.append(Lk)
    k += 1 # k表示项集中元素的个数

# 去掉L中长度为1的
L = [L[i] for i in range(len(L)) if len(L[i]) > 1]

print(L) # 打印出所有满足最小支持度的项集
print (len(L)) # 打印出每个项集的支持度

# %%
print(supportData) # 打印出每个项集的支持度
print(len(supportData)) # 打印出项集的个数

# %%
# 获得规则关联集合

rules = generateRules(L, supportData, minConf=0.1) # 生成关联规则
print(rules) # 打印出关联规则

# %%
# test3

dataSet = loadDataSet()
dataSet.head()

dataSet.drop(['COD'], axis=1, inplace=True)
dataSet.head()

# %%
L, supportData = apriori(dataSet) # 生成所有满足最小支持度的项集
print(L) # 打印出所有满足最小支持度的项集
print(supportData) # 打印出每个项集的支持度


