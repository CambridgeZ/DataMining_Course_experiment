# Apriori 算法

import numpy as np
import pandas as pd
import itertools
import math
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from itertools import chain
from itertools import groupby
from operator import itemgetter

# 1. 生成候选项集
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

# 2. 计算候选项集的支持度
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can] = 1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

# 3. 生成频繁项集
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport = 0.5):
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

# 4. 生成关联规则
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

# 5. 计算关联规则的置信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

# 6. 生成关联规则的强度
def calcStr(freqSet, H, supportData, brl, minStr=0.7):
    prunedH = []
    for conseq in H:
        str = supportData[freqSet]/(supportData[freqSet-conseq]*supportData[conseq])
        if str >= minStr:
            print(freqSet-conseq, '-->', conseq, 'str:', str)
            brl.append((freqSet-conseq, conseq, str))
            prunedH.append(conseq)
    return prunedH

# 7. 生成关联规则的提升度
def calcLift(freqSet, H, supportData, brl, minLift=1.0):
    prunedH = []
    for conseq in H:
        lift = supportData[freqSet]/(supportData[freqSet-conseq]*supportData[conseq])
        if lift >= minLift:
            print(freqSet-conseq, '-->', conseq, 'lift:', lift)
            brl.append((freqSet-conseq, conseq, lift))
            prunedH.append(conseq)
    return prunedH

# 8. 生成关联规则的卡方值
def calcChi(freqSet, H, supportData, brl, minChi=0.7):
    prunedH = []
    for conseq in H:
        chi = (supportData[freqSet]*supportData[freqSet-conseq]*supportData[conseq])/(supportData[freqSet]*supportData[freqSet-conseq]*supportData[conseq])
        if chi >= minChi:
            print(freqSet-conseq, '-->', conseq, 'chi:', chi)
            brl.append((freqSet-conseq, conseq, chi))
            prunedH.append(conseq)
    return prunedH

# 9. 生成关联规则的互信息
def calcMI(freqSet, H, supportData, brl, minMI=0.7):
    prunedH = []
    for conseq in H:
        MI = supportData[freqSet]/(supportData[freqSet-conseq]*supportData[conseq])
        if MI >= minMI:
            print(freqSet-conseq, '-->', conseq, 'MI:', MI)
            brl.append((freqSet-conseq, conseq, MI))
            prunedH.append(conseq)
    return prunedH

# 10. 生成关联规则的Jaccard系数
def calcJac(freqSet, H, supportData, brl, minJac=0.7):
    prunedH = []
    for conseq in H:
        Jac = supportData[freqSet]/(supportData[freqSet-conseq]+supportData[conseq]-supportData[freqSet])
        if Jac >= minJac:
            print(freqSet-conseq, '-->', conseq, 'Jac:', Jac)
            brl.append((freqSet-conseq, conseq, Jac))
            prunedH.append(conseq)
    return prunedH

# 11. 生成关联规则的Odds比
def calcOdds(freqSet, H, supportData, brl, minOdds=0.7):
    prunedH = []
    for conseq in H:
        Odds = supportData[freqSet]/(supportData[freqSet-conseq]*supportData[conseq])
        if Odds >= minOdds:
            print(freqSet-conseq, '-->', conseq, 'Odds:', Odds)
            brl.append((freqSet-conseq, conseq, Odds))
            prunedH.append(conseq)
    return prunedH

# 12. 生成关联规则的余弦相似度
def calcCos(freqSet, H, supportData, brl, minCos=0.7):
    prunedH = []
    for conseq in H:
        Cos = supportData[freqSet]/(math.sqrt(supportData[freqSet-conseq]*supportData[conseq]))
        if Cos >= minCos:
            print(freqSet-conseq, '-->', conseq, 'Cos:', Cos)
            brl.append((freqSet-conseq, conseq, Cos))
            prunedH.append(conseq)
    return prunedH

# 13. 生成关联规则的杰卡德相似度
def calcJacard(freqSet, H, supportData, brl, minJacard=0.7):
    prunedH = []
    for conseq in H:
        Jacard = supportData[freqSet]/(supportData[freqSet-conseq]+supportData[conseq]-supportData[freqSet])
        if Jacard >= minJacard:
            print(freqSet-conseq, '-->', conseq, 'Jacard:', Jacard)
            brl.append((freqSet-conseq, conseq, Jacard))
            prunedH.append(conseq)
    return prunedH

# 14. 生成关联规则的皮尔逊相关系数
def calcPearson(freqSet, H, supportData, brl, minPearson=0.7):
    prunedH = []
    for conseq in H:
        Pearson = supportData[freqSet]/(math.sqrt(supportData[freqSet-conseq]*supportData[conseq]))
        if Pearson >= minPearson:
            print(freqSet-conseq, '-->', conseq, 'Pearson:', Pearson)
            brl.append((freqSet-conseq, conseq, Pearson))
            prunedH.append(conseq)
    return prunedH

# 15. 生成关联规则的信息增益
def calcGain(freqSet, H, supportData, brl, minGain=0.7):
    prunedH = []
    for conseq in H:
        Gain = supportData[freqSet]-supportData[freqSet-conseq]
        if Gain >= minGain:
            print(freqSet-conseq, '-->', conseq, 'Gain:', Gain)
            brl.append((freqSet-conseq, conseq, Gain))
            prunedH.append(conseq)
    return prunedH

# 16. 生成关联规则的信息增益比
def calcGainRatio(freqSet, H, supportData, brl, minGainRatio=0.7):
    prunedH = []
    for conseq in H:
        GainRatio = supportData[freqSet]/(supportData[freqSet-conseq]*supportData[conseq])
        if GainRatio >= minGainRatio:
            print(freqSet-conseq, '-->', conseq, 'GainRatio:', GainRatio)
            brl.append((freqSet-conseq, conseq, GainRatio))
            prunedH.append(conseq)
    return prunedH

# 17. 生成关联规则的置信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

# 18. 生成关联规则的Lift
def calcLift(freqSet, H, supportData, brl, minLift=0.7):
    prunedH = []
    for conseq in H:
        lift = supportData[freqSet]/(supportData[freqSet-conseq]*supportData[conseq])
        if lift >= minLift:
            print(freqSet-conseq, '-->', conseq, 'lift:', lift)
            brl.append((freqSet-conseq, conseq, lift))
            prunedH.append(conseq)
    return prunedH

# 19. 生成关联规则的卡方
def calcChi(freqSet, H, supportData, brl, minChi=0.7):
    prunedH = []
    for conseq in H:
        Chi = (supportData[freqSet]*supportData[freqSet-conseq])/(supportData[conseq]*supportData[freqSet])
        if Chi >= minChi:
            print(freqSet-conseq, '-->', conseq, 'Chi:', Chi)
            brl.append((freqSet-conseq, conseq, Chi))
            prunedH.append(conseq)
    return prunedH

# 20. 生成关联规则的互信息
def calcMI(freqSet, H, supportData, brl, minMI=0.7):
    prunedH = []
    for conseq in H:
        MI = supportData[freqSet]/(supportData[freqSet-conseq]*supportData[conseq])
        if MI >= minMI:
            print(freqSet-conseq, '-->', conseq, 'MI:', MI)
            brl.append((freqSet-conseq, conseq, MI))
            prunedH.append(conseq)
    return prunedH

# 21. 生成关联规则的支持度
def calcSup(freqSet, H, supportData, brl, minSup=0.7):
    prunedH = []
    for conseq in H:
        Sup = supportData[freqSet]
        if Sup >= minSup:
            print(freqSet-conseq, '-->', conseq, 'Sup:', Sup)
            brl.append((freqSet-conseq, conseq, Sup))
            prunedH.append(conseq)
    return prunedH

# 22. 生成关联规则的置信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


# test
if __name__ == '__main__':
    dataSet = loadDataSet()
    L, supportData = apriori(dataSet, minSupport=0.5)
    print(L)
    print(supportData)
    rules = generateRules(L, supportData, minConf=0.7)
    print(rules)

# output



    