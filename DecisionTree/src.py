import numpy as np
import operator

def calcShannonEnt(dataSet):
    """
    函数说明:计算给定数据集的经验熵(香农熵)
    Parameters:
        dataSet - 数据集
    Returns:
        shannonEnt - 经验熵(香农熵)
    """
    numEntires = len(dataSet)                        #返回数据集的行数
    labelCounts = {}                                #保存每个标签(Label)出现次数的字典
    for featVec in dataSet:                            #对每组数据进行统计
        currentLabel = featVec[-1]                    #提取标签(Label)信息
        if currentLabel not in labelCounts.keys():    #如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1                #Label计数
    shannonEnt = 0.0                                #经验熵(香农熵)
    for key in labelCounts:                            #计算香农熵
        prob = float(labelCounts[key]) / numEntires    #选择该标签(Label)的概率
        shannonEnt -= prob * np.log2(prob)            #利用公式计算
    return shannonEnt                                #返回经验熵(香农熵)

def splitDataSet(dataSet, axis, value):
    """
    函数说明:按照给定属性的取值划分数据集
    
    Parameters:
        dataSet - 待划分的数据集
        axis - 划分数据集的属性
        value - 需要返回的属性的值
    Returns:
        无
    Author:
        Jack Cui
    Modify:
        2017-03-30
    """
    retDataSet = []                                        #创建返回的数据集列表
    for featVec in dataSet:                             #遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                #去掉axis属性,先选取axis属性之前的元素
            reducedFeatVec.extend(featVec[axis+1:])     #再将axis属性之后的元素追加到reducedFeatVec之后
            retDataSet.append(reducedFeatVec)           #将符合条件的添加到返回的数据集
    return retDataSet                                      #返回划分后的数据集

def chooseBestFeatureToSplit(dataSet):
    """
    函数说明:按照信息增益选择最优属性
    
    Parameters:
        dataSet - 数据集
    Returns:
        bestFeature - 信息增益最大的(最优)属性的索引值
    """
    numFeatures = len(dataSet[0]) - 1                    #属性数量，去除最后一列标签
    baseEntropy = calcShannonEnt(dataSet)                 #计算数据集的香农熵
    bestInfoGain = 0.0                                  #初始信息增益
    bestFeature = -1                                    #初始最优属性的索引值
    for i in range(numFeatures):                         #遍历所有属性，计算数据集中所有属性的信息增益
        
        featList = [example[i] for example in dataSet]  #获取第i个属性的所有取值
        uniqueVals = set(featList)                         #创建set集合{},元素不可重复
        newEntropy = 0.0                                  #初始经验条件熵
        for value in uniqueVals:                         #计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)         #subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))           #计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)     #根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy                     #信息增益
        print("第%d个属性的增益为%.3f" % (i, infoGain))            #打印每个属性的信息增益
        if (infoGain > bestInfoGain):                             #计算信息增益
            bestInfoGain = infoGain                             #更新信息增益，找到最大的信息增益
            bestFeature = i                                     #记录信息增益最大的属性的索引值
    return bestFeature                                             #返回信息增益最大的属性的索引值

def majorityCnt(classList):
    """
    函数说明:在数据不能进行划分后需要统计classList中出现此处最多的元素(类标签)
    
    Parameters:
        classList - 类标签列表
    Returns:
        sortedClassCount[0][0] - 出现此处最多的元素(类标签)
    """
    classCount = {}
    for vote in classList:                                        #统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0   
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True) #根据字典的值降序排序
    return sortedClassCount[0][0]                                       #返回排序后的第一个键值对的键  

def createTree(dataSet, labels, featLabels):
    """
    函数说明:创建决策树
    
    Parameters:
        dataSet - 训练数据集
        labels - 分类属性标签
        featLabels - 存储选择的最优属性标签
    Returns:
        myTree - 决策树
    """
    classList = [example[-1] for example in dataSet]         #取分类标签(是否放贷:yes or no)          
    if classList.count(classList[0]) == len(classList):            #如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:                 #遍历完所有属性时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)                #选择最优属性
    bestFeatLabel = labels[bestFeat]                            #最优属性的标签
    featLabels.append(bestFeatLabel)                            #记录每一次划分所选择的属性
    myTree = {bestFeatLabel:{}}                                    #根据最优属性的标签生成树
    del(labels[bestFeat])                                        #删除已经使用属性标签
    featValues = [example[bestFeat] for example in dataSet]        #得到训练集中最优属性的所有取值
    uniqueVals = set(featValues)                                #去掉重复的属性值
    for value in uniqueVals:                                    #遍历该属性的所有取值，创建决策树。  
        subLabels = labels[:]               
        #传入该属性对应取值的数据集
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
    return myTree

# """
# age: 0:青年:1:中年:2:老年
# work:0:没有工作:1:有工作
# house:0:有房子:1:没有房子
# credit:0:信誉情况一般:1:信誉情况良好:2:信誉情况非常好
# 类别为是否同意贷款
# """
# def createDataSet():
#     dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
#                [0, 0, 0, 1, 'no'],
#                 [0, 1, 0, 1, 'yes'],
#                 [0, 1, 1, 0, 'yes'],
#                 [0, 0, 0, 0, 'no'],
#                 [1, 0, 0, 0, 'no'],
#                 [1, 0, 0, 1, 'no'],
#                 [1, 1, 1, 1, 'yes'],
#                 [1, 0, 1, 2, 'yes'],
#                 [1, 0, 1, 2, 'yes'],
#                 [2, 0, 1, 2, 'yes'],
#                 [2, 0, 1, 1, 'yes'],
#                 [2, 1, 0, 1, 'yes'],
#                 [2, 1, 0, 2, 'yes'],
#                 [2, 0, 0, 0, 'no']]
#     labels = ['age', 'work', 'house', 'credit']        #属性标签
#     return dataSet, labels                             #返回数据集和分类属性

# dataSet, labels = createDataSet()
# featLabels = []
# myTree = createTree(dataSet, labels, featLabels)
# print(myTree) 