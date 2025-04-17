import src1

"""
age: 0:青年:1:中年:2:老年
work:0:没有工作:1:有工作
house:0:有房子:1:没有房子
credit:0:信誉情况一般:1:信誉情况良好:2:信誉情况非常好
类别为是否同意贷款
"""
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
               [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
    labels = ['age', 'work', 'house', 'credit']        #属性标签
    return dataSet, labels                             #返回数据集和分类属性

dataSet, labels = createDataSet()
featLabels = []
myTree = src1.createTree(dataSet, labels, featLabels)
print(myTree)