import src1

f = open('DecisionTree/input/lenses.txt')
a = f.readlines()
a[0].split('\n')[:-1][0].split('\t')

def datapro(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.replace('\t', ' ').split()
            data.append(fields)
    return data

labels = ['age of the patient','spectacle prescription','astigmatic','tear production rate']
data = datapro('DecisionTree/input/lenses.txt')
featureL = []
print(src1.createTree(data,labels,featureL))