# Improvements

## PreLens.py
~~Origin~~
<pre><code>import src

f = open('DecisionTree/input/lenses.txt')
a = f.readlines()
a[0].split('\n')[:-1][0].split('\t')

def datapro(filepath):
    data = []
    with open(filepath,'r') as f:
        files = f.readlines()
        for line in files:
            line = line.split('\n')[:-1][0].split('\t')
            data.append(line)
    return data

labels = ['age of the patient','spectacle prescription','astigmatic','tear production rate']
data = datapro('DecisionTree/input/lenses.txt')
featureL = []
print(src.createTree(data,labels,featureL))
</code></pre>

__Improved__

<pre><code>import src

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
print(src.createTree(data,labels,featureL))</code></pre>