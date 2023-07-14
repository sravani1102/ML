1. Find s
import numpy as np
import pandas as pd
df = pd.read_csv("ENJOYSPORT.csv")
df
h = ["" for i in range(len(df.columns)-1)]
for index, row in df.iterrows():
    ind = 0
    if(row["EnjoySport"] == 1):
        for i in range(len(row)-1):
            if h[ind] == "":
                h[ind] = row[i]s
            elif h[ind] != row[i]:
                h[ind] = "?"
            ind += 1
    print(h)
 
2.
#candidate algorithm
import pandas as pd
import numpy as np
df=pd.read_csv("ENJOYSPORT.csv")
col = list(df.columns)
print(col)
s = [0 for i in range(len(col)-1)]
g = [["?" for i in range(len(col) - 1)] for j in range(len(col) - 1)]
# print(s,g)
d = np.array(df)
# print(d)
for i in range(len(d)):
    for j in range(len(s)):
        if d[i,-1] == 1:
            if i == 0:
                s[j] = d[i,j]
            elif s[j] != d[i,j]:
                s[j] = "?"
                g[j][j] = "?"
        else:
            if s[j] != d[i,j]:
                g[j][j] = s[j]
            else:
                g[j][j] = "?"
#     print(s,"\n",g)
gen = ["?" for i in range(len(s))]
while(1):
    if gen in g: g.remove(gen)
    else: break
print("specific hypothesis:",s)
print("general hypothesis:",g)
#version space

def versionSpace(s,g):
    vs = []
    for each in g:
        for j in range(len(s)):
            if each[j] == "?" and s[j] != "?":
                dup = each[:]
                dup[j] = s[j]
                if dup not in vs:
                    vs.append(dup)
    return vs
vs =[]
print(versionSpace(s,g))
vs.extend(versionSpace(s,g))
vs.append(s)
vs.extend(g)
print("\nversion space:",vs)


3. # naive bais
import pandas as pd
import numpy as np
df=pd.read_csv('play_tennis.csv')
x_train=df[:9]
x_test=df[10:]
cy1=x_train.loc[x_train['play']=='Yes'].index.to_list()
cy=len(cy1)
#print(cy)
zc=x_train.index.to_list()
z=len(zc)
cn1=x_train.loc[x_train['play']=='No'].index.to_list()
cn=len(cn1)
#print(cn)
zn=x_train.index.to_list()
z=len(zn)
#a=np.array(x_test)
cls=df.columns
#print(cls)
d={}
for i in cls:
  new={}
  param=set(df[:][i])
  for j in param:
    ycount=0
    ncount=0
    for ind in range(0,10):
      if df.loc[ind,i]==j:
        if ind in cy1:
          ycount+=1
        if ind in cn1:
          ncount+=1
    new[j]=[ycount/10,ncount/10]
  d[i]=new
print(d)
pyes=cy/z
pno=cn/z
need=df.columns
need=list(need)
#print(need)
need.pop()
need.pop()
correct=0
wrong=0
for row in x_test.index:
  for col in need:
    now=df.loc[row,col]
    pyes*=d[col][now][0]
    pno*=d[col][now][1]
  if pyes>pno:
    if df.loc[row,'play']=='Yes':
      correct+=1
    else:
      wrong+=1
  else:
    if df.loc[row,'play']=='No':
      correct+=1
    else:
      wrong+=1
print((correct/3)*100)


4.# Text Classification

import pandas as pd
data = pd.read_csv("text_classification (1).csv", names= ["review", "classification"])
print(data)
n = 10
test_data = data[:10]
train_data= data[11:]
data

# probability of pos and neg
pos = (test_data["classification"] == "pos").sum()
neg = (test_data["classification"] == "neg").sum()
p_pos = pos/n
p_neg = neg/n
print(p_pos,p_neg)

def remove_dup(l):
  no_dup=[]
  for i in l:
    if i not in no_dup:
      no_dup.append(i)
  return no_dup

#tot vocabulary
s = []
for i in test_data["review"]:
  s.extend(i.split())
voc = remove_dup(s)
n_voc = len(voc)

# words in neg and pos riviews
pos_v = []
neg_v = []
for i in range(len(test_data["classification"])):
  if test_data["classification"][i] == "pos":
    pos_v.extend(test_data["review"][i].split())
  elif test_data["classification"][i] == "neg":
    neg_v.extend(test_data["review"][i].split())
pos_n = len(pos_v)
neg_n = len(neg_v)
# print(pos_v)
# print(neg_v)

# TESTING of test_data
pos_prod = p_pos
neg_prod = p_neg
res = []
for i in train_data["review"]:
  for word in i.split():
    pos_prod*=(pos_v.count(word)+1)/(pos_n+n_voc)
    neg_prod*=(neg_v.count(word)+1)/(neg_n+n_voc)
  if(pos_prod>neg_prod):
    res.append("pos")
  else: res.append("neg")
print(res)


tp,tn,fp,fn = 0,0,0,0
j =0
for i in train_data["classification"]:
  if i == res[j]:
    if i == 'pos':
      tp += 1
    else:
      tn += 1
  else:
    if(i == 'pos'):
      fn += 1
    else:
      fp += 1
  j += 1

# print(tp,tn,fp,fn)
acc = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
print("accuracy : ",acc,"\nprecision : ",precision,"\nrecall : ",recall)
6#decision tree
import pandas as pd
import math
data = pd.read_csv("play_tennis (1).csv")
del data['day']
target = 'play'

def entropy(d):
    pred = d[target]
    yes = len(pred[pred == 'Yes'])
    no = len(pred[pred == 'No'])
    yes /= (yes+no)
    no /= (yes+no)
    if(yes == 0):
        return -round(no*math.log2(no), 2)
    elif(no == 0):
        return -round(yes*math.log2(yes), 2)
    return -round(no*math.log2(no) + yes*math.log2(yes), 2)

def selectRoot(d):
    cols = d.columns[:-1]
    n = len(d)
    maxIG = 0
    ent = entropy(d)
    for col in cols:
        entAll = 0
        #finding all children entropy
        for val in d[col].unique():
            entAll += len(d[d[col] == val])/n*entropy(d[d[col] == val])
        #max information gain
        if(ent - entAll > maxIG):
            maxIG = ent - entAll
            root = col
    return root
tree = {}
d = data
root = selectRoot(d)
current_root = [root, d.index]
tree[root] = []
open_list = [current_root]
while open_list != []:
    current_root = open_list.pop(0)
    for val in data[current_root[0]].unique():
        #considering data rows based on index in d.index in d
        d = data.iloc[current_root[-1]]
        tree[current_root[0]].append(val)
        yes = len(d[(d[current_root[0]] == val) & (d[target] == 'Yes')])
        no = len(d[(d[current_root[0]] == val) & (d[target] == 'No')])
        if yes == 0:
            tree[val] = "no"
        elif no == 0:
            tree[val] = "yes"
        else:
#             considering data rows that hav only val - child
            d = d[d[current_root[0]] == val]
            root = selectRoot(d)
            tree[val] = [root]
            tree[root] = []
            open_list.append([root, d.index])
print("********************************final tree**********************************************")           
for i in tree:
    print(i ,": ",tree[i])


  7. # back propagation
import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0) #maximum of X array longitudinally
y = y/100

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
epoch=5 #Setting training iterations
lr=0.1 #Setting learning rate

inputlayer_neurons = 2 #number of features in data set
hiddenlayer_neurons = 1 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer
#weight and bias initialization

wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
print("wh:",wh)
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

#draws a random range of numbers uniformly of dim x*y
for i in range(epoch):
    #Forward Propogation
    hinp1=np.dot(X,wh)
    
    hinp=hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1=np.dot(hlayer_act,wout)
    outinp= outinp1+bout
    output = sigmoid(outinp)


    #Backpropagation
    EO = y-output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)#how much hidden layer wts contributed to error
    d_hiddenlayer = EH * hiddengrad

    wout += hlayer_act.T.dot(d_output) *lr   # dotproduct of nextlayererror and currentlayerop
    wh += X.T.dot(d_hiddenlayer) *lr

    print ("-----------Epoch-", i+1, "Starts----------")
    print("Input: \n" + str(X))
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n" ,output)
    print ("-----------Epoch-", i+1, "Ends----------\n")

print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)







8. # EM algorithm
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np

iris = datasets.load_iris()
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']

color = np.array(["red", "blue","green"])
model = KMeans(n_clusters = 3).fit(x)
plt.figure(figsize=(10, 7))
plt.subplot(2,3,1)
plt.scatter( x.Petal_Length,x.Petal_Width, c = color[y.Targets])
plt.title("actual clusters")
plt.xlabel("petal length")
plt.ylabel("petal width")
# plt.show()
plt.subplot(2,3,2)
plt.scatter( x.Petal_Length,x.Petal_Width, c = color[model.labels_])
plt.title("k means clusters")
plt.xlabel("petal length")
plt.ylabel("petal width")
gmm = GaussianMixture(n_components = 3).fit(x).predict(x)
plt.subplot(2,3,3)
plt.scatter( x.Petal_Length,x.Petal_Width, c = color[gmm])
plt.title("guassian clusters")
plt.xlabel("petal length")
plt.ylabel("petal width")
print('Observation: The GMM using EM algorithm based clustering matched the true labels more closely than the Kmeans.')
plt.show()
9. # KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix

iris = datasets.load_iris()
x_train, x_test, y_train_, y_test = train_test_split(iris.data, iris.target, test_size = 0.1)
clssifier = KNeighborsClassifier(5)
#training
classifier.fit(x_train, y_train)
#testing
pred_y = classifier.predict(x_test)
print("results are:")
for r in range(0, len(x_test)):
    print(" Sample:", str(x_test[r]), " Actual-label:", str(y_test[r]), "Predicted-label:", str(y_pred[r]))

print("confusion matrix:",confusion_matrix(y_test, pred_y))
print("Accuracy Metrics\n",classification_report(y_test, pred_y))

               10# Locally weightedRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('10-dataset.csv')
data


def kernal(point, x, k):
    m, n= np.shape(x)
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = point - x[j]
        weights[j,j] = np.exp(diff*diff.T/-(2*k*k))
    return weights
def localWeight(point, x, y, k):
    wt = kernal(point, x, k)
    w = (x.T*(wt*x)).I*(x.T*wt*y.T)
#     print(w)
    return w;
def localWeightedRegression(x, y, k):
    m, n= np.shape(x)
    y_pred = np.zeros(m)
    for i in range(len(x)):
        y_pred[i] += x[i]*localWeight(x[i], x, y, k)
    return y_pred

cola = np.array(data.total_bill)
colb = np.array(data.tip)
bill = np.mat(data.total_bill)
tip = np.mat(data.tip)
col = np.shape(bill)[1]
ones = np.ones((1,m), dtype = int)
d = np.hstack((ones.T, bill.T))
k = 0.8
ypred = localWeightedRegression(d, tip, k)
# print(ypred)
xsort = d.copy()
xsort.sort(axis = 0)
plt.scatter(cola,colb,color = "blue")
plt.plot(xsort[:,1],ypred[d[:,1].argsort(axis = 0)], color = "yellow", linewidth = 3)
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

# text classification
import pandas as pd
import numpy as np
df=pd.read_csv("text_classification (1).csv")
df_train=df[:10]
pos=""
neg=""
df_train=np.array(df_train)
vocab=[]
poscnt=0
negcnt=0
for i in range(len(df_train)):
  if df_train[i][1]=='pos':
    pos=pos+" "+df_train[i][0]
    poscnt+=1
  else:
    neg=neg+" "+df_train[i][0]
    negcnt+=1
  vocab.extend(df_train[i][0].split(" "))
vocab=set(vocab)
n_yes=len(set(pos.split(" ")))
n_no=len(set(neg.split(" ")))
d={}
for i in vocab:
  res=[]
  #computing for yes
  nk_yes=pos.count(i)
  res.append((nk_yes+1)/(n_yes+len(vocab)))
  nk_no=neg.count(i)
  res.append((nk_no+1)/(n_no+len(vocab)))
  d[i]=res
#preproccesing done
for i in d.keys():
  print(i,d[i])
df_test=df[11:]
df_test=np.array(df_test)
tp=0
fp=0
tn=0
fn=0
for i in range(len(df_test)):
  pyes=poscnt/10
  pno=negcnt/10
  words=df_test[i][0].split(" ")
  for word in words:
    if word in d.keys():
      pyes*=d[word][0]
      pno*=d[word][1]
    if pyes>pno:
      if df_test[i][1]=='pos':
        tp+=1
      else:
        fp+=1
    if pno>pyes:
      if df_test[i][1]=='neg':
        tn+=1
      else:
        fn+=1
precision=(tp)/(tp+fp)
recall=(tp)/(tp+fn)

print(precision)
print(recall)
