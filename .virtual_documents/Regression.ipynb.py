import numpy as np
import pandas as pd
import scipy as sp
get_ipython().run_line_magic("matplotlib", " widget")
import copy as cp
import math
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler


## Load the data
data_train = pd.read_parquet("Data/train_data.parquet",engine="pyarrow")
data_test = pd.read_csv("Data/test_data.csv")
data_train['date'] =  pd.to_datetime(data_train['date'])


#Choose one sku
sku = np.random.choice(data_train.sku.unique(),1)[0]
sku = 328494
sku = 659780

data = data_train[data_train.sku == sku]

y = np.cumsum(data.sold_quantity.values)
x = np.arange(y.size) 
x = x.reshape(-1,1)

plt.close("all")
plt.figure()
plt.scatter(x,y)
plt.xlabel("days")
plt.ylabel("sold_quantity comulative")


## En principio no utilizamos MLE para fitear, no sé que metodo usa linear_fit. Checkear

clf = linear_model.PoissonRegressor(alpha = 1, max_iter=1000)
clf.fit(x,y)


def Poisson(y:np.array,x,intercept,c):
    """ Returns the poisson distribution for the probability of get y events given x with lambda = intercept + c*x
    Parameters:
        y: np.array
        x: int, features  of the model
        intercept: float
        c: float
    Returns:
        Probability distribution array"""
  
    lamb = np.exp(intercept + x*c)
    a = lamb**y*np.exp(-lamb)
    b = [math.factorial(i) for i in y]
    return a/b

def pdf_n_days(dist_prob,n):
    """Returns the pdf after n iterations given the distribution probability of one iteration
    This is assuming that every iteration is independent of the previous one. So in this case the pdf is the convolution.
    Paraments:
    dist_prob (np.array): distribuion probability of one iteration
    n (integer): number of iterations
    
    Returns:
    probability(np.array): pdf after n iterations"""
    prob = cp.copy(dist_prob)
    for i in range(n):
        prob = np.convolve(prob,dist_prob)
    return prob




print("score = ",clf.score(x,y))
print("coefficients = ", clf.intercept_, clf.coef_)
fig, axes = plt.subplots(1,1,figsize = (10,10))
axes.scatter(x,y,label = "data")
axes.plot(x,np.exp(clf.intercept_ + clf.coef_[0]*x),"r",label = "model")

axes.set_ylabel("sold_quantity cumulative")
axes.set_xlabel("days")
axes.legend()


day_list = [5,15,30,45]
coef = clf.intercept_,clf.coef_[0]
sells = np.arange(100,dtype = int)

fig,axes = plt.subplots(1,1,figsize = (10,5))
dist_prob,bins = np.histogram(data.sold_quantity.values,bins = np.arange(0,5,1))
dist_prob = dist_prob/dist_prob.sum()


for day in day_list:
    y = Poisson(sells,day,coef[0],coef[1])
    axes.bar(sells,y,label = f"day = {day}",alpha = .7) 
    
axes.set_xlim(0,10)
axes.set_xlabel("sells")
axes.legend()
axes.set_ylabel("prob")


day_list = [5,15,30,45,50,60]
coef = clf.intercept_,clf.coef_[0]
sells = np.arange(100,dtype = int)

fig,axes = plt.subplots(2,2,figsize = (10,5))

dist_prob,bins = np.histogram(data.sold_quantity.values,bins = np.arange(0,5,1))
dist_prob = dist_prob/dist_prob.sum()

means_regression = []
means_convolution = []

for day in day_list:

    y = Poisson(sells,day,coef[0],coef[1])
    m = np.exp(coef[0] + coef[1]*day)
    means_regression.append(m)
    
    axes[0][0].bar(sells,y,label = f"day = {day}",alpha = .7)
    
    y_aux = pdf_n_days(dist_prob,day)
    m2 = np.sum(y_aux*np.arange(y_aux.size))
    means_convolution.append(m2)
    
    axes[0][1].bar(np.arange(y_aux.size),y_aux,label = f"day = {day}",alpha = .7)
    
axes[0][0].set_title("Poisson Regression")
axes[0][0].set_xlabel("sells")
axes[0][0].legend()
axes[0][0].set_ylabel("prob")

axes[0][1].set_title("Convolution")
axes[0][1].set_xlabel("sells")
axes[0][1].legend()
axes[0][1].set_ylabel("prob")

xlim = 40
axes[0][0].set_xlim(-0.5,xlim)
axes[0][1].set_xlim(-.5,xlim)

axes[1][0].scatter(means_regression,means_convolution)
axes[1][0].set_xlabel("Regression mean")
axes[1][0].set_ylabel("Convolution mean")

fig.tight_layout()


#Lets make an histogram

scores = np.array([])

for example in data_train.sku.unique()[:1000]:
    d = data_train[data_train.sku == example]
    Y = np.cumsum(d.sold_quantity.values)
    X = np.arange(Y.size) 
    X = X.reshape(-1,1)
    poisson_model= linear_model.PoissonRegressor(max_iter=10000,verbose = 0)
    poisson_model.fit(X,Y)
    score = poisson_model.score(X,Y)
    scores = np.append(scores,score)



plt.figure()
plt.hist(scores,density=False,bins = np.arange(0,1.1,.1),align="left",rwidth = .5)
plt.ylabel("counts")
plt.xlabel("Score")
plt.xticks(np.arange(0,1.1,.1))


S = 10 #stock

sells = np.arange(50) # Las ventas oscilan entre 0 y 12 según el histograma
days = np.arange(70) # Días aproximados


p_stock_out = []
p2_stock_out = []
for day in days:
    p = Poisson(sells,day,clf.intercept_,clf.coef_[0])
    p /= p.sum() ### OJO! Debo renormalizar la distribución para que siga siendo una probabilidad. Why? float problems?
    stock_out = np.sum(p[S:])
    p_stock_out.append(stock_out)
    
    p2 = pdf_n_days(dist_prob,day)
    p2 /= p2.sum()
    stock_out2 = np.sum(p2[S:])
    p2_stock_out.append(stock_out2)

plt.close("all")
fig,axes = plt.subplots(1,2,figsize = (10,5),num = f"Probability of stock out sku = {sku}, S = {S}")
fig.suptitle(f"sku = {sku}, S = {S}",fontsize = 15)

axes[0].plot(p_stock_out,label = "Regression")
axes[0].plot(p2_stock_out,label = "Convolution")
axes[0].set_title("$P(Y>S|x = day)$")

axes[1].plot(np.diff(p_stock_out),label = "Regression")
axes[1].plot(np.diff(p2_stock_out),label = "Convolution")
axes[1].set_title("$P(\mathrm{out\ of\ stock}|x = day)$")


axes[0].legend()
axes[1].legend()
axes[0].set_xlabel("Days")
axes[1].set_xlabel("Days")


def Train_poisson_model(s,max_iter = 5000,alpha = 1):
    y = np.cumsum(s.sold_quantity)
    x = np.arange(y.size) 
    
    
    data = np.array([x,y]).T
    trans = MinMaxScaler()
    data = trans.fit_transform(data).T
    x,y = data
    
    x = x.reshape(-1,1)
    
    clf = linear_model.PoissonRegressor(alpha = alpha,max_iter = max_iter)
    clf.fit(x,y)
    
    coef = [clf.intercept_,clf.coef_[0],trans.scale_]
    return coef


sku_grouped = data_train[data_train.sku.isin(np.random.choice(data_train.sku.unique(),1000))].groupby("sku")
model = sku_grouped.apply(lambda x:Train_poisson_model(x))


model.iloc[0]


plt.figure()
plt.plot(np.cumsum(data_train[data_train.sku == 659780].sold_quantity.values))






