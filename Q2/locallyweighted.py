import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import sys

# Linear regression
# Importing data from csv file for X and Y
x = np.array([])
with open('../ass1_data/weightedX.csv',newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
		x = np.append(x,row)
	
y = np.array([])
with open('../ass1_data/weightedY.csv',newline='') as csvfileY:
	reader = csv.reader(csvfileY)
	for row in reader:
		y = np.append(y,row)

#Normalisation of X	

x = x.astype(np.float)
y = y.astype(np.float)
meanX = np.mean(x)
varX  = np.var(x)
stddevX = math.sqrt(varX)
xm = np.subtract(x,meanX)
normalisedX = np.divide(xm,stddevX)

# Locally weighted method
qp = np.mean(normalisedX)


def weight(xi,x,tau):
	z = xi-x
	z1 = z*z
	p =  np.exp(-z1/(2*tau*tau))
	return p


qpoints = np.linspace(np.amin(normalisedX),np.amax(normalisedX),num=20)
print(qpoints)

o = np.ones(len(normalisedX))
#parameters
X = np.array(list(zip(o,normalisedX)))
tau = float(sys.argv[1])

def locweightreg(X,normalisedX,y,x,tau):
	weights= weight(normalisedX,x,tau)
	W = np.zeros((len(normalisedX),len(normalisedX)))
	for i in range(len(normalisedX)):
		W[i][i]=weights[i]
	XTW = np.dot(X.transpose(),W)
	XTWX = np.dot(XTW,X)
	XTXi = np.linalg.inv(XTWX)
	XTXiXT = np.dot(XTXi,X.transpose())
	XTXiXTW = np.dot(XTXiXT,W)
	theta = np.dot(XTXiXTW,y)
	return theta


prediction=np.array([])
for q in qpoints:
	theta = locweightreg(X,normalisedX,y,q,tau)
	yval = theta[1]*q+theta[0]
	prediction = np.append(prediction,yval)


# PART-2 : PLOTTING THE X-Y POINTS AND HYPOTHESIS FUNCTION LEARNED BY THE ALGORITHM IN THE PREVIOUS PART
plt.scatter(normalisedX,y)
plt.xlabel('normalised X')
plt.ylabel('Y')
plt.plot(qpoints,prediction,'-',color='red')
plt.show()



