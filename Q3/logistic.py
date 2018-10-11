import numpy as np
import csv
import math
import matplotlib.pyplot as plt
# Linear regression
# Importing data from csv file for X and Y
x1 = np.array([])
x2 = np.array([])
with open('../ass1_data/logisticX.csv',newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
		x1 = np.append(x1,row[0])
		x2 = np.append(x2,row[1])
	
y = np.array([])
with open('../ass1_data/logisticY.csv',newline='') as csvfileY:
	reader = csv.reader(csvfileY)
	for row in reader:
		y = np.append(y,row)

#Normalisation of X	

#Normalisation of X1 and X2

x1 = x1.astype(np.float)
x2 = x2.astype(np.float)
y = y.astype(np.float)
meanX1 = np.mean(x1)
varX1  = np.var(x1)
stddevX1 = math.sqrt(varX1)
xm1 = np.subtract(x1,meanX1)
normalisedX1 = np.divide(xm1,stddevX1)

meanX2 = np.mean(x2)
varX2  = np.var(x2)
stddevX2 = math.sqrt(varX2)
xm2 = np.subtract(x2,meanX2)
normalisedX2 = np.divide(xm2,stddevX2)

# Newton method
theta = np.zeros(3)

def hTheta(x1,x2,theta):
	z= x2*theta[2]+x1*theta[1]+theta[0]
	h = 1/(1+np.exp(-z))
	return h

def gradient(x1,x2,y,theta):
	z1 = np.apply_along_axis(hTheta,0,x1,x2,theta)
	z2 = y - z1
	g = np.zeros(len(theta))
	g[0] = np.sum(z2)
	g[1] = np.sum(z2*x1)
	g[2] = np.sum(z2*x2)
	return g

def log_likelihood(x1,x2,y,theta):
	return np.sum(y*np.log(hTheta(x1,x2,theta))+(1-y)*np.log(1-hTheta(x1,x2,theta)))


def hessian(x1,x2,theta):
	h = np.zeros((len(theta),len(theta)))
	z1 = np.apply_along_axis(hTheta,0,x1,x2,theta)
	tik = 0
	til = 0
	h[0][0]= -np.sum(z1*(1-z1))
	h[0][1]= -np.sum(z1*(1-z1)*x1)
	h[0][2]= -np.sum(z1*(1-z1)*x2)
	h[1][0]= h[0][1]
	h[1][1]= -np.sum(z1*(1-z1)*x1*x1)
	h[1][2]= -np.sum(z1*(1-z1)*x1*x2)
	h[2][0]= h[0][2]
	h[2][1]= h[1][2]
	h[2][2]= -np.sum(z1*(1-z1)*x2*x2)
	return h


#method

llthetanext= log_likelihood(normalisedX1,normalisedX2,y,theta)
llthetaprev= np.Infinity
epsilon = 0.0000000000001

while (abs(llthetaprev-llthetanext)>epsilon):
	g = gradient(normalisedX1,normalisedX2,y,theta)
	h = hessian(normalisedX1,normalisedX2,theta)
	hinverse = np.linalg.inv(h)
	theta = theta - hinverse.dot(g)
	llthetaprev = llthetanext
	llthetanext = log_likelihood(normalisedX1,normalisedX2,y,theta)

print("theta2=",theta[2])
print("theta1=",theta[1])
print("theta0=",theta[0])




# PLOTTING THE TRAINING DATA AND DECISION BOUNDARY FIT BY LOGISTIC REGRESSION
fig, ax = plt.subplots()
t0x1 = np.array([])
t0x2 = np.array([])
t1x1 = np.array([])
t1x2 = np.array([])

#ax.scatter(normalisedX1,normalisedX2)
for i in range(len(normalisedX1)):
	if (y[i]==0):
		t0x1=np.append(t0x1,normalisedX1[i])
		t0x2=np.append(t0x2,normalisedX2[i])
	else :
		t1x1=np.append(t1x1,normalisedX1[i])
		t1x2=np.append(t1x2,normalisedX2[i])


ax.scatter(t0x1, t0x2, c="red",label="y(i)=0")
ax.scatter(t1x1, t1x2, c="green",label="y(i)=1")

plt.plot(normalisedX1, (-theta[0]/theta[2])+((-theta[1]/theta[2])*normalisedX1), color='blue')
plt.xlabel('normalised X1')
plt.ylabel('normalised X2')
plt.title('Decision boundary: '+str(theta[2])+'*x2 +'+str(theta[1])+'*x1+'+str(theta[0])+'=0')
	
ax.legend()		
plt.show()



