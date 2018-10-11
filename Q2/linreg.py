import numpy as np
import csv
import math
import matplotlib.pyplot as plt
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
#values= np.dstack((normalisedX,y))


# Implement batch gradient descent
# Setting parameters
n = 0.001
epsilon = 0.0000001
theta = np.zeros(2)

def hTheta(x,t1,t0):
	return x*t1+t0

def jTheta(x,y,t1,t0):
	z1 = np.apply_along_axis(hTheta,0,x,t1,t0)
	z2 = z1-y
	z3 = z2*z2
	jval = np.sum(z3)
	return jval/2

jbefore = 0
jnext = jTheta(normalisedX,y,theta[1],theta[0])

#print(jnext)
# Method
while (abs(jbefore-jnext)>epsilon):
	jbefore = jnext
	for i in range(len(theta)):
		t = theta[i]
		p = y - hTheta(normalisedX,theta[1],theta[0])
		xij = 0
		if (i==0):
			xij=1
			theta[0]=theta[0]+(n*np.sum(p))
		else:
			theta[1]=theta[1]+(n*np.sum(p*normalisedX))
	jnext = jTheta(normalisedX,y,theta[1],theta[0])
	#print(jnext)

print("theta1=",theta[1])
print("theta0=",theta[0])
#print(jTheta(normalisedX,y,theta1,theta0))


# PART-2 : PLOTTING THE X-Y POINTS AND HYPOTHESIS FUNCTION LEARNED BY THE ALGORITHM IN THE PREVIOUS PART
plt.scatter(normalisedX,y)
plt.xlabel('normalised X')
plt.ylabel('Y')
plt.plot(normalisedX, theta[1]*normalisedX+theta[0], color='red')
plt.title('Unweighted Linear Regression: y ='+str(theta[1])+'x+'+str(theta[0]))
plt.show()





