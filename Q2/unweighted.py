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


# Normal Equation method
o = np.zeros(len(normalisedX))
for i in range(len(o)):
	o[i]=1
X = np.array(list(zip(o,normalisedX)))

XTX = np.dot(X.transpose(),X)
XTXi = np.linalg.inv(XTX)
XTXiXT = np.dot(XTXi,X.transpose())
theta = np.dot(XTXiXT,y) 

print("theta1=",theta[1])
print("theta0=",theta[0])


# PART-2 : PLOTTING THE X-Y POINTS AND HYPOTHESIS FUNCTION LEARNED BY THE ALGORITHM IN THE PREVIOUS PART
plt.scatter(normalisedX,y)
plt.xlabel('normalised X')
plt.ylabel('Y')
plt.plot(normalisedX, theta[1]*normalisedX+theta[0], color='red')
plt.title('y ='+str(theta[1])+'x+'+str(theta[0]))
plt.show()




