import numpy as np
import csv
import math
import matplotlib.pyplot as plt
# Linear regression
# Importing data from dat file for X and Y
x1 = np.array([])
x2 = np.array([])


datContent = [i.strip().split() for i in open("../ass1_data/q4x.dat").readlines()]
x1 = np.array([x[0] for x in datContent])
x2 = np.array([x[1] for x in datContent])

y = [i.strip().split() for i in open("../ass1_data/q4y.dat").readlines()]

# Normalisation

x1 = x1.astype(np.float)
x2 = x2.astype(np.float)
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

# PART 1
# Implement Gaussian Discriminant Analysis using the closed form equations described in class. 
#Assume that both the classes have the same co-variance matrix i.e. Σ0 = Σ1 = Σ. 
#Report the values of the means, μ0 and μ1, and the co-variance matrix Σ.
mu0 = np.zeros(2)
mu1 = np.zeros(2)
sum0 = np.zeros(2)
sum1 = np.zeros(2)
sumt = np.zeros(2)
count0 = 0
count1 = 0
count =0



# Finding the mean
for i in range(len(y)):
	if (y[i][0]=='Alaska'):
		count0+=1
		sum0+=np.array([normalisedX1[i],normalisedX2[i]])
	elif (y[i][0]=='Canada'):
		count1+=1
		sum1+=np.array([normalisedX1[i],normalisedX2[i]])
	
	count+=1
	sumt+=np.array([normalisedX1[i],normalisedX2[i]])	
	

mu0 = sum0/count0
mu1 = sum1/count1
print('mu0=',mu0)
print('mu1=',mu1)

# Finding the covariance assuming both the classes have the same covariance matrix
summation=np.zeros(2)
for i in range(len(y)):
	if (y[i][0]=='Alaska'):
		diff = np.array([normalisedX1[i],normalisedX2[i]]) - mu0
		val = np.dot(diff,diff.transpose())
		summation+=val
	elif (y[i][0]=='Canada'):
		diff = np.array([normalisedX1[i],normalisedX2[i]]) - mu1
		val = np.dot(diff,diff.transpose())
		summation+=val
		
m = len(y)		
covariance = summation/m
print('covariance=',covariance)


# PART 2
#Plot the training data corresponding to the two coordinates of the input features, and you should use a different symbol for each point plotted to 
#indicate whether that example had label Canada or Alaska.
px1Alaska = np.array([])
px1Canada = np.array([])
px2Alaska = np.array([])
px2Canada = np.array([])




for i in range(len(normalisedX1)):
	if (y[i][0]=='Alaska'):
		px1Alaska=np.append(px1Alaska,normalisedX1[i])
		px2Alaska=np.append(px2Alaska,normalisedX2[i])
	else :
		px1Canada=np.append(px1Canada,normalisedX1[i])
		px2Canada=np.append(px2Canada,normalisedX2[i])


fig, ax = plt.subplots()
ax.scatter(px1Alaska, px2Alaska, c="red",label="y(i)=Alaska")
ax.scatter(px1Canada, px2Canada, c="green",label="y(i)=Canada")
plt.xlabel('normalised X1')
plt.ylabel('normalised X2')
ax.legend()		
plt.show()


# PART 3
# Describe the equation of theboundary separating the two regions in terms of the parameters μ0,μ1 and Σ.
#RHS
si = count1/count
rhs = np.log(si/(1-si))
#LHS 










