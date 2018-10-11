import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
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
print('PART-(A) :')
print('mu0=',mu0)
print('mu1=',mu1)

# Finding the covariance assuming both the classes have the same covariance matrix
summation=np.zeros((2,2))
for i in range(len(y)):
	if (y[i][0]=='Alaska'):
		diff = np.array([[normalisedX1[i],normalisedX2[i]]]) - mu0
		val = np.dot(diff.transpose(),diff)
		summation+=val
	elif (y[i][0]=='Canada'):
		diff = np.array([[normalisedX1[i],normalisedX2[i]]]) - mu1
		val = np.dot(diff.transpose(),diff)
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

print('PART-(B) :')


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


# PART C
# Describe the equation of theboundary separating the two regions in terms of the parameters μ0,μ1 and Σ.
print('PART-(C) :')
#RHS
sigmaInv = np.linalg.inv(covariance)
si = count1/count
logsi = np.log(si/(1-si))
musum = mu1+mu0
mudiff = mu1-mu0
prod1 = np.dot(musum,sigmaInv.transpose())
prod2 = np.dot(prod1,mudiff.transpose())
prod3 = prod2/2
a0 = logsi - prod3

#LHS 
mudiff = mu1 - mu0
prod = np.dot(sigmaInv,mudiff)
# final
intercept = -a0/prod[1]
slope = -prod[0]/prod[1]


fig, ax = plt.subplots()
ax.scatter(px1Alaska, px2Alaska, c="red",label="y(i)=Alaska")
ax.scatter(px1Canada, px2Canada, c="green",label="y(i)=Canada")
plt.xlabel('normalised X1')
plt.ylabel('normalised X2')
plt.plot(normalisedX1, slope*normalisedX1+intercept, color='blue')
plt.title('Decision boundary: '+str(prod[1])+'*x2 +'+str(prod[0])+'*x1+'+str(a0)+'=0')
ax.legend()		
plt.show()


# PART D
print('PART-(D) :')
print('mu0=',mu0)
print('mu1=',mu1)
# To calculate sigma0 and sigma1 now
summation0 = np.zeros((2,2))
summation1 = np.zeros((2,2))
for i in range(len(y)):
	if (y[i][0]=='Alaska'):
		diff = np.array([[normalisedX1[i],normalisedX2[i]]]) - mu0
		val = np.dot(diff.transpose(),diff)
		summation0+=val
	elif (y[i][0]=='Canada'):
		diff = np.array([[normalisedX1[i],normalisedX2[i]]]) - mu1
		val = np.dot(diff.transpose(),diff)
		summation1+=val

sigma0 = summation0/count0
sigma1 = summation1/count1

print('sigma0=',sigma0)
print('sigma1=',sigma1)



def z_func(x,y):
 return (1-(x**2+y**3))*np.exp(-(x**2+y**2)/2)
# PART E
print('PART-(E) :')
C = (2*logsi)-np.log(np.linalg.det(sigma1)/np.linalg.det(sigma0))
print(C)
sigma000=sigma0[0][0]
sigma001=sigma0[0][1]
sigma010=sigma0[1][0]
sigma011=sigma0[1][1]
sigma100=sigma1[0][0]
sigma101=sigma1[0][1]
sigma110=sigma1[1][0]
sigma111=sigma1[1][1]
mu00=mu0[0]
mu01=mu0[1]
mu10=mu1[0]
mu11=mu1[1]
# To plot the quadratic boundary 
fig = plt.figure()
#ax = plt.axes(projection='3d') # uncomment this to see in 3D
ax=fig.add_subplot(111)   # comment this to see in 3D
ax.scatter(px1Alaska, px2Alaska, c="red",label="y(i)=Alaska")
ax.scatter(px1Canada, px2Canada, c="green",label="y(i)=Canada")
ax.set_xlabel('normalisedX1')
ax.set_ylabel('normalisedX2')
y, x = np.ogrid[-2:2:100j, -2:2:100j]
d1=np.linalg.det(sigma1)
d0=np.linalg.det(sigma0)
plt.contour(x.ravel(),y.ravel(),(((sigma111*(x-mu10)*(x-mu10))+(sigma100*(y-mu11)*(y-mu11))-(2*sigma101*(y-mu11)*(x-mu10)))/d1)-(((sigma011*(x-mu00)*(x-mu00))+(sigma000*(y-mu01)*(y-mu01))-(2*sigma001*(y-mu01)*(x-mu00)))/d0),[C],alpha=1)
# plt.contour(x.ravel(),y.ravel(),x*x+y*y,[C],alpha=0.1)
# Exact equation
equation1='((('+str(sigma111)+'*(x1-'+str(mu10)+')^2)+'+'('+str(sigma100)+'*(x2-'+str(mu11)+')^2)-('+str(2*sigma101)+'*(x2-'+str(mu11)+')*(x1-'+str(mu10)+')))/'+str(d1)+')'
equation2='((('+str(sigma011)+'*(x1-'+str(mu00)+')^2)+'+'('+str(sigma000)+'*(x2-'+str(mu01)+')^2)-('+str(2*sigma001)+'*(x2-'+str(mu01)+')*(x1-'+str(mu00)+')))/'+str(d0)+')'
lhsequation = equation1+'-'+equation2
rhsequation=str(C)
print('Quadratic boundary equation:')
finalequation = lhsequation+'='+rhsequation
print(finalequation)

plt.title('Quadratic Decision boundary')
plt.plot(normalisedX1, slope*normalisedX1+intercept, color='orange',alpha=1)
plt.show()


