import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys

# Linear regression
# Importing data from csv file for X and Y
x = np.array([])
with open('../ass1_data/linearX.csv',newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
		x = np.append(x,row)
	
y = np.array([])
with open('../ass1_data/linearY.csv',newline='') as csvfileY:
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
n = float(sys.argv[1])
epsilon = 0.000001
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



# c = 50
# theta0= np.linspace(0,2,c)
# theta1= np.linspace(-1,1,c)
# xx, yy = np.meshgrid(x, y, sparse=True)
# z = jTheta()

# fig =plt.figure()
# g = fig.add_subplot(111,projection='3d')
# g.set_xlabel('theta0')
# g.set_ylabel('theta1')
# g.set_zlabel('j(theta)')
# g.plot_surface(x,y,z)
# plt.show()



# c=60
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# xx = np.linspace(0,1.5,c)
# yy = np.linspace(-0.5,1,c)
# X, Y = np.meshgrid(xx,yy)
# Evaluating jval
# mX = X.reshape((c*c,1))
# mY = Y.reshape((c*c,1))
# z = np.empty((c*c,1))
# for i in range(c*c):
# 	fX = mX[i][0]
# 	fY = mY[i][0]
# 	JVAL = jTheta(normalisedX,y,fY,fX)
# 	z[i][0] = JVAL
# z=z.reshape((c,c))




prevtheta1 = theta[1]
prevtheta0 = theta[0]
prevjnext = jnext
#print(jnext)
# Method
a1= np.array([])
a2= np.array([])
a3= np.array([])


fig = plt.figure()
#ax = plt.axes(projection='3d') # uncomment this to see in 3D
ax=fig.add_subplot(111)   # comment this to see in 3D
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
plt.title('Contour plot after each iteration of gradient descent')
xx = np.linspace(0,2,50)
yy = np.linspace(-1,2,50)
X, Y = np.meshgrid(xx,yy)
c= 50
mX = X.reshape((c*c,1))
mY = Y.reshape((c*c,1))
z = np.empty((c*c,1))
for i in range(c*c):
	fX = mX[i][0]
	fY = mY[i][0]
	JVAL = jTheta(normalisedX,y,fY,fX)
	z[i][0] = JVAL
z=z.reshape((c,c))

plt.ion()
ii=0
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
	prevtheta0 = theta[0]
	prevtheta1 = theta[1]
	prevjnext = jnext
	ii+=1
	a1 = np.append(a1,prevtheta0)
	a2 = np.append(a2,prevtheta1)
	a3 = np.append(a3,jbefore)
	ax.contour(X,Y,z,np.unique(a3.tolist()),alpha=0.1)
	plt.draw()
	plt.pause(0.2)



plt.show()

print("theta1=",theta[1])
print("theta0=",theta[0])
#print(jTheta(normalisedX,y,theta1,theta0))

print("no. of iterations =",ii)








