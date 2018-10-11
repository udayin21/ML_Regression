import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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

c=60
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('jTheta')
plt.title('Mesh and trajectory after each iteration of gradient descent')
xx = np.linspace(0,1.5,c)
yy = np.linspace(-0.5,1,c)
X, Y = np.meshgrid(xx,yy)
# Evaluating jval
mX = X.reshape((c*c,1))
mY = Y.reshape((c*c,1))
z = np.empty((c*c,1))
for i in range(c*c):
	fX = mX[i][0]
	fY = mY[i][0]
	JVAL = jTheta(normalisedX,y,fY,fX)
	z[i][0] = JVAL

z=z.reshape((c,c))
ax.plot_surface(X,Y,z,alpha=0.7)
plt.ion()
ax.scatter([theta[0]],[theta[1]],[jnext])
plt.draw()
plt.pause(1)
prevtheta1 = theta[1]
prevtheta0 = theta[0]
prevjnext = jnext
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
	ax.scatter([theta[0]],[theta[1]],[jnext])
	ax.plot([theta[0]],[theta[1]],[jnext])
	plt.plot([theta[0],prevtheta0],[theta[1],prevtheta1],[jnext,jbefore])
	prevtheta0 = theta[0]
	prevtheta1 = theta[1]
	prevjnext = jnext
	plt.draw()
	plt.pause(0.2)

	#print(jnext)


#ax.plot([theta[1],theta[0]],[theta[0],theta[1]],[jnext,jnext])

print("theta1=",theta[1])
print("theta0=",theta[0])
#print(jTheta(normalisedX,y,theta1,theta0))










