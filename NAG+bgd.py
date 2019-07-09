import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv("dataset.csv", index_col = 0)
x = np.array(dataset['x'])
y = np.array(dataset['y'])

def cost(m, t0, t1, x, y):
	return 1/(2*m) * sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])


def gradient_descent(gamma, learning_rate, x, y, iterations):
	#initialize theta
	t0 = 0
	t1 = 0

	#number of examples
	m = x.shape[0]

	#total error
	J = cost(m, t0, t1, x, y)
	velocity0 = 0
	velocity1 = 0
	loss = np.empty(iterations)
	count = [i for i in range(1, iterations+1)]

	for it in range(iterations):
		#Calculating the gradients
		grad0 = 1/m * sum([t0-gamma*velocity0 + t1*np.asarray([x[i]]) - y[i] for i in range(m)]) 
		grad1 = 1/m * sum([(t0 + (t1-gamma*velocity1)*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])
		velocity0 = gamma*velocity0 + learning_rate*grad0
		velocity1 = gamma*velocity1 + learning_rate*grad1
		
		#Updating the parameters
		t0 = t0 - velocity0
		t1 = t1 - velocity1
		

		loss[it] = cost(m, t0, t1, x, y)

	return count, loss, t0, t1


#Implementation

gamma = 0.9
alpha = 0.001
max_iter = 10000

count, loss, theta0, theta1 = gradient_descent(gamma, alpha, x, y, max_iter)

print('theta0 = ' + str(theta0))
print('theta1 = ' + str(theta1))

plt.figure(0)
plt.scatter(x, y, c = 'red')
plt.plot(x, theta0 + theta1 * x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset')
plt.savefig('output.png')
plt.show()

plt.figure(1)
plt.plot(count, loss)
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.savefig('loss.png')
plt.show()