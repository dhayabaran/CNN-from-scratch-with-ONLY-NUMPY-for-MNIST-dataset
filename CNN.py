import numpy as np
import h5py
import copy
from random import randint


MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()


total_inputs = len(x_train)
output_size = 10
num_filters=5
filter_size=3
input_dim=28
input_size=28*28


K = np.random.randn(filter_size,filter_size, num_filters)/ np.sqrt(filter_size)
W = np.random.randn(output_size,input_dim-filter_size+1,input_dim-filter_size+1,num_filters) / np.sqrt(input_dim-filter_size+1)
bias= np.zeros((output_size, 1))/np.sqrt(output_size)

#K = np.random.randn(num_filters, filter_size,filter_size) / np.sqrt(input_size)
#W = np.random.randn(output_size,input_dim-filter_size+1,input_dim-filter_size+1,num_filters)


def conv(x,K):
	#Z=np.zeros((input_dim-filter_size+1,input_dim-filter_size+1,num_filters))
	Z=np.zeros(((x.shape[0]-K.shape[0]+1),(x.shape[0]-K.shape[0]+1),K.shape[2]))
	
	for p in range(K.shape[2]):
		for i in range(Z.shape[0]):
			for j in range(Z.shape[1]):
					if (i+3<Z.shape[0] and j+3<Z.shape[1]):
						x_temp = x[i:i+3,j:j+3]
						temp=np.multiply(x_temp,K[:,:,p])
						Z[i,j,p]=np.sum(temp)
	return Z

def convert_y(y):
    arr = np.zeros((output_size,1))
    arr[y] = 1
    return arr

def softmax_function(z):
    ZZ = np.exp(z - max(z))/np.sum(np.exp(z - max(z)))
    return ZZ

def Relu(z):
    return np.maximum(z,0)

def gradient_Relu(z):
    return np.where(z>0,1,0)

LR = .01
num_epochs = 10

for epochs in range(num_epochs):

    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001

    total_correct = 0

    for n in range(len(x_train)):
        n_random = randint(0,len(x_train)-1)
        y = y_train[n_random]
        x = x_train[n_random][:]
        x = np.reshape(x, (input_dim, input_dim))

        #forward propagation

        Z=conv(x,K)
        H=Relu(Z)
        
        U=np.zeros((output_size,1))
        for i in range(output_size):
        	temp1=W[i,:,:,:]
        	temp2=np.multiply(temp1,H)
        	U[i]=np.sum(temp2) + bias[i]
        
        rho = softmax_function(U)
        predicted_value = np.argmax(rho)

        if (predicted_value == y):
            total_correct += 1

        #backward propagation
        
        diff_U = rho - convert_y(y)
        diff_bias = diff_U
        
        diff_W=np.zeros((output_size,input_dim-filter_size+1,input_dim-filter_size+1,num_filters))
        for i in range(output_size):
        	diff_W[i,:,:,:]=diff_U[i]*H


        delta=np.zeros(H.shape)
        for i in range(input_dim-filter_size+1):
        	for j in range(input_dim-filter_size+1):
        		for p in range(num_filters):
       				delta[i,j,p]=np.sum(np.multiply(diff_U,W[:,i,j,p]))

       	grad_Zdel = np.multiply(gradient_Relu(Z),delta)
       	diff_K = conv(x, grad_Zdel)


       	#parameter updation

       	bias=bias - LR*diff_bias
       	W = W - LR*diff_W
       	K = K - LR*diff_K

    print("Training accuracy for epoch {} : {}".format(epochs+1, total_correct/np.float(len(x_train))))

#test data
total_correct = 0
for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    x = np.reshape(x, (input_dim, input_dim))

    Z=conv(x,K)
    H=Relu(Z)
        
    for i in range(output_size):
       	temp1=W[i,:,:,:]
        temp2=np.multiply(temp1,H)
        U[i]=np.sum(temp2) + bias[i]
        
    rho = softmax_function(U)
    predicted_value = np.argmax(rho)

    if (predicted_value == y):
        total_correct += 1

print("Test accuracy : {}".format(total_correct/np.float(len(x_test))))