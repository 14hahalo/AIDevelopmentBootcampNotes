
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

"""This Neuron predicts a probability of passing an exam from students' studying and sleeping time.

###Single Neuron
"""

x=np.array([10,7])      # 10 hours of studying nad 7 hours of sleep
w=np.array([0.4,0.3])   # Weights of parameters
b=-5                    # Bias
z=np.dot(w,x)+b         # Weighted sum
a=1/(1+np.exp(-z))      # Neurons' output (sigmoid)

print("Weighted sum z: ",round(z,2))
print("Output of neuron a :",round(a,2))

"""###Multiple Neuron"""

x=np.array([10,7])
w=np.array([[0.4,-0.2,0.1],   # Hidden layers
            [0.3,0.5,-0.1]
])
b=np.array([-5,-1,0])
z=x@w+b                       # Matrix dot product
a=1/(1+np.exp(-z))

print("Dimensions of hidden layers' matrix: ",w.shape)
print("Sum of layers:",np.round(z,2))
print("Output values of layers",np.round(a,2))

"""###Data Preparation


"""

import keras

(X_training,y_training),(X_test,y_test)=keras.datasets.mnist.load_data()

print("Training data values:",X_training.shape)

X_training=X_training/255
X_test=X_test/255

plt.imshow(X_training[1],cmap='gray')
plt.title('Label:'+str(y_training[1]))
plt.axis('off')
plt.show()

"""###Model Training"""

model=keras.Sequential([
    keras.Input(shape=(28,28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax'),
])

model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_training,y_training,epochs=3,batch_size=128,validation_split=0.1)

loss,accuracy=model.evaluate(
    X_test,y_test,verbose=0
)

print("Test accuracy:%",round(accuracy*100,1))