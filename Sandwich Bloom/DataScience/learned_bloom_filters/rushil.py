from bloom_filter import BloomFilter
import tensorflow as tf 
from tensorflow import keras
import numpy as np

bloom = BloomFilter(max_elements=10000, error_rate=0.01)


bloom.add("test-key")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
	keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

y_pred = model.predict(X_test)
# test_loss, test_acc = model.evaluate(X_test, y_test)

def training(model,bloom,data,tau):
    X_train=np.array(data)[0,:]
    y_train=np.array(data)[1,:]
    model.fit(X_train, y_train, epochs=50, batch_size=1)
    preds=model.predict(X_train)
    for i in range(len(preds)):
        if preds[i]<tau:
            if y_train[i]==1:
                bloom.add(X_train[i])
    return bloom,model


bloom2,model=training(model,bloom2,classificatiion_data,0.5)

def testing(data,tau):
    output=[]
    for i in range(len(data)):
        #Bloom1
        if data[i] not in bloom1:
            output.append(0)
            continue
        #Model
        output=model.predict(data[i])
        if output>tau:
            output.append(1)
        elif data[i] in bloom2:
            output.append(1)
        else:
            output.append(0)
    return output
            


