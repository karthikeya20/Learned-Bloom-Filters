
from bloom_filter import BloomFilter
import tensorflow as tf 
import tensorflow_hub as hub
import numpy as np
X_train = []
y_train = []
def gen_data():
    # print("andar")
    fs = open("../dataset/test_input.txt", "r",encoding='utf-8')
    X = []
    y = []
    # print(fs.readlines())
    for i, line in enumerate(fs.readlines()):
        # print(line)
        url = line[:-5]
        label = line[-5:-1]
        # url, label = line.split(',')
        url.strip(',')
        url.strip("")
        X.append(url)
        y.append(label)
        if label==",bad":
            y.append(0)
        else:
            y.append(1)
    X = X[1:]
    y = y[1:]
    # print(X[:50])
    data = []
    for i in range(len(X)):
        data.append([X[i], y[i]])
    # print(data)
    return data

# data = [("a",0),("b",1)]
# X_train, y_train
data = gen_data()
size_of_dataset = len(data)
print(size_of_dataset)
classifier_data = []
bloom1 = BloomFilter(max_elements=size_of_dataset, error_rate=0.01)
bloom2 = BloomFilter(max_elements=size_of_dataset//2, error_rate=0.01)
embedding_size = 10

for data_point in data:
    if data_point[1]==1:
        bloom1.add(data_point[0])
        
for data_point in data:
    if data_point[0] in bloom1:
        classifier_data.append(data_point)

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Flatten(input_shape=(embedding_size,)))
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=50, batch_size=1)
# y_pred = model.predict(X_test)

# test_loss, test_acc = model.evaluate(X_test, y_test)


def training(model,bloom,data,tau):
    # gen_data()
    print(np.array(data).shape)
    X_train=np.array(data)[:,0]
    y_train=np.array(data)[:,1]
    model.fit(X_train, y_train, epochs=50, batch_size=1)
    preds=model.predict(X_train)
    for i in range(len(preds)):
        if preds[i]<tau:
            if y_train[i]==1:
                bloom.add(X_train[i])
    return bloom,model

def testing(model,bloom1,bloom2,data,tau):
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
    

bloom2,model=training(model,bloom2,classifier_data,0.5)
print(50)
