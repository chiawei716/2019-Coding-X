with open('train.csv', 'r') as file:

    csv_lines=file.readlines()

x=[]

y=[]

for i in range(1, len(csv_lines)):

    #去掉換行符號並以逗號分割

    row=csv_lines[i].replace('\n', '').split(',')

    #去掉label欄位，並將字串轉為整數

    x.append(list(map(int, row[1:])))

    #抓出label欄位，並將字串轉為整數

    y.append(list(map(int, row[0])))

from keras.utils import to_categorical

#one-hot encoding

y=to_categorical(y, num_classes=10)

import numpy as np

#轉成np.array，正規化

x=np.array(x)/255.0

y=np.array(y)

print(x.shape)

print(y.shape)

from keras.models import Sequential

from keras.layers import Dense, Dropout

# 宣告這是一個 Sequential 循序性的深度學習模型

model = Sequential()

# 加入第一層hidden layer(512 neurons)

# 因為第一層hidden layer需連接input vector故需要在此指定input_shape, activation function

model.add(Dense(512, activation='relu', input_shape=(784,)))

model.add(Dense(512, activation='relu'))

# 指定 dropout比例

model.add(Dropout(0.2))

# 指定 第二層模型hidden layer(512 neurons)、activation function、dropout比例

# model.add(Dense(512, activation='relu'))

model.add(Dense(256, activation='relu'))



# model.add(Dropout(0.2))

# 指定 輸出層模型

model.add(Dense(10, activation='softmax'))

print(model.summary())

from keras.optimizers import RMSprop

# 指定 loss function, optimizier, metrics

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['acc'])


batch_size = 64

epochs = 10

# 指定 batch_size, epochs, validation 後，開始訓練模型

history = model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)


import matplotlib.pyplot as plt

def show_train_history(train_history):

    plt.plot(train_history.history['acc'])

    plt.plot(train_history.history['val_acc'])

    plt.xticks([i for i in range(0, len(train_history.history['acc']))])

    plt.title('Train History')

    plt.ylabel('acc')

    plt.xlabel('epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

show_train_history(history)


# 儲存模型

try:

    model.save_weights("mnist.h5")

    print("success")

except:

    print("error")


with open('test.csv', 'r') as file:

    csv_lines=file.readlines()

show_images=0

for i in range(1, len(csv_lines)):

    #去掉換行符號並以逗號分割

    row=csv_lines[i].replace('\n', '').split(',')

    #並將字串轉為整數，正規化並預測

    result=model.predict_classes(np.array([list(map(int, row))])/255.0)[0]

    #視覺化預測結果

    ax=plt.subplot(2, 5, (i%10) if i%10!=0 else 10)

    ax.imshow(255-np.array(list(map(int, row))).reshape(28,28).astype(np.uint8), cmap='gray')

    plt.title('result: '+str(result))

    plt.axis('off')

    show_images+=1

    if show_images==10:

        plt.show()

        plt.draw()

        show_images=0

plt.show()

plt.draw()