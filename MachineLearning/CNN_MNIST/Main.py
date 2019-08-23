# Open training file and read it
with open('train.csv', 'r')as file:
    csv_lines = file.readlines()


# Store pictures and labels
import numpy as np
pic = []
label = []
for i in range(1, len(csv_lines)):
    row = csv_lines[i].replace('\n', '').split(',')
    pic.append(np.array(list(map(int, row[1:]))).reshape(28, 28, 1))
    label.append(list(map(int, row[0])))


# One-hot encoding
from keras.utils import to_categorical
label = to_categorical(label, num_classes=10)


# Transform into numpy array, and normalize
pic = np.array(pic) / 255.0
label = np.array(label)

print(pic.shape)
print(label.shape)


# Build the model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D

# Declare a sequential model
model = Sequential()

# Add a convolution layer (because it's the first layer, input shape is needed)
model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    input_shape=(28, 28, 1),
    activation='relu',
    padding='same'
))

# Dropout
model.add(Dropout(0.2))

# Pooling - 2*2
model.add(MaxPool2D(pool_size=(2, 2)))

# Add a convolution layer 
model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    padding='same'
))

# Dropout
model.add(Dropout(0.2))

# Pooling - 2*2
model.add(MaxPool2D(pool_size=(2, 2)))

# Add a flatten layer (transform data back to one-dimension for DNN)
model.add(Flatten())

# Dropout
model.add(Dropout(0.25))

# MLP
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

# Print out the summary of the model
print(model.summary())


# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Start training
history = model.fit(
    pic,
    label,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
    verbose=1
)


# Evaluation
import matplotlib.pyplot as plt
def show_train_history(train_history):
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['val_acc'])
    plt.xticks([i for i in range(0, len(train_history.history['acc']))])
    plt.title('Train History')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()

show_train_history(history)


# Submit the test result to submit.csv

# The format of the output, also the title of the form
submit='ImageId,Lable\n'

# Open the testing file
with open('test.csv', 'r')as file:
    csv_lines = file.readlines()

# Count for row
image_id = 1

for i in range(1, len(csv_lines)):

    # delete '\n', and split by ',' into lists
    row = csv_lines[i].replace('\n', '').split(',')

    # do predict (format must be same as fit())
    img = np.array(list(map(int, row))).reshape(28, 28, 1)
    result = model.predict_classes(np.array([img])/255.0)[0]
    submit += str(image_id) + ',' + str(result) + '\n'
    image_id += 1

# Write file
open('submit.csv', 'w').write(submit)