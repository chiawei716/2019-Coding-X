# Open training file
with open('train.csv', 'r')as file:
    csv_lines = file.readlines()


# Access and store the data
pic = []
label = []
for i in range(1, len(csv_lines)):
    
    # delete '\n', and split by ',' into lists
    row = csv_lines[i].replace('\n', '').split(',')
    # except label
    pic.append(list(map(int, row[1:])))
    # only label
    label.append(list(map(int, row[0])))


# One-hot encoding
from keras.utils import to_categorical
label = to_categorical(label, num_classes = 10)


# Transform into numpy.array, and also normalize
import numpy as np
pic = np.array(pic)/255.0
label = np.array(label)

print(pic.shape)
print(label.shape)


# Build the model
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Declaration a sequential deep learning model
model = Sequential()
# Add the first layer(# of neurals, activation function, input shape)
model.add(Dense(512, activation='relu', input_shape=(784,)))
# Add second layer(# of neurals, activation function)s
model.add(Dense(512, activation='relu'))
# Dropout
model.add(Dropout(rate=0.2))
# Add third layer(same as second one)
model.add(Dense(256, activation='relu'))
# Add the output layer (use softmax to normalize the sum of results)
model.add(Dense(10, activation='softmax'))

print(model.summary())

# Compile the model
from keras.optimizers import RMSprop
# settings of the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['acc']
)

# Start training
batch_size = 64
epochs = 10
history = model.fit(
    pic,
    label,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
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
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(history)


# Save the model
try:
    model.save_weights('mnist.h5')
    print('saving sucess')
except:
    print('saving failed')


# Predict with the testing file

# Open the testing file
with open('test.csv', 'r')as file:
    csv_lines = file.readlines()

# Count for showing pictures
show_image = 0 

for i in range(1, len(csv_lines)):

    # delete '\n', and split by ',' into lists
    row = csv_lines[i].replace('\n', '').split(',')

    # do predict ( argument format should as same as fit() )
    result = model.predict_classes(np.array([list(map(int, row))])/255.0)[0]
    
    # setup the plot
    ax = plt.subplot(2, 5, (i % 10) if i % 10 != 0 else 10)
    ax.imshow(255 - np.array(list(map(int, row))).reshape(28, 28).astype(np.uint8), cmap='gray')
    plt.title('result: ' + str(result))
    plt.axis('off')
    show_image += 1
    if show_image == 10:    # every ten pictures a page
        plt.show()
        plt.draw()
        show_image = 0
    
plt.show()
plt.draw()


# Submit the result in .csv

# The format we'll submit, and these are titles of the two columns
submit = 'ImageId,Label\n'

# Open the testing file
with open('test.csv', 'r')as file:
    csv_lines = file.readlines()

# Count for row
image_id = 1

for i in range(1, len(csv_lines)):
    
    # delete '\n', and split by ',' into lists
    row = csv_lines[i].replace('\n', '').split(',')

    # do predict (same as above)
    result = model.predict_classes(np.array([list(map(int, row))])/255.0)[0]
    submit += str(image_id) + ',' + str(result) + '\n'
    image_id += 1

# Write file
open('submit.csv', 'w').write(submit)