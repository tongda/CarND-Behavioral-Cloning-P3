from data import read_data_generator
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Cropping2D
from keras.callbacks import TensorBoard, ProgbarLogger
from keras import activations

data_name = 'data-good'
dropout_rate = 0.2
num_epoch = 4

train_generator, num_train, valid_generator, num_valid = read_data_generator('./' + data_name, header=False)

activation = activations.elu

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(((44, 20), (0, 0))))
model.add(Conv2D(16, (1, 1), padding='same', activation=activation, kernel_initializer='glorot_normal'))
model.add(Conv2D(64, (5, 5), padding='same', activation=activation, kernel_initializer='glorot_normal'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (5, 5), padding='same', activation=activation, kernel_initializer='glorot_normal'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (5, 5), padding='same', activation=activation, kernel_initializer='glorot_normal'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), padding='same', activation=activation, kernel_initializer='glorot_normal'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), padding='same', activation=activation, kernel_initializer='glorot_normal'))
model.add(MaxPooling2D())
model.add(Dropout(dropout_rate))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(dropout_rate))
model.add(Dense(64))
model.add(Dropout(dropout_rate))
model.add(Dense(16))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

logname = '-'.join([data_name, 'dr' + str(dropout_rate), 'ep' + str(num_epoch), activations.serialize(activation)])

tfboard = TensorBoard('./logs/' + logname, histogram_freq=1, write_graph=True)
model.fit_generator(
    train_generator, steps_per_epoch=num_train / 128,
    validation_data=valid_generator, validation_steps=num_valid / 128,
    epochs=num_epoch, verbose=True,
    max_q_size=50,
    callbacks=[tfboard])

model.save('model.h5')