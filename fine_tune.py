from keras import optimizers
from keras import models
from data import read_data_generator

model = models.load_model('model.h5')
print('layers number: ', len(model.layers))
for layer in model.layers[:12]:
    layer.trainable = False

train_generator, num_train, valid_generator, num_valid = read_data_generator('./fine-tune-data', header=False)

adam = optimizers.Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)

model.fit_generator(
    train_generator, steps_per_epoch=num_train / 128,
    validation_data=valid_generator, validation_steps=num_valid / 128,
    epochs=15, verbose=True)

model.save('model-tuned.h5')
