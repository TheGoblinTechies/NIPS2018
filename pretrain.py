from keras import backend as K

from keras.models import Model
from keras.layers import Input, Dense,  Activation, Flatten, Convolution2D
from dataset import mnist

[X_train, X_test], [Y_train, Y_test], [img_rows, img_cols], nb_classes = mnist()
# We configure the input here to match the backend. If properly done this is a lot faster. 
if K._BACKEND == "theano":
    InputLayer = Input(shape=(1, img_rows, img_cols), name="input")
elif K._BACKEND == "tensorflow":
    InputLayer = Input(shape=(img_rows, img_cols,1), name="input")

# A classical architecture ...
#   ... with 3 convolutional layers,
Layers = Convolution2D(25, 5, 5, subsample = (2,2), activation = "relu")(InputLayer)
Layers = Convolution2D(50, 3, 3, subsample = (2,2), activation = "relu")(Layers)
#   ... and 2 fully connected layers.
Layers = Flatten()(Layers)
Layers = Dense(500)(Layers)
Layers = Activation("relu")(Layers)
Layers = Dense(nb_classes)(Layers)
PredictionLayer = Activation("softmax", name ="error_loss")(Layers)

# Fianlly, we create a model object:
model = Model(input=[InputLayer], output=[PredictionLayer])

model.summary()

from keras import optimizers

epochs = 100
batch_size = 256
opt = optimizers.Adam(lr=0.001)

model.compile(optimizer= opt,
              loss = {"error_loss": "categorical_crossentropy",},
               metrics=["accuracy"])

model.fit({"input": X_train, }, {"error_loss": Y_train},
          nb_epoch = epochs, batch_size = batch_size,
          verbose = 0, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:', score[1])

keras.models.save_model(model, "./my_pretrained_net")