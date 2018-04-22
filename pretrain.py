from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense,  Activation, Flatten, Convolution2D
from dataset import mnist
import keras
import numpy as np
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

epochs = 20
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
pretrained_model = keras.models.load_model("./my_pretrained_net")


from empirical_priors import GaussianMixturePrior
from extended_keras import extract_weights

pi_zero = 0.99

RegularizationLayer = GaussianMixturePrior(nb_components=16,
                                           network_weights=extract_weights(model),
                                           pretrained_weights=pretrained_model.get_weights(),
                                           pi_zero=pi_zero,
                                           name="complexity_loss")(Layers)

model = Model(input = [InputLayer], output = [PredictionLayer, RegularizationLayer])

import optimizers
from extended_keras import identity_objective

tau = 0.003
N = X_train.shape[0]

opt = optimizers.Adam(lr = [5e-4,1e-4,3e-3,3e-3],  #[unnamed, means, log(precition), log(mixing proportions)]
                      param_types_dict = ['means','gammas','rhos'])

model.compile(optimizer = opt,
              loss = {"error_loss": "categorical_crossentropy", "complexity_loss": identity_objective},
              loss_weights = {"error_loss": 1. , "complexity_loss": tau/N},
              metrics = ['accuracy'])

print("Finish the first step\n\n\n\n\n\n")


from extended_keras import VisualisationCallback

epochs = 10
model.fit({"input": X_train,},
          {"error_loss" : Y_train, "complexity_loss": np.zeros((N,10))},
          epochs = epochs,
          batch_size = batch_size,
          verbose = 1., callbacks=[VisualisationCallback(model,X_test,Y_test, epochs)])

display.clear_output()

display.Image(url='./figures/my_retraining.gif')
