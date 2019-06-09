from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers
import keras as k

class CNN:

    def __init__(self, input_shape, nb_classes, lr):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.lr = lr

    def setInputShape(self, input_shape):
        self.input_shape = input_shape

    def setNbClasses(self, nb_classes):
        self.nb_classes = nb_classes

    def setLR(self, lr):
        self.lr = lr

    def getInputShape(self):
        print(self.input_shape)

    def getNbClasses(self):
        print(self.nb_classes)

    def getLR(self):
        print(self.lr)

    def swish(self, x):
        return (k.backend.sigmoid(x)*x)

    def ConvNetModel(self):

        # Settings
        pool_size = (2,2)
        prob_drop_conv = 0.25
        prob_drop_hidden = 0.3

        # Architecture CNN Model
        model = Sequential()

        # Convolution Layer 1
        model.add(Conv2D(32,
                         (3,3),
                         padding='same',
                         activation='relu',
                         kernel_initializer="he_normal"))
        # model.add(LeakyReLU(alpha=0.01)) # LeakyReLU Activation

        # Pooling Layer 1
        model.add(MaxPool2D(pool_size=pool_size,
                            strides=(2,2),
                            padding='same'))

        # Convolution Layer 2
        model.add(Conv2D(64,
                         (3, 3),
                         padding='same',
                         activation='relu',
                         kernel_initializer="he_normal"))
        # model.add(LeakyReLU(alpha=0.01))  # LeakyReLU Activation

        # Pooling Layer 2
        model.add(MaxPool2D(pool_size=pool_size,
                            strides=(2, 2),
                            padding='same'))

        # # Convolution Layer 3
        # model.add(Conv2D(128,
        #                  (3, 3),
        #                  padding='same',
        #                  activation='relu',
        #                  kernel_initializer="he_normal"))
        # # model.add(LeakyReLU(alpha=0.01))  # LeakyReLU Activation
        #
        # # Pooling Layer 3
        # model.add(MaxPool2D(pool_size=pool_size,
        #                     strides=(2, 2),
        #                     padding='same'))

        # Fully Connected Layer / Input Layer(Flatten)
        model.add(Flatten())
        model.add(Dropout(prob_drop_conv))

        # Fully Connected Layer / Hidden Layer
        model.add(Dense(1024,
                        activation='relu',
                        kernel_initializer="he_normal",
                        bias_initializer="zeros"))
        # model.add(LeakyReLU(alpha=0.01))  # LeakyReLU Activation
        model.add(Dropout(prob_drop_hidden))

        # Fully Connected Layer / Output Layer (Softmax Activation)
        model.add(Dense(self.nb_classes, activation='softmax'))

        # Setting Optimizers
        # sgd = optimizers.sgd(lr=self.lr,
        #                      momentum=0.9,
        #                      decay=1e-6,
        #                      nesterov=True)

        adam = optimizers.Adam(lr=self.lr,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=None,
                               decay=0.0,
                               amsgrad=False)

        # adamax = optimizers.Adamax(lr=self.lr,
        #                            beta_1=0.9,
        #                            beta_2=0.999,
        #                            epsilon=None,
        #                            decay=0.0)

        # Compile Model
        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model