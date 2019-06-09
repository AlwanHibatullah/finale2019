import numpy as np
from keras_preprocessing.image import *
from sklearn.model_selection import train_test_split

class LoadData:
    img_rows, img_cols = 64,64
    seed = 20
    np.random.seed(seed)

    data_train = "./data_training"
    data_test = "./data_testing/"
    data_validation = "./data_validation"

    def loadDataTrain(self):
        x = []
        y = []

        datagen = ImageDataGenerator(rescale=1./255)
        train_generator = datagen.flow_from_directory(self.data_train,
                                                      target_size=(self.img_rows, self.img_cols),
                                                      color_mode="grayscale",
                                                      batch_size=1,
                                                      class_mode="categorical",
                                                      shuffle=True,
                                                      seed=self.seed)

        batch_index = 0
        while batch_index <= train_generator.batch_index:
            a,b = train_generator.next()
            x.append(a[0])
            y.append(b[0])
            batch_index = batch_index + 1

        x = np.asarray(x)
        y = np.asarray(y)

        x_train, x_valid, y_train, y_valid = train_test_split(x,y, test_size=0.5, random_state=0)

        return train_generator, x_train, x_valid, y_train, y_valid

    def loadDataTest(self, folder):
        x_test = []

        datagen = ImageDataGenerator(rescale=1./255)
        test_generator = datagen.flow_from_directory(folder,
                                                     target_size=(self.img_rows, self.img_cols),
                                                     color_mode="grayscale",
                                                     batch_size=1,
                                                     class_mode=None,
                                                     shuffle=False,
                                                     seed=self.seed)

        batch_index = 0
        while batch_index <= test_generator.batch_index:
            z = test_generator.next()
            x_test.append(z[0])
            batch_index = batch_index + 1

        x_test = np.asarray(x_test)

        return test_generator, x_test