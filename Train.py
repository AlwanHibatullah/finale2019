from CNN import CNN
from LoadData import LoadData

load_train = LoadData()
data_train = load_train.data_train
train_generator, x_train, x_valid, y_train, y_valid = load_train.loadDataTrain()
input_shape = (load_train.img_rows, load_train.img_cols, 1)

epochs = 100
lr = 0.0002
batch_size = 32

cnn = CNN(input_shape, len(train_generator.class_indices), lr)
model = cnn.ConvNetModel()

# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_valid, y_valid))

model.fit(x_train, y_valid,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid))

evaluation = model.evaluate(x_valid, y_valid, batch_size=batch_size, verbose=1)
print('loss : %.2f, accuracy : %.2f' % (evaluation[0], evaluation[1]))

model.save('./model.h5')