import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
import os

def loadCompositePics():
    size = 50000
    f = h5py.File("Dataset.hdf5")
    x = f['train-img'][:size]
    y = f['train-label'][:size]
    z = f['test-img'][:size]
    a = f['test-label'][:size]

    x *= 1 / 255.
    z *= 1 / 255.

    X2d = np.reshape(x, (size, 1, 84, 28))
    Z2d = np.reshape(z, (size, 1, 84,28))

    y = np.array(y)
    a = np.array(y)
    perm = np.random.permutation(range(len(x)))
    perm2 = np.random.permutation(range(len(z)))
    trainpics = X2d[perm]
    trainlabels = y[perm]
    testpic = Z2d[perm2]
    testlabels = a[perm2]
    # X_train, X_test = X[:len(X) / 2], X[len(X) / 2:]
    # y_train, y_test = Y[:len(y) / 2], Y[len(y) / 2:]

    # rtest = f['real-img'][:30]
    # rlabel = f['real-label'][:30]
    # print rtest.shape, rlabel.shape
    # rtest *= 1/255.
    # rtest = np.reshape(rtest, (29, 84,28,1))
    # perm2 = np.random.permutation(range(len(rtest)))
    # rtest = rtest[perm2]
    # rlabel = np.array(rlabel)
    # rlabel = rlabel[perm2]
    # return X_train, X_test, y_train, y_test,rtest, rlabel
    print trainpics.shape, trainlabels.shape, testpic.shape, testlabels.shape
    return trainpics,testpic,trainlabels,testlabels

def createConv():
    model = Sequential()

    # First convolutional layer
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1,84,28)))
    model.add(Activation('relu'))

    # Second convolutional layer
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    # model.add(Convolution2D(64, 5, 5))
    # model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(300))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.1))

    model.add(Dense(24))
    model.add(Activation('sigmoid'))

    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mse', optimizer="adam", metrics=['accuracy'])
    model.compile(loss='mse', optimizer="adam")

    return model

def test(model, X_train, X_test, y_train, y_test):
    predicted = model.predict(X_train)
    total = len(predicted)*3.0
    count = 0

    for i in range(len(predicted)):
        if np.argmax(predicted[i][:10]) == np.argmax(y_train[i][:10]):
            count += 1
        if np.argmax(predicted[i][14:]) == np.argmax(y_train[i][14:]):
            count += 1
        if np.argmax(predicted[i][10:14]) == np.argmax(y_train[i][10:14]):
            count +=1
        # print predicted[i][:10], y_test[i][:10]
        # print predicted[i][13:],y_test[i][13:]
    print "Training Set: ", count/total

    predicted = model.predict(X_test)
    total = len(predicted)*3.0
    count = 0

    for i in range(len(predicted)):
        if np.argmax(predicted[i][:10]) == np.argmax(y_test[i][:10]):
            count += 1
        if np.argmax(predicted[i][14:]) == np.argmax(y_test[i][14:]):
            count += 1
        if np.argmax(predicted[i][10:14]) == np.argmax(y_test[i][10:14]):
            count += 1
        # print predicted[i][:10], y_test[i][:10]
        # print predicted[i][13:],y_test[i][13:]
    print "Test Set: ", count/total

def save_model(model):
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir, "QuickStart"))
    model.save('pic_model.h5')

if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = loadDataset1D()
    # model = createDense()

    # X_train, X_test, y_train, y_test, r_test, r_label = loadCompositePics()
    X_train, X_test, y_train, y_test = loadCompositePics()
    print X_train.shape
    model = createConv()

    minibatch_size = 32

    model.fit(X_train, y_train,
              batch_size=minibatch_size,
              nb_epoch=50,
              validation_data=(X_test, y_test),
              verbose=1)

    test(model, X_train, X_test, y_train, y_test)
    save_model(model)