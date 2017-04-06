import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
import os

def loadCompositePics():
    train_size = 900
    test_size = 100
    f = h5py.File(os.path.join(os.path.dirname(__file__), "Dataset.hdf5"), 'r')
    train_img = f['train-img'][:]
    train_label = f['train-label'][:]
    test_img = f['test-img'][:]
    test_label = f['test-label'][:]
    # print type(train_img)

    # print x.shape, y.shape
    # x = f['x_train'][:size]
    #
    train_img =train_img*(1 / 255.)
    test_img = test_img* (1 / 255.)
    # print X.shape
    train_2d = np.reshape(train_img, (len(train_img), 84, 28, 1))
    test_2d = np.reshape(test_img, (len(test_img), 84,28,1))

    train_label = np.array(train_label)
    test_label = np.array(test_label)

    perm = np.random.permutation(len(train_img))
    perm2 = np.random.permutation(len(test_img))

    trainpics = train_2d[perm][:train_size]
    trainlabels = train_label[perm][:train_size]
    testpic = test_2d[perm2][:test_size]
    testlabels = test_label[perm2][:test_size]

    print trainpics.shape, trainlabels.shape, testpic.shape, testlabels.shape
    return trainpics,testpic,trainlabels,testlabels

def createConv():
    model = Sequential()


    # First convolutional layer
    #change convolution
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(84,28,1)))
    model.add(Activation('relu'))
    #
    # Second convolutional layer
    #change kernel
    model.add(Convolution2D(64, 5, 5))
    model.add(Activation('relu'))

    # model.add(Convolution2D(64, 5, 5))
    # model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(350))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.3))

    model.add(Dense(24))
    model.add(Activation('sigmoid'))

    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mse', optimizer="adam", metrics=['accuracy'])
    model.compile(loss='mse', optimizer="adam")

    return model

def test(model, train_img, test_img, train_label, test_label):
    # print train_img.shape, train_label.shape, test_img.shape, test_label.shape
    predicted = model.predict(train_img)
    total = len(predicted)*3.0
    count = 0

    for i in range(len(predicted)):
        if np.argmax(predicted[i][:10]) == np.argmax(train_label[i][:10]):
            count += 1
        if np.argmax(predicted[i][14:]) == np.argmax(train_label[i][14:]):
            count += 1
        if np.argmax(predicted[i][10:14]) == np.argmax(train_label[i][10:14]):
            count +=1
        # print predicted[i][:10], y_test[i][:10]
        # print predicted[i][13:],y_test[i][13:]
    print "Training Set: ", count/total

    predicted = model.predict(test_img)
    total = len(predicted)*3.0
    count = 0

    for i in range(len(predicted)):
        if np.argmax(predicted[i][:10]) == np.argmax(test_label[i][:10]):
            count += 1
        if np.argmax(predicted[i][14:]) == np.argmax(test_label[i][14:]):
            count += 1
        if np.argmax(predicted[i][10:14]) == np.argmax(test_label[i][10:14]):
            count += 1
        # print predicted[i][:10], y_test[i][:10]
        # print predicted[i][13:],y_test[i][13:]
    print "Test Set: ", count/total

def save_model(model):
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir, "QuickStart"))
    model.save('pic_model.h5')

# def load_model():
#     m = load_model('pic_model.h5')
#     return m


if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = loadDataset1D()
    # model = createDense()

    # X_train, X_test, y_train, y_test, r_test, r_label = loadCompositePics()
    train_img, test_img, train_label, test_label = loadCompositePics()
    print train_img.shape
    model = createConv()

    minibatch_size = 32
    for i in range(140):
        model.fit(train_img, train_label,
                  batch_size=minibatch_size,
                  nb_epoch=1,
                  validation_data=(test_img, test_label),
                  verbose=1)

        test(model, train_img, test_img, train_label, test_label)
    save_model(model)
    # predicted = model.predict(r_test)
    # total = len(predicted) * 3.0
    # count = 0
    #
    # for i in range(len(predicted)):
    #     if np.argmax(predicted[i][:10]) == np.argmax(r_label[i][:10]):
    #         count += 1
    #     if np.argmax(predicted[i][14:]) == np.argmax(r_label[i][14:]):
    #         count += 1
    #     if np.argmax(predicted[i][10:14]) == np.argmax(r_label[i][10:14]):
    #         count += 1
    # print "Real Set: ", count / total