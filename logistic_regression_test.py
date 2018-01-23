#-*- coding:utf-8 -*-
import numpy as np
import pickle
import sys
from keras import backend as K
from collections import Counter
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Dropout, Activation, Reshape
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split

def get_class_weights(y, smooth_factor=0.01):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}


stock=sys.argv[1]
mmm=sys.argv[2]

print ('running logistic_{}{}'.format(stock, mmm))

print ("Loding data")
(x, y, price)=pickle.load(open('{}{}.p'.format(stock,mmm),'rb'))

class_weight_ = get_class_weights(y)

data_dim=len(x[0][0])

lb = LabelBinarizer()
y = list(lb.fit_transform(y))
#print (lb.classes_)

class_weight={}
for key in class_weight_:
    class_weight[list(lb.classes_).index(key)]=class_weight_[key]
print (class_weight)

x=list(np.array(x).reshape((len(x), 10*data_dim)))

x_len=len(x)
y_len=len(y)


#X_train, X_test, Y_train, Y_test = train_test_split(np.array(x), np.array(y), test_size=0.2, random_state=100)
X_train, X_test, Y_train, Y_test = np.array(x[:70000]+x[80000:]), np.array(x[70000:80000]), np.array(y[:70000]+y[80000:]), np.array(y[70000:80000])

#print ("Training on {} data".format(len(X_train)))
#print ("Testing on {} data".format(len(X_test)))

hidden_dim=256
batch_size=64
epoch = 100

print('Build model...')
inputs = Input((10*data_dim,))
x = Dense(128)(inputs)
x = Dense(64)(x)
outputs = Dense(len(lb.classes_), activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.summary()
#checkpointer=ModelCheckpoint('model_logisticregression_{}{}'.format(stock,mmm), monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)
#earlystopping=EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
#model.fit(X_train, Y_train,
#          batch_size=batch_size,
#          epochs=epoch,
#          class_weight=class_weight,
#          validation_data=(X_test, Y_test),
#          callbacks=([checkpointer, earlystopping]))
model = load_model('model_logisticregression_{}{}'.format(stock,mmm))
#prediction = model.predict(X_test)
#predict_value=list(map(float,lb.inverse_transform(prediction)))
#print (predict_value[:10])
#
#f = open('{}_regression{}_prediction.p'.format(stock,mmm),'wb')
#pickle.dump(predict_value, f)
#print ('{}_regression{}_prediction.p saved'.format(stock,mmm))
test= model.evaluate(X_test,Y_test)
print(test)
