from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string

characters = string.digits + string.ascii_uppercase
width, height, n_len, n_class = 170, 80, 4, len(characters)

generator = ImageCaptcha(width=width, height=height)
random_str = ''.join([random.choice(characters) for j in range(4)])



def gen(sample_num =10000):
    X = np.zeros((sample_num, height, width, 3), dtype=np.uint8)
    y = [np.zeros((sample_num, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)

    for i in range(sample_num):
        random_str = ''.join([random.choice(characters) for j in range(4)])
        X[i] = generator.generate_image(random_str)
        for j, ch in enumerate(random_str):
            y[j][i, :] = 0
            y[j][i, characters.find(ch)] = 1
    return X,y

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])



import keras
from keras.models import *
from keras.layers import *

input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(4):
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
model = Model(input=input_tensor, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


(train_x, train_y) = gen(51200)
(val_x, val_y) = gen(1280)

np.savez('data.npz',train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
esCallBack = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=0,verbose=0,mode='auto')
model.fit(train_x, train_y, batch_size=32, nb_epoch=25,
          verbose=1, validation_data=(val_x, val_y), callbacks=[tbCallBack, esCallBack])

model.save('decaptcha_cnn.h5')


X, y = gen(1)
y_pred = model.predict(X)
plt.title('real:%s\npred:%s'%(decode(y),decode(y_pred)))
plt.imshow(X[0])
plt.savefig('res.png')


# model.fit_generator(gen(10000), samples_per_epoch=51200, nb_epoch=5,
#                     nb_worker=2, pickle_safe=True,
#                     validation_data=gen(2000), nb_val_samples=1280)