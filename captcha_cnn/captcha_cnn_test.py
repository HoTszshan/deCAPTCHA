from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
from keras.models import *
import random
import string
from PIL import Image
from tqdm import tqdm



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
    # yield X,y


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])


def test_1():
    model = load_model('decaptcha_cnn.h5')
    (val_x, val_y) = gen(128)
    score = model.evaluate(val_x, val_y,batch_size=32,verbose=1,sample_weight=None)

    print(score[-4])
    print(score[-3])
    print(score[-2])
    print(score[-1])
    print(score[-4] * score[-3] * score[-2] * score[-1])

    batch_acc =0

    y_pred = model.predict(val_x)
    y_pred = np.argmax(y_pred, axis=2).T
    y_true = np.argmax(val_y, axis=2).T
    # batch_acc = sum([np.all(a, b) for a, b in zip(y_pred, y_true)]) / float(len(val_y))
    batch_acc = np.mean(map(np.array_equal, y_true, y_pred))
    print(batch_acc)


    X, y = gen(1)
    y_pred = model.predict(X)
    plt.title('real:%s\npredict:%s'%(decode(y),decode(y_pred)))
    plt.imshow(X[0])
    plt.savefig('test_1.png')
    # plt.show()
    # pil_image = Image.fromarray(np.uint8(X[0]))
    # pil_image.save(decode(y_pred)+'.png')


def test_2(label_str):
    model = load_model('decaptcha_cnn.h5')
    generator = ImageCaptcha(width=width, height=height)
    X = generator.generate_image(label_str)
    X = np.expand_dims(X, 0)
    y_pred = model.predict(X)


    plt.title('real: %s\npred:%s'%(label_str, decode(y_pred)))
    plt.imshow(X[0], cmap='gray')
    plt.savefig('test_2.png')
    plt.show()

def evaluate(batch_num=20):
    batch_acc =0
    generator = gen(128)
    model = load_model('decaptcha_cnn.h5')
    for i in tqdm(range(batch_num)):
        X, y = gen(128) #next(generator)
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=2).T
        y_true = np.argmax(y, axis=2).T
        batch_acc += np.mean(map(np.array_equal, y_true, y_pred))
    return batch_acc / batch_num

# test_1()
test_2('0OO0')
# print evaluate()