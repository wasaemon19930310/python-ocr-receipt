import keras
from keras.datasets import mnist
from matplotlib import pyplot

X_train = None
y_train = None
X_test = None
y_test = None

# データを読み込む
def load_data():
    global X_train, y_train, X_test, y_test
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

# 一次元配列に変換する
def convert_to_array():
    global X_train, X_test
    X_train = X_train.reshape(-1, 784).astype('float32') / 255
    X_test = X_test.reshape(-1, 784).astype('float32') / 255

# ラベルデータをOne-hotベクトルに直す
def convert_to_one_hot():
    global y_train, y_test
    y_train = keras.utils.np_utils.to_categorical(y_train.astype('int32'), 10)
    y_test = keras.utils.np_utils.to_categorical(y_test.astype('int32'), 10)

# モデルを作成する
def create_model():
    global X_train, y_train, X_test, y_test
    # 入力を指定する
    in_size = 28 * 28
    # 出力を指定する
    out_size = 10
    # モデル構造を定義する
    Dense = keras.layers.Dense
    model = keras.models.Sequential()
    model.add(Dense(512, activation='relu', input_shape=(in_size,)))
    model.add(Dense(out_size, activation='softmax'))
    # モデルを構築する
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    # 学習を実行する
    model.fit(X_train, y_train, batch_size=20, epochs=20)
    # モデルを評価する
    score = model.evaluate(X_test, y_test, verbose=1)
    # モデルを保存する
    model.save('receipt.h5')
    print('accuracy: ', score[1], 'loss: ', score[0])

# データを読み込む
load_data()
# 一次元配列に変換する
convert_to_array()
# ラベルデータをOne-hotベクトルに直す
convert_to_one_hot()
# モデルを作成する
create_model()