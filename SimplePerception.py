import tensorflow
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np

def main():
    # 説明変数 (学習データ/入力データ)
    x_train = np.array ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # 目的変数 (正解データ)
    y_train = np.array ([[0.0], [1.0], [1.0], [1.0]])

    # モデル構築
    model = Sequential ()

    # 出力層 (入力数: input_dim = 2, ユニット数: units = 1) 
    # Denseは全結合を意味する
    model.add (Dense (activation = 'sigmoid', input_dim = 2, units = 1))


    # 単純パーセプトロンをコンパイル (勾配法: RMSprop, 損失関数: mean_squared_error, 評価関数: accuracy)
    model.compile (loss = 'mean_squared_error', optimizer = RMSprop (), metrics = ['accuracy'])

    # 学習 (学習データでフィッティング, バッチサイズ: 4, エポック数: 800)
    history = model.fit (x_train, y_train, batch_size = 2, epochs = 1000)

    # 検証用データの用意
    x_test = x_train
    y_test = y_train
       
    # モデルの検証（性能評価）
    test_loss, test_acc = model.evaluate (x_train, y_train, verbose = 0)
    print ('test_loss:', test_loss)  # 損失関数値(この値を最小化するようにパラメータ[重みやバイアス]の調整が行われる)
    print ('test_acc:', test_acc)    # 精度

    # 検証用データをモデルに入力し, 予測値を計算する
    predict_y = model.predict (x_test)
    print ("y_test:", y_test)        # 正解データ
    print ("predict_y:", predict_y)  # 予測データ

if __name__ == '__main__':
    main()
