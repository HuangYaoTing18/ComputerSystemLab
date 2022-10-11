TF_LITE_MODEL = './mnist2.tflite'  # 要產生的 TF Lite 檔名
SAVE_KERAS_MODEL = True  # 是否儲存 Keras 原始模型
import autokeras as ak
import tensorflow as tf
from keras.datasets import mnist
# 載入 MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 訓練 AutoKeras 模型
clf = ak.ImageClassifier(max_trials=1, overwrite=True)
clf.fit(x_train, y_train, epochs=25)
# 用測試集評估模型
loss, accuracy = clf.evaluate(x_test, y_test)
print(f'\nPrediction loss: {loss:.3f}, accurcy: {accuracy*100:.3f}%\n')
# 匯出 Keras 模型
model = clf.export_model()
model.summary()
# 儲存 Keras 模型
if SAVE_KERAS_MODEL:
    model.save('./mnist2_model')
# 將模型轉為 TF Lite 格式
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# 你也可以讀取已儲存的 Keras 模型來轉換：
# converter = tf.lite.TFLiteConverter.from_saved_model('./mnist_model')
tflite_model = converter.convert()
# 儲存 TF Lite 模型
with open(TF_LITE_MODEL, 'wb') as f:
    f.write(tflite_model)