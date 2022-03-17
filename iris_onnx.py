from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_lightgbm
from onnxmltools.utils import save_model
import numpy as np


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LGBMClassifier()
model.fit(X_train, y_train)

print (X_test[:10])
initial_types = [['inputs', FloatTensorType([10, 4])]]
onnx_model = convert_lightgbm(model, initial_types=initial_types, zipmap=False)
save_model(onnx_model, "iris.onnx")

print(model.predict(X_test[:10]))
# print(onnx_model.predict(X_test))


import onnxruntime

session = onnxruntime.InferenceSession("iris.onnx")
label = session.run(["label"], {"inputs": X_test[:10].astype("float32").reshape(10, 4)})
print(label)


inputs = np.array([
      6.1, 2.8, 4.7, 1.2, 5.7, 3.8, 1.7, 0.3, 7.7, 2.6, 6.9, 2.3, 6.,  2.9,
      4.5, 1.5, 6.8, 2.8, 4.8, 1.4, 5.4, 3.4, 1.5, 0.4, 5.6, 2.9, 3.6, 1.3,
      6.9, 3.1, 5.1, 2.3, 6.2, 2.2, 4.5, 1.5, 5.8, 2.7, 3.9, 1.2,
])

session = onnxruntime.InferenceSession("iris.onnx")
label = session.run(["label"], {"inputs": inputs.astype("float32").reshape(10, 4)})
print(label)
