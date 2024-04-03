import threading
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

data = pd.read_csv('dataset/dataset.csv')

data['Label'] = data.Label.str.contains("Botnet")
data['Label'] = data['Label'].astype(int)
data = data.drop(columns=['StartTime','Dir', 'sTos', 'dTos'])

X = data
y = data['Label']

df = data
df_label0 = df[df['Label'] == 0]
df_label1 = df[df['Label'] == 1]

df_label0 = df_label0.iloc[:250000]

df = pd.concat([df_label0, df_label1])

Y = df['Label']
X = df.drop(columns=['Label'])

le = LabelEncoder()

categorical_features = []
for col, value in X.items():
    if value.dtype == 'object':
        categorical_features.append(col)

for col in categorical_features:
    X[col] = le.fit_transform(X[col])

y = le.fit_transform(Y)

X_train,X_val,Y_train,Y_val = train_test_split(X,y,test_size=0.3,shuffle=True,random_state=1)

sns.countplot(data = X_train, x = Y_train)

plt.show()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

input_size = X_train.shape[1]

num_layers = 2
num_heads = 2
sequence_length = 1
embed_dim = 128

inputs = keras.Input(shape=(None, input_size))
x = layers.Dense(sequence_length * embed_dim, input_shape=(input_size, 1))(inputs)
x = layers.Reshape((sequence_length, embed_dim))(x)
x = layers.Dropout(0.1)(x)
for _ in range(num_layers):
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads,
                                                                  value_dim=embed_dim)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(embed_dim, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

transformers_model = keras.Model(inputs=inputs, outputs=outputs)

transformers_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
transformers_model.summary()
hist_transformers_model = transformers_model.fit(X_train, Y_train, epochs=150, batch_size=256,
                                                                 validation_data=(X_val, Y_val))

y_pred_transformers_model = transformers_model.predict(X_val)
y_pred_transformers_model = y_pred_transformers_model.flatten()
y_pred_transformers_model = y_pred_transformers_model.round(2)
y_pred_transformers_model = np.where(y_pred_transformers_model > 0.5, 1, 0)
print(f"Train acc: {max(hist_transformers_model.history['accuracy'])}")
print(f"Test acc: {accuracy_score(y_pred_transformers_model,Y_val)}")
print(classification_report(Y_val, y_pred_transformers_model))
print(confusion_matrix(Y_val, y_pred_transformers_model).ravel())

acc = hist_transformers_model.history['accuracy']
val_acc = hist_transformers_model.history['val_accuracy']
loss = hist_transformers_model.history['loss']
val_loss = hist_transformers_model.history['val_loss']
x = range(1, len(acc) + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, acc, 'b', label='Training acc')
plt.plot(x, val_acc, 'r', label='Validation acc')
plt.title('Transformers training and validation accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, loss, 'b', label='Training loss')
plt.plot(x, val_loss, 'r', label='Validation loss')
plt.title('Transformers training and validation loss')
plt.legend()