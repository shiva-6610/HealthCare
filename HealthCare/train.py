import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pd.read_csv('healthcare_dataset.csv')

input_cols = ['Name', 'Age', 'Gender', 'Blood Type', 'Medical Condition','Medication']
target_col = 'Test Results'

X = data[input_cols]
y = data[target_col]

label_encoders = {}
for col in ['Name', 'Gender', 'Blood Type', 'Medical Condition','Medication']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

scaler = StandardScaler()
X['Age'] = scaler.fit_transform(X[['Age']])

X_np = X.to_numpy().astype(np.float32)
X_np = X_np.reshape((X_np.shape[0], X_np.shape[1], 1))

target_encoder = LabelEncoder()
y_enc = target_encoder.fit_transform(y)
y_cat = to_categorical(y_enc)

X_train, X_test, y_train, y_test = train_test_split(X_np, y_cat, test_size=0.2, random_state=42)

model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_np.shape[1], 1)),
    MaxPooling1D(pool_size=2),  
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1, validation_split=0.1)


model.save("medical_cnn_model.h5")


predictions = model.predict(X_test)
predicted_labels = target_encoder.inverse_transform(np.argmax(predictions, axis=1))

print("Sample predictions:")
for i in range(len(predicted_labels)):
    print(f"Predicted: {predicted_labels[i]} | True: {target_encoder.inverse_transform([np.argmax(y_test[i])])[0]}")
