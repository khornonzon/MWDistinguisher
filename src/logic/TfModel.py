import tensorflow as tf
from logic.AudioDataset import AudioDataset
from sklearn.model_selection import train_test_split
dataset = AudioDataset()
dataset.load_data("D:\python_projects\MWDistinguisher\datasets\\train\\")
X_train, X_test, y_train, y_test = train_test_split(dataset.train_data, dataset.train_target,test_size=0.1, random_state=22)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(8,)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(2)
])
print(X_train.shape)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)
