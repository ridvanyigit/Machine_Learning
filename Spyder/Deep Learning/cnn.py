from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Modeli oluşturma
classifier = Sequential()

# 1. Adım - Convolution Katmanı
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# 2. Adım - Pooling Katmanı
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 3. Adım - 2. Convolution Katmanı
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 4. Adım - Flattening
classifier.add(Flatten())

# 5. Adım - Tam Bağlantılı Katman (Dense)
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Modeli Derleme
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Veri Ön İşleme
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim Seti
training_set = train_datagen.flow_from_directory(
    'veriler/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Test Seti
test_set = test_datagen.flow_from_directory(
    'veriler/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Modeli Eğitme
classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=1,
    validation_data=test_set,
    validation_steps=len(test_set)
)

# Tahminler
test_set.reset()
predictions = classifier.predict(test_set, verbose=1)
predictions = np.where(predictions > 0.5, 1, 0)

# Test Etiketlerini Toplama
test_labels = []
for i in range(len(test_set)):
    test_labels.extend(np.array(test_set[i][1]))

# Sonuçları DataFrame'e Aktarma
file_names = test_set.filenames
results = pd.DataFrame({
    'file_names': file_names,
    'predictions': predictions.flatten(),
    'actual': test_labels
})

# Karışıklık Matrisi
conf_matrix = confusion_matrix(test_labels, predictions)
print(conf_matrix)

