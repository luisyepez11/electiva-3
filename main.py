import os
import sys

# Oculta avisos innecesarios de GPU (Codespaces usa CPU)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import zipfile
import urllib.request
import numpy as np
import matplotlib

# Backend sin pantalla para guardar gráficos en Codespaces
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import keras
from keras import layers
from tensorflow import data as tf_data

# ==========================================
# 1. Descarga y extracción del dataset
# ==========================================
url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
zip_path = "kagglecatsanddogs_5340.zip"

if not os.path.exists("PetImages"):
    if not os.path.exists(zip_path):
        print("Descargando dataset (786 MB)...")
        urllib.request.urlretrieve(url, zip_path)
    print("Descomprimiendo archivos...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

# ==========================================
# 2. Filtrado de imágenes corruptas
# ==========================================
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "rb") as fobj:
                is_jfif = b"JFIF" in fobj.peek(10)
        except Exception:
            is_jfif = False

        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)

print(f"Imágenes corruptas eliminadas: {num_skipped}")

# ==========================================
# 3. Generación de datasets (Train / Val)
# ==========================================
# Reducimos batch_size de 128 a 32 para no agotar la RAM en Codespaces
image_size = (180, 180)
batch_size = 32

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Rutas de las imágenes de validación (el modelo no entrena con ellas)
val_file_paths = list(val_ds.file_paths)

# Guardar gráfico 1: Primeras 9 imágenes
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.suptitle("Muestras originales (0 = Cat, 1 = Dog)")
plt.savefig("1_muestras_originales.png", bbox_inches="tight")
plt.close()
print("Guardado: 1_muestras_originales.png")

# ==========================================
# 4. Data Augmentation y rendimiento
# ==========================================
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


# Guardar gráfico 2: Ejemplos de Data Augmentation
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")
plt.suptitle("Ejemplos de Data Augmentation")
plt.savefig("2_data_augmentation.png", bbox_inches="tight")
plt.close()
print("Guardado: 2_data_augmentation.png")

# prefetch(1) evita que el autotuner sature la RAM disponible
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
train_ds = train_ds.prefetch(1)
val_ds = val_ds.prefetch(1)


# ==========================================
# 5. Arquitectura del modelo (Mini-Xception)
# ==========================================
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    units = 1 if num_classes == 2 else num_classes

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)

# plot_model necesita pydot y graphviz; si faltan, se sigue sin el diagrama
try:
    keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
    print("Guardado: model.png")
except ImportError as e:
    print(f"No se generó model.png (falta pydot/graphviz): {e}")

# ==========================================
# 6. Compilación y entrenamiento
# ==========================================
# Nota: En CPU las 25 épocas pueden tardar bastante; si quieres probar rápido, cámbialo a 3 o 5
epochs = 25
# Guarda solo el modelo con mejor precisión en validación (no uno por época)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "mejor_modelo.keras",
        monitor="val_acc",
        mode="max",
        save_best_only=True,
    ),
]

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# Evaluación con el mejor modelo guardado sobre imágenes que no vio al entrenar
model = keras.models.load_model("mejor_modelo.keras")
val_loss, val_acc = model.evaluate(val_ds)
print(f"Precisión en validación: {100 * val_acc:.2f}%")

# ==========================================
# 7. Inferencia con una imagen nueva
# ==========================================
# Se puede pasar una imagen propia: python main.py ruta/a/imagen.jpg
# Si no, se usa una imagen del set de validación (no usada en el entrenamiento)
if len(sys.argv) > 1:
    test_img_path = sys.argv[1]
else:
    test_img_path = val_file_paths[0]
print(f"Imagen de prueba: {test_img_path}")

img = keras.utils.load_img(test_img_path, target_size=image_size)

plt.figure()
plt.imshow(img)
plt.axis("off")
plt.savefig("3_prediccion.png", bbox_inches="tight")
plt.close()

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"Esta imagen es {100 * (1 - score):.2f}% gato y {100 * score:.2f}% perro.")