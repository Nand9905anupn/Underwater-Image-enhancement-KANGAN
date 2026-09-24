import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

dataset_folder = "Dataset"
IMG_SIZE = 128

raw_dir = os.path.join(dataset_folder, "Raw")
ref_dir = os.path.join(dataset_folder, "Reference")

raw_images = sorted(os.listdir(raw_dir))
ref_images = sorted(os.listdir(ref_dir))

print("Raw images:", len(raw_images))
print("Reference images:", len(ref_images))

pairs = []

for r in raw_images:
    if r in ref_images:
        pairs.append((r, r))

print("Valid pairs:", len(pairs))

def load_image(path):

    img = tf.io.read_file(path)

    img = tf.image.decode_png(img, channels=3)   # safer decoder

    img = tf.image.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        method="bicubic"
    )

    img = tf.cast(img, tf.float32) / 255.0

    return img

raw_data = []
ref_data = []

for raw_name, ref_name in pairs:

    raw_path = os.path.join(raw_dir, raw_name)
    ref_path = os.path.join(ref_dir, ref_name)
    print(raw_path)
    print(ref_path)
    raw_img = load_image(raw_path)
    ref_img = load_image(ref_path)

    raw_data.append(raw_img.numpy())
    ref_data.append(ref_img.numpy())

raw_data = np.array(raw_data, dtype=np.float32)
ref_data = np.array(ref_data, dtype=np.float32)

print("Raw shape:", raw_data.shape)
print("Ref shape:", ref_data.shape)

os.makedirs("features1", exist_ok=True)

np.save("features1/raw.npy", raw_data)
np.save("features1/reference.npy", ref_data)

print("Dataset saved")

#%%
raw = np.load("features1/raw.npy")
ref = np.load("features1/reference.npy")

indices = np.random.permutation(len(raw))

train_size = int(0.8 * len(raw))

train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train = raw[train_idx]
y_train = ref[train_idx]

X_test = raw[test_idx]
y_test = ref[test_idx]

np.save("features1/X_train.npy", X_train)
np.save("features1/y_train.npy", y_train)

np.save("features1/X_test.npy", X_test)
np.save("features1/y_test.npy", y_test)

print("Dataset split saved") 

#%%
import os
import numpy as np
import tensorflow as tf

# =========================
# CONFIG
# =========================
dataset_folder = "Dataset"
IMG_SIZE = 128

raw_dir = os.path.join(dataset_folder, "GT")
ref_dir = os.path.join(dataset_folder, "input")

# =========================
# STEP 1: LOAD FILES
# =========================
raw_images = sorted(os.listdir(raw_dir))
ref_images = sorted(os.listdir(ref_dir))

print("Raw images:", len(raw_images))
print("Reference images:", len(ref_images))

# =========================
# STEP 2: CREATE DICTIONARY FOR SAFE PAIRING
# =========================
ref_dict = {}

for r in ref_images:
    ref_dict[r] = r  # direct mapping (same filename assumed)

# If filenames match exactly, this works safely

pairs = []

for r in raw_images:
    if r in ref_dict:
        pairs.append((r, ref_dict[r]))

print("Valid pairs:", len(pairs))

# =========================
# STEP 3: IMAGE LOADER
# =========================
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img

# =========================
# STEP 4: PROCESS DATA
# =========================
raw_data = []
ref_data = []
pair_index = []

for i, (raw_name, ref_name) in enumerate(pairs):

    raw_path = os.path.join(raw_dir, raw_name)
    ref_path = os.path.join(ref_dir, ref_name)
    print(raw_path)
    print(ref_path)
    raw_img = load_image(raw_path)
    ref_img = load_image(ref_path)

    raw_data.append(raw_img.numpy())
    ref_data.append(ref_img.numpy())

    pair_index.append([i, i])

# =========================
# STEP 5: CONVERT TO NUMPY
# =========================
raw_data = np.array(raw_data, dtype=np.float32)
ref_data = np.array(ref_data, dtype=np.float32)
pair_index = np.array(pair_index, dtype=np.int32)

print("Final raw shape:", raw_data.shape)
print("Final reference shape:", ref_data.shape)
print("Pairs shape:", pair_index.shape)

# =========================
# STEP 6: SAVE FILES
# =========================
np.save("features/raw.npy", raw_data)
np.save("features/reference.npy", ref_data)
np.save("features/pairs.npy", pair_index)

#%%
import numpy as np
import os

# =========================
# LOAD DATA
# =========================
raw = np.load("features/raw.npy")
ref = np.load("features/reference.npy")

print("Total samples:", raw.shape[0])

# =========================
# CREATE SHUFFLED INDICES
# =========================
indices = np.random.permutation(len(raw))

# =========================
# SPLIT (80/20)
# =========================
train_size = int(0.8 * len(raw))

train_idx = indices[:train_size]
test_idx = indices[train_size:]

# =========================
# APPLY SPLIT (IMPORTANT: KEEP PAIRING)
# =========================
X_train = raw[train_idx]
y_train = ref[train_idx]

X_test = raw[test_idx]
y_test = ref[test_idx]




'''np.save("features/X_train.npy", X_train)
np.save("features/y_train.npy", y_train)

np.save("features/X_test.npy", X_test)
np.save("features/y_test.npy", y_test)'''


"""=================================================================================
                  Image Enhancement Gan-Kan  model 
====================================================================================="""

from gan_model import KANGAN 
batch_size = 8
epochs = 300

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=500).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

model = KANGAN()

# Optimizers
g_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

# Loss functions
bce_loss = keras.losses.BinaryCrossentropy(from_logits=False)
l1_loss = keras.losses.MeanAbsoluteError()

# Compile model
model.compile(
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
    bce_loss=bce_loss,
    l1_loss=l1_loss
)

# Create callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_kan_gan.h5', 
        save_best_only=True, 
        monitor='g_loss', 
        mode='min', 
        save_weights_only=False,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='g_loss', 
        patience=10, 
        factor=0.5, 
        verbose=1,
        min_lr=1e-7
    ),
    keras.callbacks.EarlyStopping(
        monitor='g_loss', 
        patience=30, 
        restore_best_weights=True, 
        verbose=1
    )
]

print("\n5. Starting training...")
print("="*60)

# Train model
history = model.fit(
    train_dataset,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

model.generator.save('kan_gan_generator_final.h5')
model.discriminator.save('kan_gan_discriminator_final.h5')
#%%
import numpy as np
from Existing import *

# -------------------------
# Create models folder
# -------------------------
os.makedirs("models", exist_ok=True)


# -------------------------
# Load Dataset
# -------------------------
X_train = np.load("features1/X_train.npy")
y_train = np.load("features1/y_train.npy")

X_test = np.load("features1/X_test.npy")
y_test = np.load("features1/y_test.npy")

print("Dataset Loaded")
print("X_train shape:", X_train.shape)


# -------------------------
# AHE Preprocessing
# -------------------------
print("Applying AHE...")
X_train_ahe = apply_ahe(X_train)
X_test_ahe = apply_ahe(X_test)


# -------------------------
# CLAHE Preprocessing
# -------------------------
print("Applying CLAHE...")
X_train_clahe = apply_clahe(X_train)
X_test_clahe = apply_clahe(X_test)


# -------------------------
# Train UGAN
# -------------------------
print("Training UGAN...")

input_shape = X_train.shape[1:]

ugan = build_ugan(input_shape)

ugan.fit(
    X_train,
    y_train,
    epochs=1,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Save UGAN model
ugan.save("models/ugan_model.h5")

print("UGAN model saved.")


# -------------------------
# Train MSGNet
# -------------------------
print("Training MSGNet...")

msgnet = build_msgnet(input_shape)

msgnet.fit(
    X_train,
    y_train,
    epochs=1,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Save MSGNet model
msgnet.save("models/msgnet_model.h5")

 


