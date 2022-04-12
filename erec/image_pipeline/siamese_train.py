import os 
import random
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from functools import partial
from scipy.spatial.distance import cdist
import keras_toolkit as kt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import seaborn as sns
import pickle

import tensorflow as tf
from tensorflow.keras import Model, metrics, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, MaxPooling2D, Flatten, Layer, BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import unit_norm
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Convolution2D, MaxPooling2D, Flatten, Layer
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
tf.keras.backend.clear_session()
# physical_devices = tf.config.list_physical_devices('GPU') 
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)
params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black"}
plt.style.use('ggplot')


## Define Constants

DEFAULT_RANDOM_SEED = 2021
DATA_DIR = os.path.join(os.path.dirname( os.getcwd()), 'data')
TRIPLET_NAME = 'triplet.csv'
FINAL_DF = 'final_df.csv'
EMBEDDING_NAME = 'image_embedding.pkl'
PATH_TO_TRIPLET_FILE =  os.path.join(DATA_DIR, TRIPLET_NAME)
FINAL_DF_FILE =  os.path.join(DATA_DIR, FINAL_DF)
PATH_TO_IMAGE_EMBEDDING =  os.path.join(DATA_DIR, EMBEDDING_NAME)


TARGET_SHAPE = (224, 224)
SPLIT_RATIO = 0.5  # 0.8
BATCH_SIZE = 16
BATCH_SIZE_INFERENCE = 128
EPOCH_WARM = 2  # 2
EPOCH_FULL_TRAIN = 5  # 10
TRAINING = 'test'  # use 'complete' to train on all the data


def seed_everything(seed=DEFAULT_RANDOM_SEED):
    '''
    Funtion to set seeds for different libraries for reproducibility.
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def decode(path: str, target_shape=TARGET_SHAPE):
    file_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(file_bytes, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, target_shape)
    return img


def build_triplets_dset(df, bsize=32, cache=True, shuffle=False):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    build_dataset = partial(
        kt.image.build_dataset,
        decode_fn=decode,
        bsize=bsize,
        augment=True,
        cache=cache,
        shuffle=False
    )

    danchor = build_dataset(df.anchor, augment=False)
    dpositive = build_dataset(df.positive, augment=True)
    dnegative = build_dataset(df.negative, augment=True)
    
    dset = tf.data.Dataset.zip((danchor, dpositive, dnegative))
    if shuffle:
        dset = dset.shuffle(shuffle)
    
    return dset

class DistanceLayer(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class SiameseModel(Model):
    """The Siamese Network model.

    Computes the triplet loss using the triplet embeddings produced by the
    Siamese Network.
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        anchor_positive_distance, anchor_negative_distance = self.siamese_network(data)

        loss = anchor_positive_distance - anchor_negative_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

def build_model():
    base_cnn = tf.keras.applications.EfficientNetB0(
            weights = 'imagenet',
            include_top = False, 
            pooling = 'avg', 
            input_shape=TARGET_SHAPE + (3,)
        )
    trainable = False
    for layer in base_cnn.layers:
        layer.trainable = trainable

    x = base_cnn.output
    x = BatchNormalization()(x)
    x = Dropout(0.5, name='dropout')(x)
    x = Dense(256, activation='linear', name='custom_linear', kernel_initializer='he_normal')(x)
    x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
    embedding_model = Model(base_cnn.input, x, name="embedding")


    anchor_input = Input(name="anchor", shape=TARGET_SHAPE + (3,))
    positive_input = Input(name="positive", shape=TARGET_SHAPE + (3,))
    negative_input = Input(name="negative", shape=TARGET_SHAPE + (3,))

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)
    distances = DistanceLayer()(anchor_embedding,positive_embedding,negative_embedding)
    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    trainable_count = count_params(embedding_model.trainable_weights)
    non_trainable_count = count_params(embedding_model.non_trainable_weights)
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    return base_cnn, embedding_model, siamese_network

def _unfreeze_layers(embed_model, base_model, siamese_model):
    trainable = False
    for layer in embed_model.layers:
        if layer.name in ["custom_linear"]:
            layer.trainable = True
        else:
            layer.trainable = trainable
    for layer in base_model.layers[-17:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    trainable_count = count_params(siamese_model.trainable_weights)
    non_trainable_count = count_params(siamese_model.non_trainable_weights)
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    return embed_model, base_model, siamese_model


def read_data(data_path):
    raw_df = pd.read_csv(data_path)
    cols = ['image_path', 'pos_triplet', 'neg_triplet', 'product_type']
    triplets = raw_df[cols]
    triplets.columns = ['anchor', 'positive', 'negative', 'product_type']
    if TRAINING != 'complete':
        new_df = triplets.groupby("product_type").sample(n=50, random_state=1, replace=True)
        triplets = new_df
    return raw_df, triplets
    
def build_data_generator(data):
    train_triplets, val_triplets = \
                             train_test_split(data, train_size=SPLIT_RATIO, 
                                              random_state=DEFAULT_RANDOM_SEED, 
                                              stratify=data['product_type'])
    dtrain = build_triplets_dset(
    train_triplets,
    bsize=BATCH_SIZE,
    cache=True,
    shuffle=DEFAULT_RANDOM_SEED
    )

    dvalid = build_triplets_dset(
        val_triplets,
        bsize=BATCH_SIZE,
        cache=True,
        shuffle=False
    )
    return dtrain, dvalid

def scheduler(epoch, lr):
        if epoch < 5:
            return lr
        else:
            return lr * tf.math.exp(-0.1)


def generate_embedding(embed_model, df):
    data_datagen = ImageDataGenerator()

    validation_generator = data_datagen.flow_from_dataframe(
            df,
            x_col='image_path',
            y_col='product_type',
            directory=None,
            target_size=TARGET_SHAPE,
            batch_size=BATCH_SIZE_INFERENCE,
            class_mode='categorical',
            shuffle = False
        )
    df_embs = embed_model.predict(validation_generator, verbose=1)
    return df_embs


def main():
    seed_everything()
    raw_df, triplets = read_data(PATH_TO_TRIPLET_FILE)
    dtrain, dvalid = build_data_generator(data=triplets)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    lr_schedular = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callbacks_list = [early, lr_schedular]
    base_cnn, embedding_model, siamese_network = build_model()
    siamese_network_model = SiameseModel(siamese_network)
    siamese_network_model.compile(optimizer=optimizers.SGD(learning_rate=0.00001, momentum=0.9))
    siamese_network_model.fit(dtrain, epochs=EPOCH_WARM, validation_data=dvalid, callbacks=callbacks_list)
    embedding_model, _,siamese_network_model = _unfreeze_layers(embedding_model, base_cnn, siamese_network_model)
    siamese_network_model.compile(optimizer=optimizers.SGD(learning_rate=0.00001, momentum=0.6))
    siamese_network_model.fit(dtrain, epochs=EPOCH_FULL_TRAIN, validation_data=dvalid, callbacks=callbacks_list)
    feature_embedding = generate_embedding(embedding_model, raw_df)
    return raw_df, feature_embedding


if __name__=='__main__':
    raw_df, df_embs = main()
    with open(PATH_TO_IMAGE_EMBEDDING,'wb') as input_image_embedding:
        pickle.dump(df_embs, input_image_embedding)
    raw_df.to_csv(FINAL_DF_FILE, index=False)