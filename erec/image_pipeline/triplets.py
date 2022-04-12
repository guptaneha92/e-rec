import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
import cv2
import math
import tensorflow as tf
import math
from tqdm import tqdm
import tensorflow.keras
from sklearn.preprocessing import normalize
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from sklearn.neighbors import NearestNeighbors
import random
import logging


DATA_DIR = os.path.join(os.path.dirname( os.getcwd()), 'data')
FILE_NAME = 'final.csv'
DATA_PATH = os.path.join(DATA_DIR, FILE_NAME)
METADATA_PATH = os.path.join(DATA_DIR, 'image_files', 'images.csv')
IMAGE_DIR = os.path.join(DATA_DIR, 'image_files', 'images', 'small')
INPUT_SHAPE = (224,224,3)
INPUT_IMG_SIZE = 224
BATCH_SIZE = 1024 * 4
TRIPLET_NAME = 'triplet.csv'
TRIPLET_PATH =  os.path.join(DATA_DIR, TRIPLET_NAME)

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, img_size, batch_size): 
        self.df = df
        self.img_size = img_size
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.df))
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = len(self.df) // self.batch_size
        ct += int(((len(self.df)) % self.batch_size) != 0)
        return ct

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X = self.__data_generation(indexes)
        return X
            
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        X = np.zeros((len(indexes), self.img_size, self.img_size, 3), dtype = 'float32')
        df = self.df.iloc[indexes]
        for i,(index, row) in enumerate(df.iterrows()):
            img = cv2.imread(row.image_path)
            X[i,] = cv2.resize(img, (self.img_size, self.img_size)) 
        return X

class DataPreProcessing:
    def __init__(self, raw_path: str, meta_path: str, image_dir: str):
        self.raw_path = raw_path
        self.meta_path = meta_path
        self.image_dir = image_dir
        
    def pre_process_data(self):
        'Pre-process data to generate new columns' 
        raw_df = pd.read_csv(self.raw_path)
        metadata = pd.read_csv(self.meta_path)
        metadata['image_path'] = metadata['path'].apply(lambda x: os.path.join(self.image_dir, x))
        raw_df = raw_df[~raw_df['main_image_id'].isna()]
        raw_df['cat_prod_name'] = raw_df['cat_name'] + '_' + raw_df['product_type']
        raw_df['node_name_type'] = raw_df['node_name'] + '_' + raw_df['product_type']
        self.raw_df = raw_df
        self.meta_df = metadata
    
    @staticmethod
    def filter_data(df: pd.DataFrame):
        'Filter data to remove noisy categories' 
        node_filter = list(df['cat_prod_name'].value_counts(ascending=True) [df['cat_prod_name'].value_counts(ascending=True) > 50].index)
        df['use_catprod_flag'] = df['cat_prod_name'].isin(node_filter)
        df = df[df['use_catprod_flag'] == 1].reset_index(drop=True)
        df = df[~(df['cat_name'] == '/categories')].reset_index(drop=True)
        df = df[~(df['cat_name'] == '/Categories')].reset_index(drop=True)
        return df
    
    def generate_df(self):
        self.pre_process_data()
        self.raw_df = self.filter_data(self.raw_df)
        image_path_map = dict(zip(self.meta_df.image_id, self.meta_df.image_path))
        final_df = pd.DataFrame()
        final_df['main_image_id'] = self.raw_df['main_image_id']
        final_df['item_keywords'] = self.raw_df['item_keywords'].fillna('NA')
        final_df['product_type'] = self.raw_df['product_type'].str.lower()
        final_df['item_name'] = self.raw_df['item_name'].fillna('NA')
        final_df['bullet_point'] = self.raw_df['bullet_point'].fillna('NA')
        final_df['node_name'] = self.raw_df['node_name'].str.lower()
        final_df['cat_name'] = self.raw_df['cat_name'].str.lower()
        final_df['item_id'] = self.raw_df['item_id']
        final_df['node_name_type'] = self.raw_df['node_name_type']
        final_df['cat_prod_name']= self.raw_df['cat_prod_name'].astype('category')
        final_df['image_path'] = self.raw_df['main_image_id'].map(image_path_map)
        final_df = final_df[~final_df['image_path'].isna()].reset_index(drop=True)
        return final_df
    
    @staticmethod
    def get_image_embeddings(df: pd.DataFrame, batch_size=32):
        'Generate image embedding' 
        image_embed_model = tf.keras.applications.EfficientNetB0(
            weights = 'imagenet',
            include_top = False, 
            pooling = 'avg',
            input_shape = INPUT_SHAPE)
        n_images = df.shape[0]
        n_features = image_embed_model.layers[-1].output_shape[1]
        image_embeddings = np.zeros((n_images, n_features))
        input_img_size = INPUT_IMG_SIZE
        EPOCHS = math.ceil(n_images / BATCH_SIZE)
        for i in tqdm(range(EPOCHS)):
            a = i * BATCH_SIZE
            b = min((i + 1) * BATCH_SIZE, n_images)
            image_gen = DataGenerator(df = df.iloc[a:b], img_size = input_img_size, batch_size = batch_size)
            batch_embeddings = image_embed_model.predict(image_gen, verbose = 1, use_multiprocessing = True, workers = 4)
            image_embeddings[a:b] = batch_embeddings
        return image_embeddings
    
    @staticmethod
    def knn_predict_embeddings(embeddings ,metric='minkowski', n_neighbors = 300):
        'Predict nearest images for each product' 
        n_images = embeddings.shape[0]
        n_batches = math.ceil(n_images / BATCH_SIZE)
        knn_model = NearestNeighbors(n_neighbors = n_neighbors, metric=metric)
        knn_model.fit(embeddings)
        embed_distances = np.zeros((n_images, n_neighbors))
        embed_indices = np.zeros((n_images, n_neighbors))
        embed_distances, embed_indices = knn_model.kneighbors(embeddings)    
        return embed_distances, embed_indices

class GenerateTriplets:
    def __init__(self, data_df, distances, indices):
        self.data_df = data_df
        self.distances = distances
        self.indices = indices
        
    def get_positive_triplet(self):
        'Method to generate positive triplet' 
        group2df = dict(list(self.data_df.groupby(['node_name_type'])))
        group1df = dict(list(self.data_df.groupby(['cat_prod_name'])))
        for i in tqdm(range(self.distances.shape[0])):
            pred_distances = np.where(self.distances[i]>=0)[0][0:]
            pred_indices = self.indices[i, pred_distances]
            node_name_type = self.data_df.iloc[i]['node_name_type']
            nearest_image_path = self.data_df.iloc[pred_indices]['image_path'].values.tolist()[1]
            self.data_df.loc[i, 'pos_triplet'] = nearest_image_path
    
    def get_negative_triplet(self):
        'Method to generate negative triplet' 
        group1df = dict(list(self.data_df.groupby(['cat_prod_name'])))
        for i in tqdm(range(self.distances.shape[0])):
            dist_arr = np.array((self.distances[i]))
            dist_75perc = max(np.percentile(dist_arr, 75), 0.4)
            dist_50perc = np.percentile(dist_arr, 50)
            pred_distances = np.where(self.distances[i]>dist_75perc)[0][2:]
            pred_distances_50 = np.where(self.distances[i]<dist_50perc)[0]
            pred_indices = self.indices[i, pred_distances]
            current_prod = (self.data_df.iloc[i]['product_type'])
            current_node = (self.data_df.iloc[i]['cat_prod_name'])
            nearest_image_path = self.data_df.iloc[pred_indices]['image_path'].values.tolist()
            all_node_name = self.data_df.iloc[pred_indices]['cat_prod_name'].values.tolist()
            all_node_name_50 = self.data_df.iloc[pred_distances_50]['cat_prod_name'].values.tolist()
            try:
                non_match = (next(x for x in all_node_name if not x == current_node))
            except StopIteration:
                non_match = None
            if non_match:
                nearest_non_match = all_node_name.index(non_match)
                non_match_name = nearest_image_path[nearest_non_match]
                self.data_df.loc[i, 'neg_triplet'] = non_match_name
            else:
                ids = set(group1df.keys())
                remove_cats = list(ids - set(all_node_name_50))
                remove_cats = [x for x in remove_cats if not x.startswith('/Categories')]
                cat_prod_name = random.choice(remove_cats)
                random_item = group1df[cat_prod_name].item_id.tolist()
                negative = random.choice(random_item)
                negative = self.data_df[self.data_df['item_id'] == negative]['image_path'].values[0]
                self.data_df.loc[i, 'neg_triplet'] = negative
    
    def main(self):
        logging.info('Generating Positive Triplets')
        self.get_positive_triplet()
        logging.info('Generating Negative Triplets')
        self.get_negative_triplet()
        self.data_df.to_csv(TRIPLET_PATH, index=False)

if __name__=='__main__':
    pre_process = DataPreProcessing(raw_path=DATA_PATH, meta_path=METADATA_PATH, image_dir=IMAGE_DIR)
    logging.info('Pre-processing data')
    processed_data = pre_process.generate_df()
    logging.info('Generating image embedding')
    image_embed = pre_process.get_image_embeddings(processed_data)
    logging.info('Getting nearest images')
    image_distances, image_indices = pre_process.knn_predict_embeddings(embeddings=image_embed, metric='cosine')
    gen_triplets = GenerateTriplets(processed_data, image_distances, image_indices)
    gen_triplets.main()