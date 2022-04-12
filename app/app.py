import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import math 
from mpl_toolkits.axes_grid1 import ImageGrid

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import matplotlib.pyplot as plt


DATA_DIR = os.path.join(os.path.dirname( os.getcwd()), 'data')
TRIPLET_NAME = 'triplet.csv'
IMAGE_EMBEDDING_NAME = 'image_embedding.pkl'
TEXT_EMBEDDING_NAME = 'text_embedding.pkl'
PATH_TO_TRIPLET_FILE =  os.path.join(DATA_DIR, TRIPLET_NAME)
PATH_TO_IMAGE_EMBEDDING =  os.path.join(DATA_DIR, IMAGE_EMBEDDING_NAME)
PATH_TO_TEXT_EMBEDDING =  os.path.join(DATA_DIR, TEXT_EMBEDDING_NAME)
ASIN_LIST = ['B06XTFJ5CL', 'B078GV4F8H', 'B085GXH8YR']
FEATURE_LIST = ['Image', 'Text', 'Combined']


@st.cache(suppress_st_warning=True)
def get_raw_data():
    raw_df = pd.read_csv(PATH_TO_TRIPLET_FILE)
    return raw_df

def combined_pred(image, text, image_weight=0.5):
    res = image_weight*image + (1-image_weight)*text
    return res

def get_rec(data, similarity_score, item_id, k=50):
    sim_index = data[data['item_id'] == item_id].index[0]
    res = list(enumerate(similarity_score[sim_index]))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = res[1:k+1]
    rec_idx = [i[0] for i in res]
    rec_scores = [i[1] for i in res]
    return rec_idx, rec_scores

def combined_rec(data, image_similarity_scores, comb_sim_scores, item_id, k=100):
    sim_index = data[data['item_id'] == item_id].index[0]
    text_val = data.iloc[sim_index]['bullet_point']
    mask = False
    if isinstance(text_val, float):
        mask = math.isnan(text_val)
    if mask:
        res = list(enumerate(image_similarity_scores[sim_index]))
    else:
        res = list(enumerate(comb_sim_scores[sim_index]))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = res[1:k+1]
    rec_idx = [i[0] for i in res]
    rec_scores = [i[1] for i in res]
    return rec_idx, rec_scores

@st.cache(suppress_st_warning=True)
def load_embedding():
    raw_df = get_raw_data()
    with open(PATH_TO_IMAGE_EMBEDDING, "rb") as input_file:
        df_embs = pickle.load(input_file)
    with open(PATH_TO_TEXT_EMBEDDING, "rb") as input_file:
        df_embs_text = pickle.load(input_file)
    image_similarity_scores = np.zeros((raw_df.shape[0],raw_df.shape[0]))
    image_similarity_scores = cosine_similarity(df_embs,df_embs)
    text_similarity_scores = np.zeros((raw_df.shape[0],raw_df.shape[0]))
    text_similarity_scores = cosine_similarity(df_embs_text,df_embs_text)
    return image_similarity_scores, text_similarity_scores, raw_df

def display_images(data_df, rec_list_idx, random=False, rows=10, cols=5):
    img_arr = list(data_df.iloc[rec_list_idx]['image_path'].values)
    iter_temp=0
    fig = plt.figure(figsize=(15,8))
    grid = ImageGrid(fig, 111, 
                nrows_ncols=(10, 5),
                axes_pad=0.2,
                )
    for i, (ax, im) in enumerate(zip(grid, img_arr)):
        img = tf.keras.preprocessing.image.load_img(im, target_size=(224,224))
        ax.imshow(img)
        ax.set_title(f'{data_df.iloc[int(rec_list_idx[i])].product_type}', fontsize=5)
        ax.axis('off')
    fig.tight_layout()
    st.pyplot(fig)

def main():
    st.set_page_config(page_title='Erec', page_icon="🕵️", layout='wide')
    st.title('Product Recommender')
    st.sidebar.title("🕵️ Erec 🕵️".center(20))
    selected_app = st.sidebar.selectbox("Choose ASIN", ASIN_LIST)
    selected_embedding = st.sidebar.selectbox("Choose Feature", FEATURE_LIST)
    with st.sidebar.form(key ='my_form'):
        submit_button = st.form_submit_button(label = 'Generate Recommendation 🔎')
        with st.spinner("Erec at work ⌛⌛"):
            if submit_button:
                image_similarity_scores, text_similarity_scores, raw_df = load_embedding()
                anchor_image_type =  raw_df[raw_df['item_id'] == selected_app]['product_type'].values[0]
                st.success(anchor_image_type)
                anchor_image = raw_df[raw_df['item_id'] == selected_app]['image_path'].values[0]
                st.sidebar.image(tf.keras.preprocessing.image.load_img(anchor_image, target_size=(224,224)), use_column_width=True)
                st.session_state['raw_df'] = raw_df
                st.session_state['image_similarity_scores'] = image_similarity_scores
                st.session_state['text_similarity_scores'] = text_similarity_scores
                st.session_state['selected_app'] = selected_app
                st.session_state['selected_embedding'] = selected_embedding
    try:
        if selected_embedding == 'Combined':
            print('c')
            comb_sim_scores = combined_pred(image_similarity_scores, text_similarity_scores, image_weight=0.5)
            rec_idx, rec_scores_comb = combined_rec(raw_df, image_similarity_scores, comb_sim_scores, selected_app)
            display_images(raw_df, rec_idx, random=True)
        elif selected_embedding == 'Image':
            print('i')
            rec_idx, rec_scores = get_rec(raw_df, image_similarity_scores, selected_app)
            display_images(raw_df, rec_idx, random=True)
        else:
            print('t')
            rec_idx, rec_scores = get_rec(raw_df, text_similarity_scores, selected_app)
            display_images(raw_df, rec_idx, random=True)
    except AttributeError as e:
        st.error(f'Please submit your selection to generate recommendations')
    except UnboundLocalError as e:
        st.error(f'PLease use session state to cache variable')
    

if __name__=='__main__':
    main()

    
    