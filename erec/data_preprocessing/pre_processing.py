''' Pre-processing script to generate dataframe from raw files'''

import os
import glob
import numpy as np
import pandas as pd
import math
import json
from tqdm import tqdm
from loguru import logger


class DataProcessing():
    def __init__(self, data_path: str, language_filter: list, extension: str='json') -> None:
        """Pre Prcoessing class to convert extension files to dataframe.

        Args:
            data_path (str): Datapath of raw files
            language_filter (list): List of Languages to extract data for
            extension (str, optional): File extension of raw data. Defaults to 'json'.
        """
        self.data_path = data_path
        self.listing_files = glob.glob(f'{self.data_path}/*.{extension}')
        self.language_filter = language_filter
    
    def merge_data_files(self):
        """Merge raw data files into a dataframe.
        """
        df_list = []
        unique_cols = set()
        logger.info('Reading raw files')
        for file in tqdm(self.listing_files):
            listings = []
            for line in open(file, 'r', ):
                listings.append(json.loads(line))
            temp = pd.json_normalize(listings)
            df_list.append(temp)
        self.merged_df = pd.concat(df_list)
        logger.info('Raw files merged')   
    
    def extract_text(self, row:str, use_set:bool=False, dedup:bool=False) -> str:
        """Method to extract raw text from the features.

        Args:
            row (str): Raw record
            use_set (bool, optional): Flag to extract unique values in raw record. Defaults to False.

        Returns:
            str: Formatted text
        """
        if use_set:
            column_val = set()
        else:
            column_val = []
        if isinstance(row, float):
            if math.isnan(row):
                if isinstance(column_val, set):
                    column_val.add(np.nan)
                else:
                    column_val.append(np.nan)
        elif row:
            for val in row:
                language_val = val.get('language_tag')
                if language_val in self.language_filter:
                    feature_val = val.get('value').lower()
                    if isinstance(column_val, set):
                        column_val.add(feature_val)
                    else:
                        if dedup:
                            if feature_val not in column_val:
                                column_val.append(feature_val)
                        else:
                            column_val.append(feature_val)
        if not column_val:
            column_val = np.nan
        elif np.nan not in column_val and isinstance(column_val, set):
            column_val = ' '.join(list(column_val))
        elif np.nan not in column_val:
            column_val = ' '.join(column_val)
        else:
            column_val = np.nan
        return column_val
    
    @staticmethod
    def get_non_ascii_text(row:str, key_val: str, remove_ascii: bool=False) -> str:
        """Method to extract raw text from the features with flag to remove
           ascii characters.

        Args:
            row (str): Raw record
            key_val (str): Key to fetch data from
            remove_ascii (bool, optional): Flag to remove ascii characters. Defaults to False.

        Returns:
            str: Formatted text with optionally removed ascii characters
        """
        text_name = np.nan
        if isinstance(row, float):
            if math.isnan(row):
                text_name = np.nan
        elif row:
            for val in row:
                text_val = val.get(key_val)
                if text_val:
                    text_name = text_val
                    if remove_ascii:
                        text_name = ''.join([i if ord(i) < 128 else '' for i in text_name])
                        if not text_name:
                            text_name = np.nan
        else:
            text_name = np.nan
        return text_name

    @staticmethod
    def get_normalized_value(row:str, key_val: str) -> str:
        """Method to extract normalized value from raw records.

        Args:
            row (str): Raw record
            key_val (str): Key to fetch data from

        Returns:
            str: Normalized value associated with the record
        """
        if isinstance(row, float):
            if math.isnan(row):
                text_val = np.nan
        elif row:
            text_name = row[0].get(key_val)
            if text_name: 
                text_val = text_name.get('value')
            else:
                text_val = np.nan
        else:
            text_val = np.nan
        return text_val
    
    def main(self):
        logger.info('Initializing data preprocessing class')
        self.merge_data_files()
        color_mapping = {'multi-colored': 'multicolor'}
        final_df = pd.DataFrame()
        extract_text_cols = ['brand', 'bullet_point', 'item_name', 'model_name', 'item_keywords',
                             'material', 'style', 'fabric_type', 'product_description', 'finish_type',
                             'item_shape', 'pattern']
        metadata_cols = ['item_id', 'marketplace', 'country', 'domain_name', 'country',
                         'main_image_id', 'other_image_id']
        extract_ascii_dict = {
                          'model_year': ['value', False],
                          'node': ['node_id', False],
                          'node': ['node_name', True]
                        }
        logger.info('Extracting text columns')
        for col_val in tqdm(extract_text_cols):
            final_df[col_val] = self.merged_df[col_val].apply(self.extract_text)
        logger.info('Extracting Ascii filtered text columns')
        for col_val in tqdm(extract_ascii_dict.keys()):
            key_val = extract_ascii_dict[col_val][0]
            remove_ascii_flag = extract_ascii_dict[col_val][1]
            final_df[key_val] = \
                self.merged_df[col_val].apply(self.get_non_ascii_text, key_val=key_val, remove_ascii=remove_ascii_flag)
        logger.info('Extracting metadata columns')
        for col_val in tqdm(metadata_cols):
            final_df[col_val] = self.merged_df[col_val]
        logger.info('Extracting weight, color and product type')
        final_df['weight'] = self.merged_df['item_weight'].apply(self.get_normalized_value, key_val='normalized_value')
        final_df['color'] = self.merged_df['color'].apply(self.extract_text, use_set=True).replace(color_mapping)
        final_df['product_type'] = self.merged_df['product_type'].apply(lambda x: x[0].get('value'))
        final_df['height'] = self.merged_df['item_dimensions.height.normalized_value.value']
        final_df['length'] = self.merged_df['item_dimensions.length.normalized_value.value']
        final_df['width'] = self.merged_df['item_dimensions.width.normalized_value.value']
        return final_df


if __name__=='__main__':
    DATA_PATH = os.path.abspath(__file__ + "/../../../data/raw_files")
    FILE_PATH = os.path.abspath(__file__ + "/../../../data")
    LANGUAGE_FILTER = ['en_IN', 'en_AE', 'en_US', 'en_CA', 'en_GB', 'en_SG', 'en_AU']
    pre_process = DataProcessing(data_path=DATA_PATH, language_filter=LANGUAGE_FILTER)
    final_df = pre_process.main()
    logger.info(f'Processed Data dimensions: {final_df.shape}')
    final_df.to_parquet(os.path.join(FILE_PATH, 'final_df_parquet'), index=False)
    logger.info(f'Written Data file to: {FILE_PATH}')