''' Pre-processing sciprt to generate dataframe from raw files'''

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
    
    def extract_text(self, row:str, use_set:bool=False) -> str:
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
        elif len(row):
            for val in row:
                language_val = val.get('language_tag')
                if language_val in self.language_filter:
                    feature_val = val.get('value').lower()
                    if feature_val not in column_val:
                        if isinstance(column_val, set):
                            column_val.add(feature_val)
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
        elif len(row):
            for val in row:
                text_val = val.get(key_val)
                if text_val:
                    text_name = text_val
                    if remove_ascii:
                        text_name = ''.join([i if ord(i) < 128 else '' for i in text_name])
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
        elif len(row):
            text_name = row[0].get(key_val)
            text_val = text_name.get('value')
        else:
            text_val = np.nan
        return text_val