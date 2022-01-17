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