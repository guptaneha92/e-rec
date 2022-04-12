
import os
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname( os.getcwd()), 'data')
FILE_NAME = 'AmazonBerkeleyDatasetOutput.parquet'
FIL_PATH_PATH = os.path.join(DATA_DIR, FILE_NAME)
META_DATA_PATH = os.path.join(DATA_DIR, 'image_files', 'images.csv')
IMAGE_DIR = os.path.join(DATA_DIR, 'image_files', 'images', 'small')
FINAL_NAME = 'final.csv'
FINAL_PATH =  os.path.join(DATA_DIR, FINAL_NAME)


class Preprocess:

    def read_data(self):
        raw_df = pd.read_parquet(FIL_PATH_PATH)
        raw_df = raw_df[raw_df['domain_name'] == 'amazon.com'].reset_index(drop=True)
        raw_df = raw_df.iloc[: , 1:]
        self.df = raw_df

    def get_category_name(self, index_val=2):
        cat_name = []
        mapping_dict = {}
        for val in self.df[~self.df['node_name'].isna()]['node_name']:
            node_len = len(val.split('/'))
            if node_len > index_val:
                node_val = val.split('/')[index_val]
                mapping_dict[val] = node_val
            else:
                node_val = val
                mapping_dict[val] = node_val
            cat_name.append(node_val)
        self.df['cat_name'] = self.df['node_name'].replace(mapping_dict)

    def merge_metadata(self):
        metadata = pd.read_csv(META_DATA_PATH)
        metadata['image_path'] = metadata['path'].apply(lambda x: os.path.join(IMAGE_DIR, x))
        image_path_map = dict(zip(metadata.image_id,metadata.image_path))

        self.df = self.df[~self.df['main_image_id'].isna()]
        self.df['cat_prod_name'] = self.df['cat_name'] + '_' + self.df['product_type']

        filter_cat_name = list(self.df['cat_name'].value_counts(ascending=True)[self.df['cat_name'].value_counts(ascending=True) > 100].index)
        self.df['use_cat_flag'] = self.df['cat_name'].isin(filter_cat_name)
        self.df = self.df[self.df['use_cat_flag'] == 1].reset_index(drop=True)
        filter_prod = list(self.df['product_type'].value_counts(ascending=True)[self.df['product_type'].value_counts(ascending=True) > 50].index)
        self.df['use_prod_flag'] = self.df['product_type'].isin(filter_prod)
        self.df = self.df[self.df['use_prod_flag'] == 1].reset_index(drop=True)
        self.df['image_path'] = self.df['main_image_id'].map(image_path_map)
        self.df = self.df[~self.df['image_path'].isna()].reset_index(drop=True)

    def main(self):
        self.read_data()
        self.get_category_name()
        self.merge_metadata()
        self.df.to_csv(FINAL_PATH, index=False)

if __name__=='__main__':
    pre_process_data = Preprocess()
    pre_process_data.main()
                