import pandas as pd
import os

from recbole.data import create_dataset, data_preparation

def generate_data(args):
    '''
    dataframe 불러와서 전처리 통해 result_df 만드는 과정은 추후 전처리 함수로 수정할 예정
    args: setting.yaml
    '''
    recbole_data_path = './data/movie_rec/'
    if os.path.exists(recbole_data_path):
        pass
    else:
        os.makedirs(recbole_data_path)
        rating_df = pd.read_csv(args.data_path + 'train_ratings.csv')
        directors_df = pd.read_csv(args.data_path + 'directors.tsv', delimiter='\t')
        genres_df = pd.read_csv(args.data_path + 'genres.tsv', delimiter='\t')
        titles_df = pd.read_csv(args.data_path + 'titles.tsv', delimiter='\t')
        writers_df = pd.read_csv(args.data_path + 'writers.tsv', delimiter='\t')
        years_df = pd.read_csv(args.data_path + 'years.tsv', delimiter='\t')
        result_df = rating_df.copy()

        genres_df = genres_df.groupby('item').agg(
            genre = ('genre', lambda x : list(x))
        ).reset_index()
        writers_df = writers_df.groupby('item').agg(
            writer = ('writer', lambda x : list(x))
        ).reset_index()

        dfs = [directors_df, titles_df, years_df, writers_df, genres_df]
            
        for df in dfs:
            result_df = pd.merge(result_df, df, on='item', how='left')

        rating_df.rename(columns={'user':'user_id:token', 
                                'item':'item_id:token', 
                                'time':'timestamp:float'}, inplace=True)
        
        item_df = result_df[['item', 'director', 'genre', 'year']].rename({'item':'item_id:token', 
                                                                           'year':'release_year:token', 
                                                                           'genre':'genre:token_seq'})
        
        rating_df.to_csv('./recbox_data/recbox_data.inter', index=False, sep='\t')
        item_df.to_csv('./recbox_data/recbox_data.item', index=False, sep='\t')