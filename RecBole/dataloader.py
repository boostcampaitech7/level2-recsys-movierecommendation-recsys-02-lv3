import pandas as pd
import os

from recbole.data import create_dataset, data_preparation

def get_data(config):
    '''
        [description]
        RecBole 형식에 맞게 data를 생성하고, train/valid/test로 나누는 함수입니다.

        [arguments]
        config: RecBole에 필요한 config
    '''
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config=config, dataset=dataset)
    return train_data, valid_data, test_data

def generate_data(args, config):
    '''
        [description]
        train data를 불러오고 전처리한 후, RecBole의 .item, .inter 데이터로 변환하는 함수입니다.

        [arguments]
        args: 기본 세팅 args
        config: RecBole에 필요한 config
    '''

    if os.path.exists(config['data_path']):
        pass
    else:
        print("==== generating data ====")
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
        
        recbole_data_path, recbole_data_name = config['data_path'], config['dataset']
        os.makedirs(recbole_data_path)
        rating_df.to_csv(f'{recbole_data_path}/{recbole_data_name}.inter', index=False, sep='\t')
        item_df.to_csv(f'{recbole_data_path}/{recbole_data_name}.item', index=False, sep='\t')