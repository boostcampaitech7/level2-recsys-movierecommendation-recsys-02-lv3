import pandas as pd

def total_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser

    Returns
    -------
    data : pd.Dataframe
        train_rating.csv, directors.tsv, genres.tsv, titls.tsv, writers.tsv, years.tsv를 모두 합친 데이터프레임을 반환합니다.
    """
    path = args.dataset.data_path
    rating_df = pd.read_csv(path + 'train_ratings.csv')
    directors_df = pd.read_csv(path + 'directors.tsv', delimiter='\t')
    genres_df = pd.read_csv(path + 'genres.tsv', delimiter='\t')
    titles_df = pd.read_csv(path + 'titles.tsv', delimiter='\t')
    writers_df = pd.read_csv(path + 'writers.tsv', delimiter='\t')
    years_df = pd.read_csv(path + 'years.tsv', delimiter='\t')
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

    return result_df