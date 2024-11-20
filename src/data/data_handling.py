import os
import pandas as pd
from scipy import sparse
import numpy as np
import re

##### data handing function

def preprocessing_time(data, time_drop=True):
    '''
    data : 'time' 컬럼을 가진 데이터 프레임
    time 컬럼을 'time_year', 'month', 'day_of_week', 'hour'로 변경하여 데이터 프레임 반환
    time_drop : 반환 시 time 컬럼을 드랍할 것인지 결정 (default=True)
    '''
    data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
    
    data['time_year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month
    data['day_of_week'] = data['time'].dt.dayofweek  # 0=월요일, 6=일요일
    data['hour'] = data['time'].dt.hour

    if time_drop:
        data.drop(columns=['time'], inplace=True)

    return data

def handle_missing_value(data):
    '''
    data : 영화 메타 데이터(director, writer, year, title)를 포함한 데이터 프레임
    director, writer, year에 대한 결측치 처리 후 반환
    '''
    # 'director' 컬럼의 결측치는 'unknown'으로 채우기
    data['director'] = data['director'].fillna('unknown')
    
    # 'writer' 컬럼의 결측치는 'unknown'으로 채우기
    data['writer'] = data['writer'].fillna('unknown')
    
    # 'year' 결측치는 'title' 컬럼의 끝부분에 있는 (year) 부분을 추출해서 채우기
    def extract_year_from_title(title):
        '''
        제목에서 (year) 형태의 개봉 연도를 추출하는 함수
        '''
        match = re.search(r'\((\d{4})\)', title)
        if match:
            return float(match.group(1))  # 'year'는 float64 타입이므로 변환
        return None
    
    data['year'] = data['year'].fillna(data['title'].apply(extract_year_from_title))
    
    return data