import pandas as pd
import os
from pytz import timezone
from collections import defaultdict
import numpy as np

import warnings
warnings.filterwarnings(action="ignore")

class Dataloader():
    def __init__(self, args):
        self.args = args
        self.df = pd.read_csv(os.path.join(self.args.dataset.data_path, 'train_ratings.csv'))

        self.users = self.df.user.unique()
        self.items = self.df.item.unique()
        

    def dataloader(self):
        return self.df, self.users, self.items
