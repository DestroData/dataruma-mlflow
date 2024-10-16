import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def ingest(data_version):

    data = pd.read_csv(f'data/data-{data_version}.csv')

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    return train_data, test_data
