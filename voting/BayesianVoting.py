import pandas as pd
from data.DataLoading import df

def bay_consensus():

    df_length = len(df["Expert 1"])
    post_normal = [0] * df_length
    post_abnormal = [0] * df_length
    for x in range(0, df_length):
        df_length = 0

bay_consensus(df)