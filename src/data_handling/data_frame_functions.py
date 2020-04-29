import numpy as np
import pandas as pd

def explode(df, col_expl):
    """Separate string in column col_expl and explode elements into multiple rows."""

    s = df[col_expl].str.split('|', expand=True).stack()
    i = s.index.get_level_values(0)
    df2 = df.loc[i].copy()
    df2[col_expl] = s.values

    return df2


def explode_mult(df_in, col_list):
    """Explode each column in col_list into multiple rows."""

    df = df_in.copy()

    for col in col_list:
        df.loc[:, col] = df.loc[:, col].str.split("|")

    df_out = pd.DataFrame(
        {col: np.repeat(df[col].to_numpy(),
                        df[col_list[0]].str.len())
         for col in df.columns.drop(col_list)}
    )

    for col in col_list:
        df_out.loc[:, col] = np.concatenate(df.loc[:, col].to_numpy())

    return df_out


def group_concat(df, gr_cols, col_concat):
    """Concatenate multiple rows into one."""

    df_out = (
        df
        .groupby(gr_cols)[col_concat]
        .apply(lambda x: ' '.join(x))
        .to_frame()
        .reset_index()
    )

    return df_out


def get_target_rows(df):
    """Restrict data frame to rows for which a prediction needs to be made."""
    
    df_target = df[
        (df.action_type == "clickout item") & 
        (df["reference"].isna())
    ]

    return df_target


def summarize_recs(df, rec_col):
    """Bring the data frame into submission format."""

    df_rec = (
        df
        .sort_values(by=["user_id", "session_id", "timestamp", "step", rec_col],
                        ascending=[True, True, True, True, False])
        .groupby(["user_id", "session_id", "timestamp", "step"])["impressed_item"]
        .apply(lambda x: ' '.join(x))
        .to_frame()
        .reset_index()
        .rename(columns={'impressed_item': 'item_recommendations'})
    )

    return df_rec
