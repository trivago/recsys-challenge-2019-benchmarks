import numpy as np
import pandas as pd
from scipy import sparse

import data_handling.helper_functions as f
import data_handling.data_frame_functions as dff


def calc_item_sims(df, item_col, reference_col):
    """Calculate similarity of items based on nearest neighbor algorithm.

    The final data frame will have similarity scores for pairs of items.

    :param df: Data frame of training data
    :param item_col: Name of data frame column that contains the item ID
    :param reference_col: Name of the reference column, depending on the model either
        1. session_id for the similarity based on session co-occurrences
        2. properties for the similarity based on item metadata
    :return: Data frame with item pairs and similarity scores
    """

    # Create data frame with item and reference indices
    f.print_time("item and reference indices")
    unique_items = df[item_col].unique()
    unique_refs = df[reference_col].unique()

    d_items = {item_col: unique_items, 'item_idx': range(0, len(unique_items))}
    d_refs = {reference_col: unique_refs, 'ref_idx': range(0, len(unique_refs))}

    df_items = pd.DataFrame(data=d_items)
    df_refs = pd.DataFrame(data=d_refs)

    df = (
        df
        .merge(
            df_items,
            how="inner",
            on=item_col
        )
        .merge(
            df_refs,
            how="inner",
            on=reference_col
        )
    )

    df_idx = (
        df
        .loc[:, ["item_idx", "ref_idx"]]
        .assign(data=lambda x: 1.)
        .drop_duplicates()
    )

    # Build item co-ooccurrence matrix
    f.print_time("item co-occurrence matrix")
    mat_coo = sparse.coo_matrix((df_idx.data, (df_idx.item_idx, df_idx.ref_idx)))
    mat_item_coo = mat_coo.T.dot(mat_coo)

    # Calculate Cosine similarities
    f.print_time("Cosine similarity")
    inv_occ = np.sqrt(1 / mat_item_coo.diagonal())
    cosine_sim = mat_item_coo.multiply(inv_occ)
    cosine_sim = cosine_sim.T.multiply(inv_occ)

    # Create item similarity data frame
    f.print_time("item similarity data frame")
    idx_ref, idx_item, sim = sparse.find(cosine_sim)
    d_item_sim = {'idx_ref': idx_ref, 'idx_item': idx_item, 'similarity': sim}
    df_item_sim = pd.DataFrame(data=d_item_sim)

    df_item_sim = (
        df_item_sim
        .merge(
            df_items.assign(item_ref=df_items[item_col]),
            how="inner",
            left_on="idx_ref",
            right_on="item_idx"
        )
        .merge(
            df_items.assign(item_sim=df_items[item_col]),
            how="inner",
            left_on="idx_item",
            right_on="item_idx"
        )
        .loc[:, ["item_ref", "item_sim", "similarity"]]
    )

    return df_item_sim


def predict_nn(df, df_item_sim):
    """Calculate predictions based on the item similarity scores."""

    # Select columns that are of interest for this function
    f.print_time("start")
    cols = ['user_id', 'session_id', 'timestamp', 'step',
            'action_type', 'reference', 'impressions']
    df_cols = df.loc[:, cols] 

    # Get previous reference per user
    f.print_time("previous reference")
    df_cols["previous_reference"] = (
        df_cols
        .sort_values(by=["user_id", "session_id", "timestamp"],
                     ascending=[True, True, True])
        .groupby(["user_id"])["reference"]
        .shift(1)
    )

    # Target row, withheld item ID that needs to be predicted
    f.print_time("target rows")
    df_target = dff.get_target_rows(df_cols)

    # Explode to impression level
    f.print_time("explode impression array")
    df_impressions = dff.explode(df_target, "impressions")

    df_item_sim["item_ref"] = df_item_sim["item_ref"].astype(str)
    df_item_sim["item_sim"] = df_item_sim["item_sim"].astype(str)

    # Get similarities
    f.print_time("get similarities")
    df_impressions = (
        df_impressions
        .merge(
            df_item_sim,
            how="left",
            left_on=["previous_reference", "impressions"],
            right_on=["item_ref", "item_sim"]
        )
        .fillna(value={'similarity': 0})
        .sort_values(by=["user_id", "timestamp", "step", "similarity"],
                        ascending=[True, True, True, False])
    )

    # Summarize recommendations
    f.print_time("summarize recommendations")
    df_rec = dff.group_concat(
        df_impressions, ["user_id", "session_id", "timestamp", "step"], 
        "impressions"
    )

    df_rec = (
        df_rec
        .rename(columns={'impressions': 'item_recommendations'})
        .loc[:, ["user_id", "session_id", "timestamp", "step", "item_recommendations"]]
    )

    return df_rec
