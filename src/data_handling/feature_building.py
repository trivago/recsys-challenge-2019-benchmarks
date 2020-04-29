import pandas as pd
import numpy as np

import data_handling.helper_functions as f
import data_handling.data_frame_functions as dff


def build_features(df):
    """Build features for the lightGBM and logistic regression model."""

    # Select columns that are of interest for this method
    f.print_time("start")
    cols = ['user_id', 'session_id', 'timestamp', 'step',
            'action_type', 'reference', 'impressions', 'prices']
    df_cols = df.loc[:, cols] 

    # We are only interested in action types, for wich the reference is an item ID
    f.print_time("filter interactions")
    item_interactions = [
        'clickout item', 'interaction item deals', 'interaction item image',
        'interaction item info', 'interaction item rating', 'search for item'
    ]
    df_actions = (
        df_cols
        .loc[df_cols.action_type.isin(item_interactions), :]
        .copy()
        .rename(columns={'reference': 'referenced_item'})
    )

    f.print_time("cleaning")
    # Clean of instances that have no reference
    idx_rm = (df_actions.action_type != "clickout item") & (df_actions.referenced_item.isna())
    df_actions = df_actions[~idx_rm]

    # Get item ID of previous interaction of a user in a session
    f.print_time("previous interactions")
    df_actions.loc[:, "previous_item"] = (
        df_actions
        .sort_values(by=["user_id", "session_id", "timestamp", "step"],
                        ascending=[True, True, True, True])
        .groupby(["user_id"])["referenced_item"]
        .shift(1)
    )

    # Combine the impressions and item column, they both contain item IDs
    # and we can expand the impression lists in the next step to get the total
    # interaction count for an item
    f.print_time("combining columns - impressions")
    df_actions.loc[:, "interacted_item"] = np.where(
        df_actions.impressions.isna(),
        df_actions.referenced_item,
        df_actions.impressions
    )
    df_actions = df_actions.drop(columns="impressions")

    # Price array expansion will get easier without NAs
    f.print_time("combining columns - prices")
    df_actions.loc[:, "prices"] = np.where(
        df_actions.prices.isna(),
        "",
        df_actions.prices
    )

    # Convert pipe separated lists into columns
    f.print_time("explode arrays")
    df_items = dff.explode_mult(df_actions, ["interacted_item", "prices"]).copy()

    # Feature: Number of previous interactions with an item
    f.print_time("interaction count")
    df_items.loc[:, "interaction_count"] = (
        df_items
        .groupby(["user_id", "interacted_item"])
        .cumcount()
    )

    # Reduce to impression level again 
    f.print_time("reduce to impressions")
    df_impressions = (
        df_items[df_items.action_type == "clickout item"]
        .copy()
        .drop(columns="action_type")
        .rename(columns={"interacted_item": "impressed_item"})
    )

    # Feature: Position of item in the original list.
    # Items are in original order after the explode for each index
    f.print_time("position feature")
    df_impressions.loc[:, "position"] = (
        df_impressions
        .groupby(["user_id", "session_id", "timestamp", "step"])
        .cumcount()+1
    )

    # Feature: Is the impressed item the last interacted item
    f.print_time("last interacted item feature")
    df_impressions.loc[:, "is_last_interacted"] = (
        df_impressions["previous_item"] == df_impressions["impressed_item"]
    ).astype(int)

    f.print_time("change price datatype")
    df_impressions.loc[:, "prices"] = df_impressions.prices.astype(int)

    return_cols = [
        "user_id",
        "session_id",
        "timestamp",
        "step",
        "position",
        "prices",
        "interaction_count",
        "is_last_interacted",
        "referenced_item",
        "impressed_item",
    ]

    df_return = df_impressions[return_cols]

    return df_return
