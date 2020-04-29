import numpy as np
import pandas as pd

def produce_train_data():
    """Produce example training data to test the model runs."""

    d_train = {
        "user_id": [
            "64BL89", "64BL89", "64BL89", "64BL89",
            "64BLF", "64BLF",
            "64BL89", "64BL89", "64BL89", "64BL89"
        ],
        "session_id": [
            "3579f89", "3579f89", "3579f89", "3579f89",
            "4504h9", "4504h9",
            "5504hFL", "5504hFL", "5504hFL", "5504hFL"
        ],
        "timestamp": [
            1, 2, 3, 4,
            2, 4,
            7, 8, 9, 10
        ],
        "step": [
            1, 2, 3, 4,
            1, 2,
            1, 2, 3, 4
        ],
        "action_type": [
            "interaction item image", "clickout item", 
                "interaction item info", "filter selection",
            "interaction item image", "clickout item",
                "filter selection", "clickout item", 
            "interaction item image", "clickout item"
        ],
        "reference": [
            "5001", "5002", "5003", "unknown",
            "5010", "5001",
            "unknown", "5004", "5001", "5001"
        ],
        "impressions": [
            np.NaN, "5014|5002|5010", np.NaN, np.NaN,
            np.NaN, "5001|5023|5040|5005",
            np.NaN, "5010|5001|5023|5004|5002|5008", 
                np.NaN, "5010|5001|5023|5004|5002|5008"
        ],
        "prices": [
            np.NaN, "100|125|120", np.NaN, np.NaN,
            np.NaN, "75|110|65|210",
            np.NaN, "120|89|140|126|86|110", np.NaN, "120|89|140|126|86|110"
        ]
    }

    df_train = pd.DataFrame(d_train)

    return df_train


def produce_test_data():
    """Produce example test data to test the model runs."""

    d_test = {
        "user_id": [
            "64BL89", "64BL89",
            "64BL91F2", "64BL91F2", "64BL91F2"
        ],
        "session_id": [
            "3579f90", "3579f90",
            "3779f92", "3779f92", "3779f92"
        ],
        "timestamp": [
            5, 6,
            9, 10, 11
        ],
        "step": [
            1, 2,
            1, 2, 3
        ],
        "action_type": [
            "interaction item image", "clickout item",
            "interaction item info", "clickout item", "filter selection"
        ],
        "reference": [
            "5023", np.NaN,
            "5010", np.NaN, "unknown"
        ],
        "impressions": [
            np.NaN, "5002|5003|5010|5004|5001|5023",
            np.NaN, "5001|5004|5010|5014", np.NaN
        ],
        "prices": [
            np.NaN, "120|75|110|105|89|99",
            np.NaN, "76|102|115|124", np.NaN
        ]
    }

    df_test = pd.DataFrame(d_test)

    return df_test


def produce_item_metadata():
    """Produce example item metadata to test the model runs."""

    d_item_meta = {
        "item_id": ["5001", "5002", "5003", "5004"],
        "properties": [
            "Wifi|Croissant|TV",
            "Wifi|TV",
            "Croissant",
            "Shoe dryer"
        ]
    }

    df_item_meta =  pd.DataFrame(d_item_meta)

    return df_item_meta
