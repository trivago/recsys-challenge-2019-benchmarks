from pathlib import Path

from models.model_gbm_rank import ModelGbmRank
from models.model_log_reg import ModelLogReg
from models.model_nn_interaction import ModelNNInteraction
from models.model_nn_item import ModelNNItem
from models.model_pop_absolute import ModelPopAbs
from models.model_pop_users import ModelPopUsers
from models.model_position import ModelPosition
from models.model_random import ModelRandom
import data_handling.helper_functions as f

import click
import pandas as pd

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')


@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
@click.option('--train-file', default='train.csv', help='Training data file')
@click.option('--meta-file', default='item_metadata.csv', help='Meta data file')
@click.option('--test-file', default='test.csv', help='Test data file')
def main(data_path, meta_file, train_file, test_file):
    """
    This script runs all models at once.

    \b
    The following models are run:
    - gbm_rank: lightGBM model
    - log_reg: Logistic regression
    - nn_interaction: kNN w/ session co-occurrence
    - nn_item: kNN w/ metadata similarity
    - pop_abs: Popularity - total clicks
    - pop_user: Popularity - distinct users
    - position: Original display position
    - random: Random order
    """

    # Calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = data_directory.joinpath(train_file)
    meta_csv = data_directory.joinpath(meta_file)
    test_csv = data_directory.joinpath(test_file)

    f.print_header("Running all models")

    print(f"Reading {train_csv} ...")
    df_train = pd.read_csv(train_csv)
    print(f"Reading {meta_csv} ...")
    df_meta = pd.read_csv(meta_csv)
    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv)

    model_paras = {
        'gbm_rank': [
            ModelGbmRank(), 
            df_train, 
            df_test,
        ],
        'log_reg': [
            ModelLogReg(), 
            df_train, 
            df_test,
        ],
        'nn_interaction': [
            ModelNNInteraction(), 
            df_train, 
            df_test,
        ],
        'nn_item': [
            ModelNNItem(), 
            df_meta, 
            df_test,
        ],
        'pop_abs': [
            ModelPopAbs(), 
            df_train, 
            df_test,
        ],
        'pop_user': [
            ModelPopUsers(), 
            df_train, 
            df_test,
        ],
        'position': [
            ModelPosition(), 
            None, 
            df_test,
        ],
        'random': [
            ModelRandom(), 
            None, 
            df_test,
        ]
    }

    for model in model_paras.keys():
        print()
        print(f"Running model {model} ...")
        print()
        subm_file = f"submission_{model}.csv"
        subm_csv = data_directory.joinpath(subm_file)
        model, df_train, df_test = model_paras[model]

        print(f"Fit model ...")
        model.fit(df_train)

        print(f"Calculate recommendations ...")
        df_rec = model.predict(df_test)

        print(f"Writing {subm_csv}...")
        df_rec.to_csv(subm_csv, index=False)

    print()
    print("Finished calculating recommendations.")

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
