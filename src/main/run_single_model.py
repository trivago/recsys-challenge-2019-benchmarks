import click
import pandas as pd
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

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')


#@click.command()
@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files.')
@click.option('--train-file', default='train.csv', help='Training data file.')
@click.option('--test-file', default='test.csv', help='Test data file.')
@click.option('--subm-file', default=None, help='Output file for submission.')
@click.option('--model-name', default=None,
help= """Model name. Must be one of the following: \
gbm_rank, log_reg, nn_interaction, nn_item, \
pop_user, pos_alg, position, random.""")

def main(data_path, train_file, test_file, subm_file, model_name):
    """
    This script runs a single model.

    \b
    The following models are supported:
    - gbm_rank: lightGBM model
    - log_reg: Logistic regression
    - nn_interaction: kNN w/ session co-occurrence
    - nn_item: kNN w/ metadata similarity
    - pop_abs: Popularity - total clicks
    - pop_user: Popularity - distinct users
    - position: Original display position
    - random: Random order
    """

    f.validate_model_name(model_name)

    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = data_directory.joinpath(train_file)
    test_csv = data_directory.joinpath(test_file)
    subm_csv = data_directory.joinpath(subm_file)

    models = {
        'gbm_rank': ModelGbmRank(),
        'log_reg': ModelLogReg(),
        'nn_interaction': ModelNNInteraction(),
        'nn_item': ModelNNItem(),
        'pop_abs': ModelPopAbs(),
        'pop_user': ModelPopUsers(),
        'position': ModelPosition(),
        'random': ModelRandom()
    }

    model = models[model_name]

    f.print_header(f"Run model {model_name}")

    print(f"Reading {train_csv} ...")
    df_train = pd.read_csv(train_csv)

    print(f"Fit model ...")
    model.fit(df_train)

    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv)

    print(f"Calculate recommendations ...")
    df_recommendations = model.predict(df_test)

    print(f"Writing {subm_csv}...")
    df_recommendations.to_csv(subm_csv, index=False)

    print("Finished calculating recommendations.")


if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
