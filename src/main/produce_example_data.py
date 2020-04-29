from pathlib import Path
import data_handling.test_file_creation as tf
import data_handling.helper_functions as f

import click

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')


@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
def main(data_path):
    """
    This script produces example data.

    \b
    The following small example files are produced:
    - train_example.csv
    - test_example.csv
    - item_metadata_example.csv

    \b
    The files can be used to test the execution of the models.
    """

    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory

    f.print_header("Produce example data")

    df_train = tf.produce_train_data()
    df_test = tf.produce_test_data()
    df_meta = tf.produce_item_metadata()

    train_csv = data_directory.joinpath("train_example.csv")
    test_csv =  data_directory.joinpath("test_example.csv")
    meta_csv =  data_directory.joinpath("item_metadata_example.csv")

    print(f"Writing {train_csv}...")
    df_train.to_csv(train_csv, index=False)
    print(f"Writing {test_csv}...")
    df_test.to_csv(test_csv, index=False)
    print(f"Writing {meta_csv}...")
    df_meta.to_csv(meta_csv, index=False)

    print("Finished producing example data.")

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
    