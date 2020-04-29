import sys
import time

def print_time(s):
    """Print string s and current time."""

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(f"{current_time} | {s}")


def print_header(s):
    """Print a nice header for string s."""

    print()
    print(f"##{'#'*len(s)}##")
    print(f"# {s} #")
    print(f"##{'#'*len(s)}##")
    print()


def validate_model_name(model_name):
    """Check if the inserted model name is valid."""

    model_names = [
        'gbm_rank', 'logistic_regression',
        'nn_interaction', 'nn_item',
        'pop_abs', 'pop_user', 
        'position', 'random'
    ]

    try:
        if model_name not in model_names: raise NameError
    except NameError:
        print("No such model. Please choose a valid one.")
        sys.exit(1)
