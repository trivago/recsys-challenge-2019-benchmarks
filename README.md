# RecSys Challenge 2019

This repository contains code to reproduce the benchmarks calculated for the data from the [ACM RecSys Challenge 2019](http://www.recsyschallenge.com/2019/) organized by trivago, TU Wien, Politecnico di Milano, and Karlsruhe Institute of Technology.

## Installation and usage

[Optional] Before installing the code, you can create a virtual environment so
the installed packages don't get mixed with the ones in your system. To do it,
execute the following commands in your terminal:

    python3 -m venv trvrecsys2019benchmarks
    source trvrecsys2019benchmarks/bin/activate

This will create a folder in the current directory which will contain the Python executable files.

To install the package and its dependencies use:

    pip install git+https://github.com/trivago/recsys-challenge-2019-benchmarks.git#egg=trvrecsys2019benchmarks

### Producing example data
Before running any models you can create a small example data set to test the execution of the models before running them with the bigger RecSysChallenge data sets.

    produce-example-data --data-path <target-location-for-csv-files>

### Running a single model
To run an individual model, run:

    run-single-model --data-path <path-to-csv-files-directory> --train-file <training-data-file> --test-file <test-data-file> --subm-file <submission-file-to-be-created> --model-name <name-of-model>

To see what models are available, run:

    run-single-model --help

Note, that some models need a different set of training data. For example, the nn-item model relies on the item_metadata.csv as input.

### Running all models
You can run all models at once. This might take a while and the name of the submission files will be inferred from the model names. To do so, run:

    run-all-models --data-path <path-to-csv-files-directory> --train-file <training-data-file> --test-file <test-data-file> --meta-file <item-metadata-file>
