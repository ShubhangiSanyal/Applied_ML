# Assignment 2: Experiment Tracking

1. **data version control**<br>
    in prepare.ipynb track the versions of data using dvc
    - load the raw data into raw_data.csv and save the split data into train.csv/validation.csv/test.csv
    - update train/validation/test split by choosing different random seed
    - checkout the first version (before update) using dvc and print the distribution of target variabl (number of 0s and number of 1s) in train.csv, validation.csv, and test.csv
    - checkout the updated version using dvc and print the distribution of target variable in train.csv, validation.csv, test.csv<br>
    - **bonus:** (decouple compute and storage) track the data versions using google drive as storage
2. **model version control and experiment tracking**<br>
    in train.ipynb track the experiments and model versions using mlflow
    - build, track, and register 3 benchmark models using MLflow
    - checkout and print AUCPR for each of the three benchmark models
