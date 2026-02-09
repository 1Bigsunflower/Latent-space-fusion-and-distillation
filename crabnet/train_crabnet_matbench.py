import argparse
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matbench.bench import MatbenchBenchmark
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

import train_crabnet
from train_crabnet import get_model, load_model, get_results

parser = argparse.ArgumentParser(description='')
parser.add_argument('--emb_method', default='mat2vec', type=str,
                    help='embedding methods to use')
parser.add_argument('--subset', default='matbench_jdft2d', type=str,
                    # choices=['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric', 'matbench_log_gvrh', 'matbench_log_kvrh', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form', 'matbench_glass'],
                    help='subset dataset to use')
parser.add_argument('--fold', type=int, default=0, help='number of fold')
args = parser.parse_args()
# torch.set_printoptions(threshold=torch.inf)
# torch.set_printoptions(sci_mode=False)
# Defining a number of helper function to prepare the data for the CrabNet algorithm

# condense_formula takes a material and returns the chemical formula in the correct format for CrabNet
def condense_formula(mat):
    if isinstance(mat, str):
        return mat
    else:
        return mat.formula.replace(' ', '')


# change_input runs condense_formula on all the input data used for training
def change_input(train_inputs):
    inputs = []
    for input in train_inputs:
        inputs.append(condense_formula(input))
    return inputs


# make_df creates a data frame containing the train inputs and outputs for CrabNet
def make_df(train_inputs, train_outputs):
    input_df = pd.DataFrame({'formula': train_inputs, 'target': train_outputs})
    return input_df


# make_df_test creates a data frame containing the test inputs for CrabNet
def make_df_test(test_inputs):
    test_df = pd.DataFrame({'formula': test_inputs})
    test_df['target'] = np.nan
    return test_df


# split_train_val splits the training data into two sets: training and validation
def split_train_val(df):
    df = df.sample(frac=1.0, random_state=7)
    val_df = df.sample(frac=0.25, random_state=7)
    train_df = df.drop(val_df.index)

    return train_df, val_df


# Defining a subset containing all of the regression tasks from the matbench tasks
if __name__ == '__main__':
    subset = [
        args.subset
              #"matbench_jdft2d",
              # "matbench_steels",
              # "matbench_perovskites",
              # "matbench_expt_gap",
              # "matbench_phonons",
              # "matbench_dielectric",
              # "matbench_log_gvrh",
              # "matbench_log_kvrh",
              # "matbench_mp_gap",
              # "matbench_mp_e_form"
              ]

    mb = MatbenchBenchmark(autoload=False, subset=subset)
    data_dir = 'data/matbench_temp'
    os.makedirs(data_dir, exist_ok=True)

    results_dict = {}

    for task in mb.tasks:
        task.load()

        task_type = task.metadata.task_type
        is_classification = task_type == "classification"

        fold = args.fold
        mat_prop = f'{task.dataset_name}_fold{fold}_{args.emb_method}'
        os.makedirs(f'{data_dir}/{mat_prop}', exist_ok=True)

        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

        # Preparing the inputs data for CrabNet
        inputs = change_input(train_inputs)
        df = make_df(inputs, train_outputs)

        # Creating the training and validation sets
        train_df, val_df = split_train_val(df)
        train_df.to_csv(f'{data_dir}/{mat_prop}/train.csv')
        val_df.to_csv(f'{data_dir}/{mat_prop}/val.csv')

        # Getting and preparing the testing data
        test_inputs = change_input(test_inputs)
        output_df = make_df(test_inputs, test_outputs)
        output_df.to_csv(f'{data_dir}/{mat_prop}/test.csv')

        # Training CrabNet
        model = get_model(data_dir, mat_prop, classification=is_classification, verbose=True, embedding_dir=f'{args.emb_method}'
                          # drop_unary=False
                          )

        # Predicting on the testing data
        model = load_model(data_dir, mat_prop, classification=is_classification, file_name='test.csv', verbose=True, embedding_dir=f'{args.emb_method}'
                           # drop_unary=False
                           )
        model, output = get_results(model)
        y_tar = output[0]
        y_pre = output[1]

        if is_classification:
            auc = roc_auc_score(y_tar, y_pre)
            print("ROC-AUC =", auc)
        else:
            mae = mean_absolute_error(y_tar, y_pre)
            mse = mean_squared_error(y_tar, y_pre)

            print("MAE =", mae)
            print("MSE =", mse)

