# %%
import pandas as pd
import numpy as np
import torch
import sys
from math import sqrt
from torch import optim
import torch.nn as nn
import argparse

from utils.set_seed import set_seed

from Dataset.custom_dataset import get_dataloader, load_manufacturing_data
from Chronos import ChronosForecaster

pd.set_option('display.max_columns', 50)



def get_argument_parser():
    parser = argparse.ArgumentParser()
    # Pretrain dataset directory and files
    parser.add_argument('--data_dir', type=str, default='/home/minjunson/project/Chronos_2.0/Models/Dataset', help='data directory')
    parser.add_argument('--pretrain_files', nargs='+', default=['ai4i2020.csv', 'IoT.csv', 'Steel_industry.csv'], help='pretraining dataset files')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='SAMYANG_dataset', help='dataset name')
    parser.add_argument('--target', type=str, default='SATURATOR_ML_SUPPLY_F_PV.Value', help='label') 
    parser.add_argument('--minute_interval', type=int, default=15, help='dataset collection interval')
    
    # Dataloader
    parser.add_argument('--seq_len', type=int, default=512, help='sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='label length')
    parser.add_argument('--pred_len', type=int, default=6, help='prediction length')
    #parser.add_argument('--step_size', type=int, default=1, help='data slicing step size')
    parser.add_argument('--model_name', type=str, default='Chronos', help='model name')

    # Train
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_steps', type=int, default=400, help='number of fine-tune steps')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--device_num', type=str, default='0', help='set gpu number')
    #parser.add_argument('--patience', type=int, default=100, help='train patience')
    #parser.add_argument('--num_workers', type=int, default=0, help='num_workers')

    # Chronos specific
    parser.add_argument('--use_chronos', default = False, action='store_true', help='run Chronos-2 inference instead of TCN training')
    parser.add_argument('--use_covariates',default = False,  action='store_true', help='use covariates for Chronos-2')
    parser.add_argument('--use_cross_learning', default = False, action='store_true', help='use cross learning for Chronos-2')
    parser.add_argument('--cov_corr_threshold', type=float, default=0.5, help='min abs correlation to keep a covariate')
    parser.add_argument('--fine_tune', default = False, action='store_true', help='fine-tune Chronos-2 model')
    parser.add_argument('--continual_pretrain', default = False, action='store_true', help='continual pretrain Chronos-2 model')

    #args = parser.parse_args(args=[])
    args = parser.parse_args()

    return args



# %%
if __name__ == "__main__":

    args = get_argument_parser()
    
    print("\n" + "="*60)
    print(f"{'Argument':<30} | {'Value'}")
    print("-" * 60)
    for arg, value in vars(args).items():
        print(f"{arg:<30} | {value}")
    print("="*60 + "\n")

    data = pd.read_csv('Dataset/{}.csv'.format(args.dataset), parse_dates=['TimeStamp'])
    print(f"Original shape: {data.shape} from Dataset/{args.dataset}.csv")

    cutoff_date = pd.Timestamp('2021-04-30')
    data = data[data['TimeStamp'] > cutoff_date].reset_index(drop= True)

    # Filter by date
    data = data[data['TimeStamp'] > cutoff_date].reset_index(drop=True)
    total_samples = len(data)

    # Count Outliers vs Normal
    n_outliers = int(data['outlier'].sum())
    n_normal = total_samples - n_outliers
    target_column = args.target
    target_idx = data.columns.get_loc(target_column)

    print(f"전체 샘플: {total_samples:,}개 (2021-04-30 이후)")
    print(f"Outlier: {n_outliers:,}개 ({n_outliers/total_samples*100:.2f}%)")
    print(f"Normal: {n_normal:,}개 ({n_normal/total_samples*100:.2f}%)")


    print(f"Target: {target_column} (index: {target_idx})")
    print(f"Date range: {data['TimeStamp'].min()} to {data['TimeStamp'].max()}")

    set_seed(args.seed)

    data_loader = {}
    data_loader['train'], y_scaler = get_dataloader(data, args, shuffle = True, flag = 'train',drop_last = True)
    data_loader['val'] = get_dataloader(data, args, shuffle = True, flag = 'val', drop_last = True)
    data_loader['test'] = get_dataloader(data, args, shuffle = False, flag = 'test', drop_last = False)

    n_train = len(data_loader['train'].dataset)
    n_val = len(data_loader['val'].dataset)
    n_test = len(data_loader['test'].dataset)
    total_seq = n_train + n_val + n_test

    print(f"\nTrain: {n_train:,} 시퀀스 ({n_train/total_seq*100:.2f}%)")
    print(f"Validation: {n_val:,} 시퀀스 ({n_val/total_seq*100:.2f}%)")
    print(f"Test: {n_test:,} 시퀀스 ({n_test/total_seq*100:.2f}%)")
    print(f"총 유효 시퀀스: {total_seq:,}개")


    # If use_chronos flag is set, run Chronos-2 inference on test loader
    forecaster = ChronosForecaster(args)
    if args.use_chronos and args.continual_pretrain:
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        # Load manufacturing datasets for continual pretraining
        manufacturing_datasets = load_manufacturing_data(
            args.data_dir,
            args.pretrain_files
        )
        
        # 1. Continual Pretraining
        forecaster.continual_pretrain(manufacturing_datasets)
        
        # 2. Fine-tuning on target dataset
        forecaster.fine_tune(data_loader)
        
        # 3. Inference
        result = forecaster.run(data_loader)
        
        print("Chronos-2 inference with continual pretraining + fine-tuning completed.")

    elif args.use_chronos and args.fine_tune and not args.continual_pretrain:
        forecaster.fine_tune(data_loader)
        print("Chronos-2 fine-tuning completed.")
        result = forecaster.run(data_loader)
    
    elif args.use_chronos:
        result = forecaster.run(data_loader)
        print("Chronos-2 zero-shot inference completed.")