import pandas as pd
import numpy as np
import argparse
import sys
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from utils.set_seed import set_seed
from Dataset.custom_dataset import get_dataloader

# Use only 1 GPU if available (though not needed for LR, keeps env consistent)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_argument_parser():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='SAMYANG_dataset', help='dataset name')
    parser.add_argument('--target', type=str, default='SATURATOR_ML_SUPPLY_F_PV.Value', help='label')
    parser.add_argument('--minute_interval', type=int, default=15, help='dataset collection interval')

    #Model
    parser.add_argument('--model_name', type=str, default='LinearRegression', help='model name')

    # Dataloader
    parser.add_argument('--seq_len', type=int, default=512, help='sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='label length')
    parser.add_argument('--pred_len', type=int, default=6, help='prediction length')
    parser.add_argument('--batch_size', type=int, default=128, help='pretrain batch size')
    parser.add_argument('--seed', type=int, default=1, help='seed')

    args = parser.parse_args()
    return args

def run_linear_regression_baseline(dataset, args):
    """
    Performs per-window Linear Regression (Trend Extrapolation) on the test dataset.
    Matches the logic of ChronosForecaster's window slicing.
    """
    df = dataset.data.copy().reset_index(drop=True)
    target_col = args.target
    seq_len = args.seq_len
    pred_len = args.pred_len
    total_len = seq_len + pred_len
    
    metrics = []
    
    print(f"Running Linear Regression Baseline on {len(dataset.using_index_list)} windows...")
    
    for window_id, end in enumerate(dataset.using_index_list):
        start = end - total_len
        if start < 0:
            continue

        window = df.iloc[start:end]
        
        # Skip if any outlier in the window (though dataset.using_index_list should handle this if logic matches)
        if "outlier" in window.columns and window["outlier"].any():
            continue

        context = window.iloc[:seq_len]
        future = window.iloc[seq_len:]
        
        if len(context) < seq_len or len(future) < pred_len:
            continue

        # Prepare Training Data (Context)
        # X: Time index (0 to seq_len-1)
        # y: Target value
        X_train = np.arange(seq_len).reshape(-1, 1)
        y_train = context[target_col].values
        
        # Fit Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        # Predict (Future)
        # X: Time index (seq_len to seq_len + pred_len - 1)
        X_pred = np.arange(seq_len, seq_len + pred_len).reshape(-1, 1)
        y_pred = lr.predict(X_pred)
        y_true = future[target_col].values
        
        # Compute Metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics.append({
            'id': window_id,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        })
    print(f"y_true.shape: {y_true.shape}, y_pred.shape: {y_pred.shape}, X_train.shape: {X_train.shape}, X_pred.shape: {X_pred.shape}")
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

if __name__ == "__main__":
    args = get_argument_parser()
    print("\n" + "="*60)
    print(f"{'Argument':<30} | {'Value'}")
    print("-" * 60)
    for arg, value in vars(args).items(): 
        print(f"{arg:<30} | {value}")
    print("="*60 + "\n")

    # Load Data
    data = pd.read_csv('Dataset/{}.csv'.format(args.dataset), parse_dates=['TimeStamp'])

    # Filter by date
    data = data[data['TimeStamp'] > '2021-04-30'].reset_index(drop=True)
    total_samples = len(data)

    # Count Outliers vs Normal
    n_outliers = int(data['outlier'].sum())
    n_normal = total_samples - n_outliers
    
    print(f"전체 샘플: {total_samples:,}개 (2021-04-30 이후)")
    print(f"Outlier: {n_outliers:,}개 ({n_outliers/total_samples*100:.2f}%)")
    print(f"Normal: {n_normal:,}개 ({n_normal/total_samples*100:.2f}%)")


    set_seed(args.seed)

    # Get Dataloaders to count sequences
    train_loader, _ = get_dataloader(data, args, shuffle=False, flag='train', drop_last=False)
    val_loader = get_dataloader(data, args, shuffle=False, flag='val', drop_last=False)
    test_loader = get_dataloader(data, args, shuffle=False, flag='test', drop_last=False)
    
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    total_seq = n_train + n_val + n_test
    
    print(f"\nTrain: {n_train:,} 시퀀스 ({n_train/total_seq*100:.2f}%)")
    print(f"Validation: {n_val:,} 시퀀스 ({n_val/total_seq*100:.2f}%)")
    print(f"Test: {n_test:,} 시퀀스 ({n_test/total_seq*100:.2f}%)")
    print(f"총 유효 시퀀스: {total_seq:,}개")

    # Use test_loader for the baseline
    data_loader = test_loader

    # Run Baseline
    metrics_df = run_linear_regression_baseline(data_loader.dataset, args)
    
    print("\nAggregated Metrics (Mean/Std):")
    print(metrics_df[['mse', 'rmse', 'mae']].describe())
    