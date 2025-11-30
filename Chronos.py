import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="Fine-tuning support is experimental")

# Use only 1 GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pandas as pd
import numpy as np
from math import sqrt
from chronos import BaseChronosPipeline, Chronos2Pipeline
from transformers import logging

# Suppress transformers logging
logging.set_verbosity_error()


class ChronosForecaster:
    def __init__(self, args):
        """
        Chronos forecaster wrapping BaseChronosPipeline.

        Args:
            args: argparse.Namespace with at least seq_len, pred_len, target.
        """
        self.args = args
        self.covariate_list = None

        self.pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2",
            device_map="auto",
            dtype=torch.bfloat16,
        )

    def select_covariates(self, dataset):
        """
        Select covariates based on average correlation with target across training sequences.
        """
        if not self.args.use_covariates:
            self.covariate_list = []
            return

        print("Selecting covariates based on Training set correlation (per-sequence average)...")
        
        df = dataset.data
        target_col = self.args.target
        candidates = [c for c in df.columns if c not in ['TimeStamp', 'outlier', target_col]]
        
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        total_len = seq_len + pred_len
        
        corrs = []
        
        # Iterate over training sequences
        for end in dataset.using_index_list:
            start = end - total_len
            if start < 0: continue
            
            window = df.iloc[start:end]
            if "outlier" in window.columns and window["outlier"].any():
                continue
            
            # Calculate correlation of candidates with target for this window
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                c = window[candidates].corrwith(window[target_col], numeric_only=True)
            c = c.fillna(0) # Replace NaN correlations (due to constant values) with 0
            corrs.append(c)
            
        if not corrs:
            print("No valid windows found for correlation calculation.")
            self.covariate_list = []
            return

        # Average correlation across all windows
        avg_corr = pd.concat(corrs, axis=1).mean(axis=1).sort_values(ascending=False)
        
        self.covariate_list = [c for c in candidates if abs(avg_corr.get(c, 0)) >= self.args.cov_corr_threshold]
        

    def fine_tune(self, data_loader):
        """
        Fine-tune the Chronos model on the training data.
        """
        train_dataset = data_loader['train'].dataset
        val_dataset = data_loader['val'].dataset
        
        self.select_covariates(train_dataset)

        print("Preparing data for fine-tuning...")
        train_inputs = []
        val_inputs = []
        
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        total_len = seq_len + pred_len
        target_col = self.args.target
        
        df = train_dataset.data
        
        for end in train_dataset.using_index_list:
            start = end - total_len
            if start < 0: continue
            
            window = df.iloc[start:end]
            if "outlier" in window.columns and window["outlier"].any():
                continue
                
            # Construct input dict
            item = {
                "target": window[target_col].values,
            }
            
            if self.args.use_covariates and self.covariate_list:
                item["past_covariates"] = {
                    c: window[c].values for c in self.covariate_list
                }
                # Future values of covariates are not used during training.
                # However, we need to include their names to indicate that these columns will be available at prediction time
                item["future_covariates"] = {c: None for c in self.covariate_list}
            
            train_inputs.append(item)
        
        for end in val_dataset.using_index_list:
            start = end - total_len
            if start < 0: continue
            
            window = df.iloc[start:end]
            if "outlier" in window.columns and window["outlier"].any():
                continue
                
            # Construct input dict
            item = {
                "target": window[target_col].values,
            }
            
            if self.args.use_covariates and self.covariate_list:
                item["past_covariates"] = {
                    c: window[c].values for c in self.covariate_list
                }
                # Future values of covariates are not used during training.
                # However, we need to include their names to indicate that these columns will be available at prediction time
                item["future_covariates"] = {c: None for c in self.covariate_list}
            
            val_inputs.append(item)
            
        print(f"Fine-tuning on {len(train_inputs)} samples...")
        print(f"Validating on {len(val_inputs)} samples...")
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Use a smaller batch size for fine-tuning to avoid OOM/NVML errors
        # args.batch_size is typically 128, which is too large for fine-tuning
        ft_batch_size = min(self.args.batch_size, 8) 
        
        """
        # 파인튜닝 대상 모듈 확인용 코드
        print("=== 학습(업데이트) 대상 파라미터 목록 ===")
        for name, param in self.pipeline.model.named_parameters():
            if param.requires_grad:
                print(f"Training: {name}")
            else:
                print(f"Frozen: {name}")
        """
        print(f"Starting fine-tuning with batch_size={ft_batch_size}...")

        self.pipeline = self.pipeline.fit(
            inputs=train_inputs,
            validation_inputs=val_inputs,
            prediction_length=pred_len,
            num_steps=self.args.num_steps, # As requested in snippet
            learning_rate=self.args.learning_rate,
            batch_size=ft_batch_size,
            logging_steps=100
        )
        print("Fine-tuning completed.")

    def continual_pretrain(self, manufacturing_datasets):
        """
        Continual pretraining on additional manufacturing datasets.
        manufacturing_datasets: List of numpy arrays (T, F)
        """
        print("Preparing data for continual pretraining...")
        
        ft_batch_size = min(self.args.batch_size, 8)
        
        for idx, data in enumerate(manufacturing_datasets):
            print(f"Processing dataset {idx + 1}/{len(manufacturing_datasets)} for pretraining...")
            train_inputs = []
            
            # data is (T, F)
            # Treat each feature as a separate univariate time series
            for i in range(data.shape[1]):
                series = data[:, i]
                # Chronos expects 1D array for target
                train_inputs.append({"target": series})
                
            print(f"Dataset {idx + 1}: {len(train_inputs)} time series")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"Starting pretraining step {idx + 1} with batch_size={ft_batch_size}...")
            
            # We use the same fit method but on this new data
            self.pipeline = self.pipeline.fit(
                inputs=train_inputs,
                prediction_length=self.args.pred_len,
                num_steps=self.args.num_steps, 
                learning_rate=self.args.learning_rate,
                batch_size=ft_batch_size,
                logging_steps=500,
                save_strategy="no"
            )
        
        print("Continual pretraining completed.")
    
    def run(self, data):
        """
        Run the full Chronos forecasting experiment:
        1. Select covariates (if enabled)
        2. Predict on test set
        3. Evaluate and print metrics
        """
        if self.covariate_list == None:
            self.select_covariates(data['train'].dataset)

        print(f"Selected {len(self.covariate_list)} covariates: {self.covariate_list}")

        pred_df, y_true_df = self.predict_from_dataloader(data['test'])
        
        # Get scaler for inverse transform
        scaler = None
        if hasattr(data['test'].dataset, 'y_scaler'):
            scaler = data['test'].dataset.y_scaler
        
        # --- Apply Inverse Transform BEFORE evaluation ---
        if scaler is not None:
            #print("Applying Inverse Transform to predictions and targets...")
            # Inverse transform y_true
            if self.args.target in y_true_df.columns:
                y_true_df[self.args.target] = scaler.inverse_transform(y_true_df[[self.args.target]].values).flatten()
            
            # Inverse transform predictions (all quantile columns)
            # Filter only numeric columns and exclude id/TimeStamp/Target
            pred_cols = [c for c in pred_df.select_dtypes(include=[np.number]).columns 
                         if c not in ['id', 'TimeStamp', self.args.target]]
            
            for col in pred_cols:
                 pred_df[col] = scaler.inverse_transform(pred_df[[col]].values).flatten()
        # -------------------------------------------------

        metrics_df = self.evaluate(pred_df, y_true_df)
        
        return metrics_df, pred_df, y_true_df

    def evaluate(self, pred_df, y_true_df):
        # Merge to present a comparison table (id, timestamp, y_true, prediction)
        pred = pred_df.copy()
        pred['TimeStamp'] = pd.to_datetime(pred['TimeStamp'])
        y_true = y_true_df.copy()
        y_true['TimeStamp'] = pd.to_datetime(y_true['TimeStamp'])

        merged = pred.merge(
            y_true,
            on=['id', 'TimeStamp'],
            how='inner',
            suffixes=('_pred', '_true'),
        )

        # choose prediction column (median / 0.5)
        pred_col = None
        for c in ('0.5', 'median', 'prediction', 'pred'):
            if c in merged.columns:
                pred_col = c
                break
        if pred_col is None:
            num_cols = [c for c in merged.select_dtypes(include=[np.number]).columns if c not in ['id']]
            pred_col = num_cols[0]

        true_col = None
        if self.args.target in merged.columns:
            true_col = self.args.target
        else:
            true_cols = [c for c in merged.columns if c.endswith('_true')]
            if true_cols:
                true_col = true_cols[0]

        if true_col is None:
            raise RuntimeError('Could not find true target column in merged Chronos output')

        result_table = merged[['id', 'TimeStamp', true_col, pred_col]].rename(
            columns={true_col: 'y_true', pred_col: 'y_pred'}
        )


        # --- Per-sequence metrics (MSE / RMSE / MAE) across all sequences in data_loader['test'] ---
        metrics = []
        for sid, g in result_table.groupby('id'):
            a = g['y_pred'].values
            b = g['y_true'].values
            err = a - b
            mse = float((err ** 2).mean())
            rmse = float(sqrt(mse))
            mae = float(np.mean(np.abs(err)))
            metrics.append({'id': sid, 'mse': mse, 'rmse': rmse, 'mae': mae, 'n_points': len(g)})

        metrics_df = pd.DataFrame(metrics)
        print(metrics_df.describe()[['mse', 'rmse', 'mae']])
        
        return metrics_df

    def _build_zero_shot_dfs(self, dataset):
        """
        Build context/future/y_true DataFrames from Dataset_Custom.

        dataset: Dataset_Custom (from Dataset.custom_dataset).
        Returns (context_df, future_df, y_true_df).
        """
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        target_col = self.args.target

        df = dataset.data.copy().reset_index(drop=True)
        
        # --- Apply Scaling if available ---
        if hasattr(dataset, 'scale') and dataset.scale and hasattr(dataset, 'y_scaler'):
            # Scale Target
            target_data = df[[target_col]].values
            df[target_col] = dataset.y_scaler.transform(target_data).flatten()
            
            # Scale Covariates
            if hasattr(dataset, 'x_scaler'):
                # Dataset_Custom fits x_scaler on all columns except TimeStamp and target.
                # This includes 'outlier' if it exists. We must include it for transform to work.
                scaler_cols = [c for c in df.columns if c not in ['TimeStamp', target_col]]
                
                try:
                    # Transform all columns including outlier
                    scaled_values = dataset.x_scaler.transform(df[scaler_cols].values)
                    
                    # Create temporary dataframe to map back to columns
                    scaled_df = pd.DataFrame(scaled_values, columns=scaler_cols, index=df.index)
                    
                    # Update df with scaled values, BUT skip 'outlier' to preserve 0/1 flags for filtering
                    update_cols = [c for c in scaler_cols if c != 'outlier']
                    df[update_cols] = scaled_df[update_cols]
                    
                except Exception as e:
                    print(f"Warning: Could not scale features: {e}")
        # ----------------------------------

        time_col = "TimeStamp"

        # Determine covariates
        if self.args.use_covariates:
            if self.covariate_list is not None:
                # Use explicitly provided list (safe from leakage if computed on train)
                covariate_cols = [c for c in self.covariate_list if c in df.columns]
            else:
                # Fallback: Compute correlation (WARNING: Potential Leakage if df is full dataset)
                # Ideally, this should be avoided by passing covariate_list.
                all_candidates = [c for c in df.columns if c not in [time_col, "outlier", target_col]]
                corr_series = df[all_candidates + [target_col]].corr(numeric_only=True)[target_col]
                covariate_cols = [
                    c for c in all_candidates
                    if abs(corr_series.get(c, 0.0)) >= self.args.corr_threshold
                ]
        else:
            covariate_cols = []
        total_len = seq_len + pred_len
        context_parts = []
        future_parts = []
        y_true_parts = []

        for window_id, end in enumerate(dataset.using_index_list):
            start = end - total_len
            if start < 0:
                continue

            window = df.iloc[start:end]
            if "outlier" in window.columns and window["outlier"].any():
                continue

            context = window.iloc[:seq_len].copy()
            future = window.iloc[seq_len:].copy()

            context["id"] = window_id
            future["id"] = window_id

            ctx_cols = ["id", time_col]
            if target_col in context.columns:
                ctx_cols.append(target_col)
            ctx_cols += covariate_cols
            ctx = context[ctx_cols]
            context_parts.append(ctx)

            fut_cols = ["id", time_col] + covariate_cols
            fut = future[fut_cols]
            future_parts.append(fut)

            if target_col in future.columns:
                y_true = future[["id", time_col, target_col]]
            else:
                y_true = future[["id", time_col]]
            y_true_parts.append(y_true)

        if not context_parts:
            raise RuntimeError("No valid windows found in dataset for ChronosForecaster")

        context_df = pd.concat(context_parts, ignore_index=True)
        future_df = pd.concat(future_parts, ignore_index=True)
    
        y_true_df = pd.concat(y_true_parts, ignore_index=True)

        return context_df, future_df, y_true_df

    def predict_from_dataloader(self, test_loader):
        """
        Run Chronos predictions given data_loader['test'].

        Returns pred_df, y_true_df for downstream evaluation.
        """
        dataset = test_loader.dataset
        context_df, future_df, y_true_df = self._build_zero_shot_dfs(dataset)

        # Clear GPU cache to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Process in chunks to avoid OOM
        unique_ids = context_df['id'].unique()
        chunk_size = 100  # Number of time series to process at once
        pred_dfs = []

        for i in range(0, len(unique_ids), chunk_size):
            batch_ids = unique_ids[i:i + chunk_size]
            batch_context = context_df[context_df['id'].isin(batch_ids)].copy()
            
            # Call predict_df with a reasonable batch_size for the internal dataloader
            batch_pred = self.pipeline.predict_df(
                batch_context,
                future_df=None, 
                prediction_length=self.args.pred_len,
                quantile_levels=[0.1, 0.5, 0.9],
                predict_batches_jointly=self.args.use_cross_learning,
                id_column="id",
                timestamp_column="TimeStamp",
                target=self.args.target,
                batch_size=16 # Explicitly set batch size to control memory usage
            )
            pred_dfs.append(batch_pred)
            
            # Clear cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pred_df = pd.concat(pred_dfs, ignore_index=True)

        return pred_df, y_true_df


__all__ = ["ChronosForecaster"]
