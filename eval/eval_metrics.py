import torch
import numpy as np
import ruptures as rpt
from ruptures.metrics import hausdorff

def calculate_hausdorff_distances(ts_0, ts_1, bkp_1):
    ts_0 = ts_0.numpy()
    ts_1 = ts_1.numpy()
    bkp_1 = bkp_1.numpy()
    ts_concat = np.concatenate((ts_0, ts_1), axis=1)
    batch_size, sequence_length, _ = ts_concat.shape
    hausdorff_distances = []

    for i in range(batch_size):
        signal = ts_concat[i, :, 0]  
        true_bkp = np.array([bkp_1[i],sequence_length]) 
        algo = rpt.Dynp(model="l2").fit(signal)
        predicted_bkp = np.array(algo.predict(n_bkps=1))
        hausdorff_dist = hausdorff(true_bkp, predicted_bkp)
        hausdorff_distances.append(hausdorff_dist)

    return np.array(hausdorff_distances).mean()


def swinging_doors_batch(data, epsilon):
    batch_size, seq_len, _ = data.shape
    SD_ts = torch.zeros_like(data)
    lower_slope = torch.full((batch_size,), float('inf'), device=data.device)
    upper_slope = torch.full((batch_size,), float('-inf'), device=data.device)
    start_idx = torch.zeros(batch_size, dtype=torch.long, device=data.device)

    for i in range(1, seq_len):
        current_lower_slope = (data[:, i, 0] - (data[torch.arange(batch_size), start_idx, 0] - epsilon)) / (i - start_idx)
        current_upper_slope = (data[:, i, 0] - (data[torch.arange(batch_size), start_idx, 0] + epsilon)) / (i - start_idx)

        lower_slope = torch.min(lower_slope, current_lower_slope)
        upper_slope = torch.max(upper_slope, current_upper_slope)

        condition = (lower_slope < upper_slope) | (i == seq_len - 1)
        if condition.any():
            current_slope = (data[:, i, 0] - data[torch.arange(batch_size), start_idx, 0]) / (i - start_idx)
            for batch_idx in torch.where(condition)[0]:
                indices = torch.arange(i - start_idx[batch_idx], device=data.device)
                SD_ts[batch_idx, start_idx[batch_idx]:i, 0] = current_slope[batch_idx] * indices + data[batch_idx, start_idx[batch_idx], 0]

            start_idx[condition] = i
            lower_slope[condition] = float('inf')
            upper_slope[condition] = float('-inf')

    SD_ts[:, -1, 0] = data[:, -1, 0]
    return SD_ts


def mae_batch(ramp_approx_1, ramp_approx_2):
    return torch.mean(torch.abs(ramp_approx_1 - ramp_approx_2), dim=(1, 2))


def ramp_score_batch(true_batch, predicted_batch, epsilon):
    ramp_approx_1 = swinging_doors_batch(true_batch, epsilon)
    ramp_approx_2 = swinging_doors_batch(predicted_batch, epsilon)
    return torch.mean(mae_batch(ramp_approx_1, ramp_approx_2))