import torch

def hausdorff_distance(true_batch, predicted_batch):
    diff = true_batch.unsqueeze(2) - predicted_batch.unsqueeze(1)  # Shape: (batch_size, 168, 168, 1)
    distances = torch.norm(diff, dim=-1)  # Shape: (batch_size, 168, 168)

    forward_distances = torch.max(torch.min(distances, dim=2)[0], dim=1)[0]  # Shape: (batch_size,)
    backward_distances = torch.max(torch.min(distances, dim=1)[0], dim=1)[0]  # Shape: (batch_size,)

    hausdorff_distances = torch.max(forward_distances, backward_distances)

    return hausdorff_distances


def swinging_doors(data, epsilon):
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


def mae_m(ramp_approx_1, ramp_approx_2):
    return torch.mean(torch.abs(ramp_approx_1 - ramp_approx_2), dim=(1, 2))


def ramp_score(true_batch, predicted_batch, epsilon):
    ramp_approx_1 = swinging_doors(true_batch, epsilon)
    ramp_approx_2 = swinging_doors(predicted_batch, epsilon)
    return mae_m(ramp_approx_1, ramp_approx_2)