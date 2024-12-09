import numpy as np
from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(true_series, predicted_series):
    A = np.array(true_series).reshape(-1, 1)
    B = np.array(predicted_series).reshape(-1, 1)
    
    forward_distance = directed_hausdorff(A, B)[0]
    backward_distance = directed_hausdorff(B, A)[0]
    
    return max(forward_distance, backward_distance)


def swinging_doors(data, epsilon):
    SD_ts = np.zeros_like(data)
    lower_slope = +np.inf
    upper_slope = -np.inf
    start_idx = 0

    for i in range(1, len(data)):
        lower_slope = min(lower_slope, (data[i] - (data[start_idx] - epsilon)) / (i - start_idx))
        upper_slope = max(upper_slope, (data[i] - (data[start_idx] + epsilon)) / (i - start_idx))

        if lower_slope < upper_slope or i==len(data)-1:
            current_slope = (data[i] - data[start_idx]) / (i - start_idx)
            SD_ts[start_idx:i] = np.array([current_slope*j+data[start_idx] for j in range(i-start_idx)])
            start_idx = i      
            lower_slope = +np.inf
            upper_slope = -np.inf
    SD_ts[-1] = data[-1]
    return SD_ts


def mae(ramp_approx_1, ramp_approx_2):
    return np.mean(np.abs(ramp_approx_1-ramp_approx_2))


def ramp_score(true_series, predicted_series, epsilon):
    ramp_approx_1 = swinging_doors(true_series, epsilon)
    ramp_approx_2 = swinging_doors(predicted_series, epsilon)
    print(ramp_approx_1, ramp_approx_2, "\n")
    print(mae(ramp_approx_1, ramp_approx_2), "\n")
    return mae(ramp_approx_1, ramp_approx_2)