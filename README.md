# DILATE: Time Series Modeling with Adaptive Losses


## Prerequisites

- Libraries: matplotlib, numba, numpy, pandas, torch, tslearn, wandb, ruptures
- CUDA environment recommended for GPU acceleration.

## Parameters

Here are the main parameters you can use with the `main.py` script:

- **Dataset**:  
  - `--data synthetic`: uses a synthetic dataset.
  - `--data traffic`: uses the traffic dataset (requires a path with `--path_traffic "/path/to/traffic.txt"`).
  - Default: `synthetic`.

- **Visualization**: 
  - Enable with `--viz`, disable with `--no-viz`.

- **Training**: 
  - Enable with `--train`, disable with `--no-train` (to reuse pre-trained models).

- **Number of Epochs**: 
  - `--n_epochs <value>` (default: `2`).

- **Gamma and Alpha**: 
  - `--gamma`: Coefficient for the **Soft-DTW** loss.
  - `--alpha`: Coefficient for weighting **DILATE** components.

- **Logger**: 
  - Enable logging to **Weights & Biases** with `--logger`.

## Experiments

### 1. Identify the Best Gamma
To test different gamma values for the Soft-DTW loss:  
```bash
python main.py --data synthetic --n_epochs 5 --gamma choice --viz
```
The model will be trained with various gamma values to find the best balance.

---

### 2. Identify the Best Alpha
Once the optimal gamma is identified, use this command to test different alpha values with the DILATE loss:  
```bash
python main.py --data synthetic --n_epochs 5 --gamma 0.01 --alpha choice --viz
```

---

### 3. Compare Models with Fixed Gamma and Alpha
After determining the optimal values, compare the performance of models trained with different metrics:  
```bash
python main.py --data synthetic --n_epochs 5 --gamma 0.01 --alpha 0.2 --viz
```

If models are already trained, disable training to speed up evaluation:  
```bash
python main.py --data synthetic --n_epochs 5 --gamma 0.01 --alpha 0.2 --no-train --viz
```

---

## Results and Visualizations

- Metrics such as **MSE**, **DTW**, **TDI**, **Hausdorff**, and **RAMP** are automatically calculated for each experiment.
- Visualizations can be enabled with `--viz` to examine model predictions and metrics.
