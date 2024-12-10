import matplotlib.pyplot as plt
import os

def plot_preds(X_train, y_train, preds, save_path="figures", file_name="plot.png"):
    os.makedirs(save_path, exist_ok=True)
    
    save_file = os.path.join(save_path, file_name)

    plt.figure(figsize=(12, 6))

    plt.plot(range(len(X_train)), X_train, label="Past values", color="blue")
    plt.plot(range(len(X_train), len(X_train) + len(y_train)), y_train, label="Actual values", color="green")

    for name, predictions in preds.items():
        plt.plot(
            range(len(X_train), 
                  len(X_train) + len(predictions)), 
                  predictions, 
                  label=f"Predicted values ({name})", 
                  linestyle="--",
                  )
        
    plt.axvline(x=len(X_train), color="black", linestyle=":", label="Start of predictions")
    plt.title("Time Series Example (Train, Actual, Predictions)")
    plt.xlabel("Time (indices)")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_file, bbox_inches="tight")
    print(f"Figure saved to {save_file}")

    plt.show()