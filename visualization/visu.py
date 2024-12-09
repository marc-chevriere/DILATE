import matplotlib.pyplot as plt

def plot_preds(X_train, y_train, preds):

    plt.figure(figsize=(24, 12))

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
    plt.show()