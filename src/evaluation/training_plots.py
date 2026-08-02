from pathlib import Path

Path("outputs/plots").mkdir(parents=True, exist_ok=True)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/logs/training_log.csv")

# Loss Plot
plt.figure(figsize=(8,5))

plt.plot(df["Epoch"], df["Train Loss"], marker="o", label="Train Loss")
plt.plot(df["Epoch"], df["Validation Loss"], marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("outputs/plots/loss_curve.png", dpi=300)

plt.close()


# Accuracy Plot
plt.figure(figsize=(8,5))

plt.plot(df["Epoch"], df["Train Accuracy"], marker="o", label="Train Accuracy")
plt.plot(df["Epoch"], df["Validation Accuracy"], marker="o", label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("outputs/plots/accuracy_curve.png", dpi=300)

plt.close()

print("Plots saved successfully.")