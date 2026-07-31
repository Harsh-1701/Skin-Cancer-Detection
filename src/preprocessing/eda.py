import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import METADATA_PATH, PLOT_DIR

def main():

    df = pd.read_csv(METADATA_PATH)

    print("=" * 60)
    print("HAM10000 Dataset Summary")
    print("=" * 60)

    print(f"\nTotal Images : {len(df)}")

    print("\nDiagnosis Distribution:\n")

    print(df["dx"].value_counts())

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    df["dx"].value_counts().plot(kind="bar")

    plt.title("Diagnosis Distribution")

    plt.xlabel("Diagnosis")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "diagnosis_distribution.png")

    print("\nPlot saved successfully!")


if __name__ == "__main__":
    main()