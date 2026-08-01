import pandas as pd

from src.utils.config import METADATA_PATH


def main():

    df = pd.read_csv(METADATA_PATH)

    print("=" * 60)
    print("HAM10000 DATASET REPORT")
    print("=" * 60)

    print(f"\nTotal Images : {len(df)}")

    print("\nDiagnosis Counts")
    print(df["dx"].value_counts())

    print("\nSex Distribution")
    print(df["sex"].value_counts())

    print("\nLocalization")
    print(df["localization"].value_counts())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nAge Statistics")
    print(df["age"].describe())


if __name__ == "__main__":
    main()