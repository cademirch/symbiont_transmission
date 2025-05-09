import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def get_frequency_bin(freq):
    if freq == 1:
        return "1"
    elif freq == 2:
        return "2"
    elif freq == 3:
        return "3"
    elif freq == 4:
        return "4"
    elif freq == 5:
        return "5"
    elif 6 <= freq <= 10:
        return "6-10"
    else:
        return "11+"


def process_csv_to_dataframe(base_dir="results"):
    """
    Walks through:
      results/sample_size={n}/bottleneck={nb}/replicate_{i}.csv
    and returns a concatenated DataFrame with columns:
      frequency, relative_frequency, frequency_bin,
      sample_size, bottleneck, replicate
    """
    pattern = os.path.join(base_dir, "sample_size=*", "bottleneck=*", "replicate_*.csv")
    all_data = []
    for file_path in glob.glob(pattern):
        if file_path.endswith(".og.csv"):
            continue
        try:
            p = Path(file_path)
            # extract wildcards from the path pieces:
            sample_size = int(p.parts[-3].split("=")[1])
            bottleneck = int(p.parts[-2].split("=")[1])
            replicate = int(p.stem.split("_")[1])

            df = pd.read_csv(file_path)
            df["frequency_bin"] = df["frequency"].apply(get_frequency_bin)
            df["sample_size"] = sample_size
            df["bottleneck"] = bottleneck
            df["replicate"] = replicate

            all_data.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return pd.concat(all_data, ignore_index=True)


def main():
    try:
        # running inside Snakemake
        from snakemake.script import Snakemake  # noqa: F401

        base_dir = "results"
    except ImportError:
        # standalone invocation: expect one CLI arg
        import argparse

        parser = argparse.ArgumentParser(description="Plot AFS boxplots per bottleneck")
        parser.add_argument(
            "results_dir",
            help="Path to top-level results directory (contains sample_size=*/bottleneck=*/replicate_*.csv)",
        )
        args = parser.parse_args()
        base_dir = args.results_dir

    # load and concatenate all CSVs
    df = process_csv_to_dataframe(base_dir)

    # for each bottleneck, dump CSV + boxplot
    for nb in sorted(df["bottleneck"].unique()):
        sub = df[df["bottleneck"] == nb]
        sub.to_csv(f"bottleneck_{nb}.csv", index=False)

        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(
            x="frequency_bin",
            y="count",
            order=["1", "2", "3", "4", "5", "6-10", "11+"],
            hue="sample_size",
            data=sub,
        )
        ax.set_yscale("log")
        ax.set_title(f"Allele counts by sample size\n(bottleneck = {nb})")
        ax.set_xlabel("Frequency bin")
        ax.set_ylabel("Count")
        plt.legend(title="sample size")
        plt.tight_layout()
        plt.savefig(f"seaborn_plot_bottleneck_{nb}.jpg", dpi=300)
        plt.clf()


if __name__ == "__main__":
    main()