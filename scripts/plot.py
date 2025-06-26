import os
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt


def process_csv_to_dataframe(base_dir="results/"):
    """
    Process CSV files and create a dataframe with columns:
    sample_size, frequency_bin, relative_frequency
    """
    # Define sample sizes
    sample_sizes = [100, 1000, 10000, 100000, 1000000]
    nb = [2, 4, 16, 32, 64, 128]

    # Define function to determine frequency bin
    def get_frequency_bin(freq):
        if freq == 1:
            return "1"
        elif 2 <= freq <= 5:
            return "2-5"
        elif 6 <= freq <= 10:
            return "6-10"
        else:
            return "11+"

    # Initialize list to store all data
    all_data = []

    # Process each sample size
    for sample_size in sample_sizes:
        for b in nb:
            # Get all replicate files for this sample size
            pattern = os.path.join(
                base_dir,
                f"sample_size={str(sample_size)}",
                f"bottleneck={str(b)}",
                "replicate_*.csv",
            )
            files = glob.glob(pattern)
            print(
                f"processing sample_size={sample_size} nb={b}, found {len(list(files))} files"
            )
            # Process each replicate file
            for file_path in files:
                try:
                    # Extract replicate number from filename
                    replicate = int(
                        os.path.basename(file_path).split("_")[1].split(".")[0]
                    )

                    # Read the CSV
                    df = pd.read_csv(file_path)

                    # Add bin and sample size information
                    df["frequency_bin"] = df["frequency"].apply(get_frequency_bin)
                    df["sample_size"] = sample_size
                    df["bottleneck_size"] = b
                    df["replicate"] = replicate

                    # Add to our collection
                    all_data.append(df)

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    result_df = pd.concat(all_data, ignore_index=True)
    return result_df


if __name__ == "__main__":
    # Process data
    # df = process_csv_to_dataframe()
    # df.to_csv("freqs.csv", index=False)
    df = pd.read_csv("freqs.csv")
    g = sns.FacetGrid(df, col="bottleneck_size", row="sample_size", margin_titles=True)
    g.map_dataframe(
        sns.boxplot,
        x="frequency_bin",
        y="relative_frequency",
        order=["1", "2-5", "6-10", "11+"],
    )
    plt.savefig("seaborn_plot.png", dpi=300)
