import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
df_train = pd.read_csv("aggregated_shared_mutations.csv")
emp = pd.read_csv("empirical_shared_mutations.csv")

# Calculate AFs
df_train["circles_af"] = df_train["circles_alt"] / df_train["circles_depth"]
df_train["duplex_af"] = df_train["duplex_alt"] / df_train["duplex_depth"]
emp["circles_af"] = emp["circles_alt"] / emp["circles_depth"]
emp["duplex_af"] = emp["duplex_alt"] / emp["duplex_depth"]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ===== AF DISTRIBUTIONS =====
# Circles AF
ax = axes[0, 0]
ax.hist(
    df_train["circles_af"], bins=50, alpha=0.6, label="Training", color="blue", log=True
)
ax.hist(
    emp["circles_af"],
    bins=20,
    alpha=0.8,
    label="Empirical",
    color="red",
    edgecolor="black",
)
ax.set_xlabel("Allele Frequency")
ax.set_ylabel("Count (log scale)")
ax.set_title("Circles AF Distribution")
ax.set_xscale("log")
ax.legend()
ax.grid(alpha=0.3)

# Duplex AF
ax = axes[0, 1]
ax.hist(
    df_train["duplex_af"],
    bins=50,
    alpha=0.6,
    label="Training",
    color="orange",
    log=True,
)
ax.hist(
    emp["duplex_af"],
    bins=20,
    alpha=0.8,
    label="Empirical",
    color="red",
    edgecolor="black",
)
ax.set_xlabel("Allele Frequency")
ax.set_ylabel("Count (log scale)")
ax.set_title("Duplex AF Distribution")
ax.set_xscale("log")
ax.legend()
ax.grid(alpha=0.3)

# ===== DEPTH DISTRIBUTIONS =====
# Circles depth
ax = axes[1, 0]
ax.hist(
    df_train["circles_depth"],
    bins=50,
    alpha=0.6,
    label="Training",
    color="blue",
    log=True,
)
ax.hist(
    emp["circles_depth"],
    bins=20,
    alpha=0.8,
    label="Empirical",
    color="red",
    edgecolor="black",
)
ax.set_xlabel("Depth")
ax.set_ylabel("Count (log scale)")
ax.set_title("Circles Depth Distribution")
ax.legend()
ax.grid(alpha=0.3)

# Duplex depth
ax = axes[1, 1]
ax.hist(
    df_train["duplex_depth"],
    bins=50,
    alpha=0.6,
    label="Training",
    color="orange",
    log=True,
)
ax.hist(
    emp["duplex_depth"],
    bins=20,
    alpha=0.8,
    label="Empirical",
    color="red",
    edgecolor="black",
)
ax.set_xlabel("Depth")
ax.set_ylabel("Count (log scale)")
ax.set_title("Duplex Depth Distribution")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("empirical_vs_training_comparison.png", dpi=300, bbox_inches="tight")
print("✓ Saved comparison plot to empirical_vs_training_comparison.png")
plt.show()

# Print summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print("\nCircles AF:")
print(
    f"  Training - Mean: {df_train['circles_af'].mean():.4f}, Median: {df_train['circles_af'].median():.4f}"
)
print(
    f"  Empirical - Mean: {emp['circles_af'].mean():.4f}, Median: {emp['circles_af'].median():.4f}"
)

print("\nDuplex AF:")
print(
    f"  Training - Mean: {df_train['duplex_af'].mean():.4f}, Median: {df_train['duplex_af'].median():.4f}"
)
print(
    f"  Empirical - Mean: {emp['duplex_af'].mean():.4f}, Median: {emp['duplex_af'].median():.4f}"
)

print("\nCircles Depth:")
print(
    f"  Training - Mean: {df_train['circles_depth'].mean():.1f}, Median: {df_train['circles_depth'].median():.1f}"
)
print(
    f"  Empirical - Mean: {emp['circles_depth'].mean():.1f}, Median: {emp['circles_depth'].median():.1f}"
)

print("\nDuplex Depth:")
print(
    f"  Training - Mean: {df_train['duplex_depth'].mean():.1f}, Median: {df_train['duplex_depth'].median():.1f}"
)
print(
    f"  Empirical - Mean: {emp['duplex_depth'].mean():.1f}, Median: {emp['duplex_depth'].median():.1f}"
)
