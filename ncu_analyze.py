import pandas as pd

def load_ncu_csv(path):
    # Nsight CSVs have metadata rows → skip until actual table
    df = pd.read_csv(path, skiprows=0)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Keep only useful columns (varies by version)
    # Try to standardize
    possible_cols = ["Metric Name", "Metric Value", "Kernel Name"]
    cols = [c for c in possible_cols if c in df.columns]

    return df[cols]


def pivot_metrics(df):
    """
    Convert long format → wide format:
    Metric Name → column
    """
    if "Kernel Name" in df.columns:
        # Aggregate across kernels (sum or mean depending on metric)
        grouped = df.groupby("Metric Name")["Metric Value"].mean()
    else:
        grouped = df.set_index("Metric Name")["Metric Value"]

    return grouped


def compare_reports(df1, df2, name1="lstsq", name2="normal"):
    merged = pd.concat([df1, df2], axis=1)
    merged.columns = [name1, name2]

    # Convert to numeric where possible
    merged = merged.apply(pd.to_numeric, errors='coerce')

    # Drop NaNs (metrics not shared)
    merged = merged.dropna()

    # Compute differences
    merged["abs_diff"] = merged[name1] - merged[name2]
    merged["pct_diff"] = (merged["abs_diff"] / merged[name2]) * 100

    # Sort by absolute % difference
    merged = merged.sort_values(by="pct_diff", key=abs, ascending=False)

    return merged


def main():
    # Load CSVs
    df_lstsq = load_ncu_csv("lstsq.csv")
    df_normal = load_ncu_csv("normal.csv")

    # Pivot
    metrics_lstsq = pivot_metrics(df_lstsq)
    metrics_normal = pivot_metrics(df_normal)

    # Compare
    comparison = compare_reports(metrics_lstsq, metrics_normal)

    # Show top differences
    print("\n🔥 Top Metric Differences:\n")
    print(comparison.head(20))

    # Save full report
    comparison.to_csv("comparison.csv")
    print("\nSaved full comparison → comparison.csv")


if __name__ == "__main__":
    main()