import pandas as pd

pd.set_option("mode.copy_on_write", True)

jobpostDF = pd.read_parquet("data/jobpostDF.parquet")

# Pre processing
### Time related
jobpostDF["converted_time"] = pd.to_datetime(
    jobpostDF["original_listed_time"], unit="ms"
)
jobpostDF["time_period"] = jobpostDF["converted_time"].dt.to_period("M")

### Location related
jobpostDF = jobpostDF.dropna(subset=["fips"])  # Drop rows where fips is NaN
jobpostDF["fips"] = jobpostDF["fips"].astype(int)
jobpostDF["fips"] = jobpostDF["fips"].astype(str).str.zfill(5)
jobpostDF["state_fips"] = jobpostDF["fips"].str[:2]  # Extract state FIPS


print(jobpostDF.columns)  # Rows with NaN values


# Columns to keep
columns_to_keep = [
    "matched_skills",
    "fips",
    "converted_time",
    "time_period",
    "state_fips",
    "normalized_salary",
]

# Subset the DataFrame
jobpostDF_subset = jobpostDF.loc[:, columns_to_keep]
jobpostDF_subset.to_parquet("data/jobpostDF_subset.parquet", index=False)
