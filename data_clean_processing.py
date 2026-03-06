import pandas as pd

data_path = "PJMW_hourly.csv"

print("Loading and mapping data...")
# 1. Load and format Datetime
df = pd.read_csv(data_path, usecols=['Datetime', 'PJMW_MW'])
df['Datetime'] = pd.to_datetime(df['Datetime'])

# 2. Sort chronologically and drop exact DST duplicates
df = df.sort_values(by='Datetime')
df = df.drop_duplicates(subset=['Datetime'], keep='first')

# 3. Set the index so Pandas understands the flow of time
df.set_index('Datetime', inplace=True)

initial_rows = len(df)
initial_nans = df['PJMW_MW'].isna().sum()

# 4. Expose the "hidden" missing gaps by forcing a strict 1-hour frequency
df_resampled = df.resample('h').asfreq()

new_rows = len(df_resampled)
hidden_gaps_found = new_rows - initial_rows
total_missing_values = df_resampled['PJMW_MW'].isna().sum()

# 5. Interpolate to fill the gaps with straight mathematical lines
df_resampled['PJMW_MW'] = df_resampled['PJMW_MW'].interpolate(method='linear')

# ==========================================
# 6. ADVANCED FEATURE: The T-168 Anchor
# ==========================================
print("Generating 1-week historical lag feature...")
df_resampled['MW_Lag_168'] = df_resampled['PJMW_MW'].shift(168)

# Shifting creates 168 blank rows (NaN) at the very start of our dataset 
# because the first week of data has no "previous week" to look at. We must drop them.
df_resampled = df_resampled.dropna()

# 7. Reset the index and save
df_clean = df_resampled.reset_index()
df_clean.to_csv("fully_cleaned_features.csv", index=False)

# 8. The Final Report
print("\n--- CLEANING & FEATURE REPORT ---")
print(f"(NaNs) found: {initial_nans}")
print(f"Hidden missing hours (gaps) found and created: {hidden_gaps_found}")
print(f"Values filled with interpolation: {total_missing_values}")
print("Saved 'fully_cleaned_features.csv'.")