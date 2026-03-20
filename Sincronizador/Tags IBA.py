# Databricks notebook source
# DBTITLE 1,Import libraries
import pandas as pd
import glob
import os

# COMMAND ----------

# DBTITLE 1,Read sample CSV file (1.csv)
# Read first CSV to understand structure (trying different encoding and delimiter)
sample_path = '/Workspace/Users/exbrunmn@ex.gerdau.com/sincronizador_extracao_luciano/1.csv'

# Try different combinations of encoding and delimiter
try:
    df_sample = pd.read_csv(sample_path, encoding='latin-1', sep=';')
except:
    try:
        df_sample = pd.read_csv(sample_path, encoding='latin-1', sep=',')
    except:
        df_sample = pd.read_csv(sample_path, encoding='iso-8859-1', sep=';')

print(f"Shape: {df_sample.shape}")
print(f"\nColumns: {list(df_sample.columns)}")
print(f"\nFirst few rows:")
display(df_sample.head())

# COMMAND ----------

# DBTITLE 1,Concatenate all CSV files (1.csv to 29.csv)
# Read and concatenate all CSV files from 1.csv to 29.csv
base_path = '/Workspace/Users/exbrunmn@ex.gerdau.com/sincronizador_extracao_luciano/'

all_dfs = []
for day in range(1, 30):  # 1 to 29
    file_path = f'{base_path}{day}.csv'
    try:
        df_day = pd.read_csv(file_path, encoding='latin-1', sep=';')
        all_dfs.append(df_day)
        print(f"Day {day}: {df_day.shape[0]} rows loaded")
    except Exception as e:
        print(f"Error loading day {day}: {e}")

# Concatenate all dataframes
df_concatenated = pd.concat(all_dfs, ignore_index=True)

print(f"\n{'='*60}")
print(f"Total concatenated shape: {df_concatenated.shape}")
print(f"Total rows: {df_concatenated.shape[0]:,}")
print(f"Total columns: {df_concatenated.shape[1]}")

# COMMAND ----------

# DBTITLE 1,Column information and data types
# Display column information
print("COLUMN INFORMATION")
print("="*80)
print(f"\nTotal columns: {len(df_concatenated.columns)}\n")

for i, col in enumerate(df_concatenated.columns, 1):
    print(f"{i:2d}. {col}")

# COMMAND ----------

# DBTITLE 1,Data types and non-null counts
# Show data types and non-null counts
print("\nDATA TYPES AND NON-NULL COUNTS")
print("="*80)
print(df_concatenated.info())

# COMMAND ----------

# DBTITLE 1,Missing data analysis
# Missing data analysis
print("\nMISSING DATA ANALYSIS")
print("="*80)

missing_data = pd.DataFrame({
    'Column': df_concatenated.columns,
    'Missing_Count': df_concatenated.isnull().sum(),
    'Missing_Percentage': (df_concatenated.isnull().sum() / len(df_concatenated) * 100).round(2),
    'Non_Missing_Count': df_concatenated.notnull().sum(),
    'Data_Type': df_concatenated.dtypes
})

missing_data = missing_data.sort_values('Missing_Count', ascending=False)

print(f"\nTotal rows: {len(df_concatenated):,}")
print(f"\nColumns with missing data:")

missing_cols = missing_data[missing_data['Missing_Count'] > 0]
if len(missing_cols) > 0:
    display(missing_cols)
else:
    print("No missing data found in any column!")

print(f"\nColumns without missing data: {len(missing_data[missing_data['Missing_Count'] == 0])}")

# COMMAND ----------

# DBTITLE 1,Statistical summary (describe)
# Statistical summary for numeric columns
print("\nSTATISTICAL SUMMARY (NUMERIC COLUMNS)")
print("="*80)

# Convert columns to numeric where possible
df_numeric = df_concatenated.copy()
for col in df_numeric.columns:
    if col != 'Time':  # Skip time column
        try:
            df_numeric[col] = pd.to_numeric(df_numeric[col].str.replace(',', '.'), errors='coerce')
        except:
            pass

describe_stats = df_numeric.describe()
display(describe_stats)

# COMMAND ----------

# DBTITLE 1,Summary statistics per column
# Detailed summary per column
print("\nDETAILED SUMMARY PER COLUMN")
print("="*80)

summary_list = []
for col in df_concatenated.columns:
    summary_list.append({
        'Column': col,
        'Total_Values': len(df_concatenated[col]),
        'Non_Null': df_concatenated[col].notnull().sum(),
        'Null': df_concatenated[col].isnull().sum(),
        'Unique_Values': df_concatenated[col].nunique(),
        'Data_Type': str(df_concatenated[col].dtype)
    })

summary_df = pd.DataFrame(summary_list)
display(summary_df)

# COMMAND ----------

# DBTITLE 1,Read and display equivalenciaTags.xlsx
# Read the equivalenciaTags.xlsx file
equivalencia_path = '/Workspace/Users/exbrunmn@ex.gerdau.com/sincronizador_extracao_luciano/equivalenciaTags.xlsx'

print("\nEQUIVALENCIA TAGS FILE")
print("="*80)

try:
    df_equivalencia = pd.read_excel(equivalencia_path)
    
    print(f"\nShape: {df_equivalencia.shape}")
    print(f"Rows: {df_equivalencia.shape[0]}")
    print(f"Columns: {df_equivalencia.shape[1]}")
    print(f"\nColumn names: {list(df_equivalencia.columns)}")
    
    print(f"\n\nFull content of equivalenciaTags.xlsx:")
    print("="*80)
    print(df_equivalencia.to_string())
    
    print(f"\n\nData types:")
    print(df_equivalencia.dtypes)
    
    print(f"\n\nBasic info:")
    print(df_equivalencia.info())
    
except Exception as e:
    print(f"Error reading Excel file: {e}")

# COMMAND ----------

# DBTITLE 1,Create column name mapping from equivalenciaTags
# Create mapping dictionary from equivalenciaTags
print("CREATING COLUMN MAPPING")
print("="*80)

# The first column contains the tag codes, second column contains real names
tag_code_col = df_equivalencia.columns[0]  # [23.149]
real_name_col = df_equivalencia.columns[1]  # ACI@FEA_ELET_CORR1

# Create mapping dictionary
column_mapping = {}

# Add the header row mapping (from column names)
# Add both formats: with dots and with colons
tag_code_with_dots = tag_code_col
tag_code_with_colons = tag_code_col.replace('.', ':')
column_mapping[tag_code_with_dots] = real_name_col
column_mapping[tag_code_with_colons] = real_name_col

# Add mappings from the data rows
for idx, row in df_equivalencia.iterrows():
    tag_code = row[tag_code_col]
    real_name = row[real_name_col]
    if pd.notna(tag_code) and pd.notna(real_name):
        # Add both formats: original with dots and normalized with colons
        tag_code_with_dots = str(tag_code)
        tag_code_with_colons = str(tag_code).replace('.', ':')
        column_mapping[tag_code_with_dots] = real_name
        column_mapping[tag_code_with_colons] = real_name

print(f"\nTotal mappings created: {len(column_mapping)}")
print(f"\nFirst 10 mappings:")
for i, (code, name) in enumerate(list(column_mapping.items())[:10], 1):
    print(f"{i:2d}. {code:15s} -> {name}")

print(f"\n\nChecking match with CSV columns:")
print(f"CSV columns that will be renamed: {sum([1 for col in df_concatenated.columns if col in column_mapping])}")
print(f"CSV columns not in mapping: {sum([1 for col in df_concatenated.columns if col not in column_mapping])}")

# COMMAND ----------

# DBTITLE 1,Rename columns in concatenated dataframe
# Create new dataframe with renamed columns
print("\nRENAMING COLUMNS IN CONCATENATED DATAFRAME")
print("="*80)

# Rename columns using the mapping
df_renamed = df_concatenated.rename(columns=column_mapping)

print(f"\nOriginal columns (first 10):")
for i, col in enumerate(list(df_concatenated.columns)[:10], 1):
    print(f"{i:2d}. {col}")

print(f"\nRenamed columns (first 10):")
for i, col in enumerate(list(df_renamed.columns)[:10], 1):
    print(f"{i:2d}. {col}")

print(f"\n\nColumns successfully renamed: {sum([1 for col in df_concatenated.columns if col in column_mapping])}")
print(f"Columns not found in mapping: {sum([1 for col in df_concatenated.columns if col not in column_mapping])}")

# COMMAND ----------

# DBTITLE 1,Display renamed dataframe info
# Display information about the renamed dataframe
print("\nRENAMED DATAFRAME INFORMATION")
print("="*80)

print(f"\nShape: {df_renamed.shape}")
print(f"Total rows: {df_renamed.shape[0]:,}")
print(f"Total columns: {df_renamed.shape[1]}")

print(f"\n\nAll column names in renamed dataframe:")
print("="*80)
for i, col in enumerate(df_renamed.columns, 1):
    print(f"{i:2d}. {col}")

print(f"\n\nFirst few rows of renamed dataframe:")
display(df_renamed.head())

# COMMAND ----------

# DBTITLE 1,Save renamed dataframe to Unity Catalog table
import re

# Function to clean column names for Delta tables
def clean_column_name(col_name):
    """
    Clean column names to be compatible with Delta tables.
    Delta allows: letters, numbers, underscores, and @ symbol
    Invalid characters: space, comma, semicolon, braces, parentheses, newline, tab, equals
    """
    # Replace invalid characters with underscore
    # Keep @ symbol as it's allowed in Delta
    cleaned = re.sub(r'[\s,;{}()\n\t=\\]', '_', col_name)
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    return cleaned

# Create a copy of the dataframe with cleaned column names
print("Cleaning column names for Delta compatibility...")
df_for_save = df_renamed.copy()

# Show original vs cleaned names
print("\nColumn name changes:")
print("="*80)
column_name_mapping = {}
for col in df_for_save.columns:
    cleaned_col = clean_column_name(col)
    if col != cleaned_col:
        print(f"{col:50s} -> {cleaned_col}")
        column_name_mapping[col] = cleaned_col
    else:
        column_name_mapping[col] = col

# Rename columns
df_for_save.columns = [clean_column_name(col) for col in df_for_save.columns]

print(f"\n\nCleaned column names (first 10):")
for i, col in enumerate(list(df_for_save.columns)[:10], 1):
    print(f"{i:2d}. {col}")

# Convert pandas dataframe to Spark dataframe
print("\n\nConverting pandas DataFrame to Spark DataFrame...")
spark_df = spark.createDataFrame(df_for_save)

# Define the target table name
table_name = "industrial_iatemperaturascha_refined_dev.computed_features.hist_tags_IBA_novembro"

print(f"\nPreparing to save to Unity Catalog table: {table_name}")
print(f"Total rows to save: {df_for_save.shape[0]:,}")
print(f"Total columns: {df_for_save.shape[1]}")
print(f"\nMode: overwrite (will replace existing data if table exists)")

# Check if table exists
try:
    existing_table = spark.sql(f"DESCRIBE TABLE {table_name}")
    print(f"\n⚠️  WARNING: Table {table_name} already exists and will be overwritten!")
except:
    print(f"\n✓ Table does not exist yet, will create new table.")

# Save to Unity Catalog
print(f"\nSaving data...")
spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

print(f"\n✓ Successfully saved to {table_name}")
print(f"\nYou can now query the table using:")
print(f"SELECT * FROM {table_name} LIMIT 10")

# COMMAND ----------

# DBTITLE 1,Read table from Unity Catalog
# Read the saved table from Unity Catalog
table_name = "industrial_iatemperaturascha_refined_dev.computed_features.hist_tags_IBA_novembro"

print(f"Reading table: {table_name}")
print("="*80)

# Read as Spark DataFrame
df_spark = spark.table(table_name)

# Convert to pandas for easier analysis
df_analysis = df_spark.toPandas()

print(f"\nTable loaded successfully!")
print(f"Shape: {df_analysis.shape}")
print(f"Total rows: {df_analysis.shape[0]:,}")
print(f"Total columns: {df_analysis.shape[1]}")

# COMMAND ----------

# DBTITLE 1,Show all column names
# Display all column names
print("\nALL COLUMN NAMES")
print("="*80)

for i, col in enumerate(df_analysis.columns, 1):
    print(f"{i:2d}. {col}")

# COMMAND ----------

# DBTITLE 1,Convert columns to numeric and parse time
# Convert string columns to numeric
print("Converting columns to numeric...")
print("="*80)

df_numeric = df_analysis.copy()

# Parse Time column to datetime
df_numeric['Time'] = pd.to_datetime(df_numeric['Time'], format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')

# Convert all other columns to numeric (replacing comma with dot for decimal)
for col in df_numeric.columns:
    if col != 'Time':
        try:
            df_numeric[col] = pd.to_numeric(df_numeric[col].str.replace(',', '.'), errors='coerce')
        except:
            pass

print(f"\n✓ Conversion completed!")
print(f"\nData types after conversion:")
print(df_numeric.dtypes)

# COMMAND ----------

# DBTITLE 1,Statistical summary (describe)
# Show describe statistics for all numeric columns
print("\nSTATISTICAL SUMMARY (DESCRIBE)")
print("="*80)

# Get describe for numeric columns only
describe_stats = df_numeric.describe()

print(f"\nStatistics for {describe_stats.shape[1]} numeric variables:")
print(f"\n")
display(describe_stats)

# COMMAND ----------

# DBTITLE 1,Create histograms for all variables
import matplotlib.pyplot as plt
import numpy as np

# Get numeric columns (excluding Time)
numeric_cols = [col for col in df_numeric.columns if col != 'Time']

print(f"Creating histograms for {len(numeric_cols)} variables...")
print("="*80)

# Create subplots - 5 columns per row
n_cols = 5
n_rows = int(np.ceil(len(numeric_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    ax = axes[idx]
    data = df_numeric[col].dropna()
    
    if len(data) > 0:
        ax.hist(data, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(col, fontsize=8)
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Frequency', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(numeric_cols), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.suptitle('Histograms of All Variables', fontsize=16, y=1.001)
plt.show()

print(f"\n✓ Histograms created successfully!")

# COMMAND ----------

# DBTITLE 1,Filter data for one day
# Get unique dates
df_numeric['Date'] = df_numeric['Time'].dt.date
unique_dates = df_numeric['Date'].unique()

print(f"Available dates in dataset:")
print("="*80)
for i, date in enumerate(sorted(unique_dates), 1):
    count = len(df_numeric[df_numeric['Date'] == date])
    print(f"{i:2d}. {date} - {count:,} records")

# Select first day for plotting
first_day = sorted(unique_dates)[0]
df_one_day = df_numeric[df_numeric['Date'] == first_day].copy()
df_one_day = df_one_day.sort_values('Time')

print(f"\n\nSelected day for time series plots: {first_day}")
print(f"Total records: {len(df_one_day):,}")

# COMMAND ----------

# DBTITLE 1,Plot time series for all variables (one day)
# Plot all variables for one day
print(f"\nCreating time series plots for {first_day}...")
print("="*80)

# Create subplots - 3 columns per row for better visibility
n_cols = 3
n_rows = int(np.ceil(len(numeric_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    ax = axes[idx]
    
    # Plot time series
    ax.plot(df_one_day['Time'], df_one_day[col], linewidth=0.8, alpha=0.8)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.set_xlabel('Time', fontsize=7)
    ax.set_ylabel('Value', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Hide unused subplots
for idx in range(len(numeric_cols), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.suptitle(f'Time Series - All Variables ({first_day})', fontsize=16, y=1.001)
plt.show()

print(f"\n✓ Time series plots created successfully!")

# COMMAND ----------

display(df_numeric.head(10))