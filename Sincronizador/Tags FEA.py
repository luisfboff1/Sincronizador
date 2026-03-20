# Databricks notebook source
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import glob
import os

# COMMAND ----------

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

# Display all column names
print("\nALL COLUMN NAMES")
print("="*80)

for i, col in enumerate(df_analysis.columns, 1):
    print(f"{i:2d}. {col}")

# COMMAND ----------

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

def convert_datetime(df, date_col, hour_col, new_col):
    return (
        df.withColumn(
            "date_clean",
            F.to_date(F.col(date_col))   # extrai YYYY-MM-dd
        )
        .withColumn(
            "hour_clean",
            F.when(
                F.length(F.col(hour_col)) == 4,   # caso HHmm (ex: 0830)
                F.concat_ws(
                    ":",
                    F.col(hour_col).substr(1, 2),
                    F.col(hour_col).substr(3, 2)
                )
            ).otherwise(
                F.col(hour_col)                   # caso HH:mm (ex: 09:28)
            )
        )
        .withColumn(
            new_col,
            F.to_timestamp(
                F.concat_ws(" ", F.col("date_clean"), F.col("hour_clean")),
                "yyyy-MM-dd HH:mm"
            )
        )
        .drop("date_clean", "hour_clean")
    )


# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ======================================================
# 1. LEITURA DAS TABELAS
# ======================================================
fea_df = spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_corridas")
fp_df  = spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida")
vd_df  = spark.table("industrial_iatemperaturascha_refined_prd.computed_features.tb_corridatempos")
lc_df  = spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento")

# ======================================================
# 2. TRATAMENTO FEA (corrigido conforme você avisou)
# ======================================================
# HRVAZAMENTO já vem como HH:mm
# DATA vem como timestamp ("YYYY-MM-dd 00:00:00")

# Monta timestamp de vazamento no FEA
fea_df = fea_df.withColumn(
    "hr_vaz_fea",
    F.to_timestamp(
        F.concat_ws(" ",
            F.to_date("DATA"),
            F.col("HRVAZAMENTO")
        ),
        "yyyy-MM-dd HH:mm"
    )
)

# Converte TTT de string "58.0000000000" → double → segundos
fea_df = fea_df.withColumn(
    "TTT_min",
    F.col("TTT").cast("double")
).withColumn(
    "TTT_seconds",
    (F.col("TTT_min") * 60).cast("long")
)

# final_fea = hr_vaz_fea
# inicio_fea = hr_vaz_fea - (TTT_min * 60)
fea_df = fea_df.withColumn("final_fea", F.col("hr_vaz_fea")) \
               .withColumn(
                    "inicio_fea",
                    F.from_unixtime(
                        F.col("hr_vaz_fea").cast("long") - F.col("TTT_seconds")
                    ).cast("timestamp")
               )


# ======================================================
# 3. TRATAMENTO FP (estes sim vêm com HHmm)
# ======================================================
fp_df = convert_datetime(fp_df, "DATACHEGADAFP", "HORACHEGADAFP", "chegada_fp")
fp_df = convert_datetime(fp_df, "DATASAIDAFP", "HORASAIDAFP", "saida_fp")

fp_df = fp_df.withColumn("inicio_fp", F.col("chegada_fp")) \
             .withColumn("final_fp", F.col("saida_fp"))

# Converter TTTFP (string) para double → minutos → segundos
fp_df = fp_df.withColumn("TTTFP_min", F.col("TTTFP").cast("double")) \
             .withColumn("TTTFP_seconds", (F.col("TTTFP_min") * 60).cast("long"))

# inicio_fp_tttfp = saida_fp - TTTFP em segundos
fp_df = fp_df.withColumn(
    "inicio_fp_tttfp",
    F.from_unixtime(
        F.col("saida_fp").cast("long") - F.col("TTTFP_seconds")
    ).cast("timestamp")
)


# ======================================================
# 4. TRATAMENTO VD
# ======================================================

# 1. Primeiro juntamos FP + VD para trazer o TTTVD
vd_joined = (
    vd_df.alias("vd")
    .join(fp_df.select("corrida", "TTTVD"), "corrida", "left")  # <-- TTTVD vem do FP
)

# 2. Convertemos TTTVD e montamos inicio_vd
vd_joined = (
    vd_joined
    .withColumn("final_vd", F.to_timestamp("datahorasaidavd"))
    .withColumn("TTTVD_min", F.col("TTTVD").cast("double"))
    .withColumn("TTTVD_seconds", (F.col("TTTVD_min") * 60).cast("long"))
    .withColumn(
        "inicio_vd",
        F.from_unixtime(
            F.col("datahorasaidavd").cast("long") - F.col("TTTVD_seconds")
        ).cast("timestamp")
    )
)


# ======================================================
# 5. TRATAMENTO LC (tempo real)
# ======================================================

lc_df = lc_df.withColumn("inicio_lc", F.to_timestamp("INICIOLINGOTAMENTO")) \
             .withColumn("final_lc", F.to_timestamp("FINALLINGOTAMENTO"))

# ======================================================
# 6. JUNÇÃO FINAL (ajuste corrida se o nome for diferente)
# ======================================================
df_final = (
    fea_df.alias("fea")
    .join(fp_df.alias("fp"), "corrida", "left")
    .join(vd_joined.alias("vd"), "corrida", "left")
    .join(lc_df.alias("lc"), "corrida", "left")
    .select(
        "corrida",
        # FEA
        "inicio_fea", "final_fea",
        # FP
        "inicio_fp", "final_fp", 
        # VD
        "inicio_vd", "final_vd",
        # LC
        "inicio_lc", "final_lc", "ADICIONALTTTFP", "ADICIONALTTTVD"
    )
)

# ======================================================
# 7. EXIBE O RESULTADO FINAL
# ======================================================
display(df_final.orderBy(F.desc("corrida")))


# COMMAND ----------

# DBTITLE 1,Convert df_final to pandas and prepare data
# Convert df_final to pandas
df_stages = df_final.toPandas()

# Convert timestamp columns to datetime
for col in ['inicio_fea', 'final_fea', 'inicio_fp', 'final_fp', 'inicio_vd', 'final_vd', 'inicio_lc', 'final_lc']:
    df_stages[col] = pd.to_datetime(df_stages[col])

print(f"Total corridas: {len(df_stages):,}")
print(f"\nDate range in df_stages:")
print(f"  FEA: {df_stages['inicio_fea'].min()} to {df_stages['final_fea'].max()}")
print(f"\nDate range in df_numeric:")
print(f"  Tags: {df_numeric['Time'].min()} to {df_numeric['Time'].max()}")

# Show sample
print(f"\nSample of df_stages:")
display(df_stages[['corrida', 'inicio_fea', 'final_fea']].head(10))

# COMMAND ----------

# DBTITLE 1,Select 5-hour period and filter data
# Select a specific day and 5-hour window
# Using the first day from df_numeric
selected_date = sorted(df_numeric['Date'].unique())[0]
print(f"Selected date: {selected_date}")

# Define 5-hour window (you can adjust these times)
start_time = pd.to_datetime(f"{selected_date} 08:00:00")
end_time = start_time + pd.Timedelta(hours=5)

print(f"Time window: {start_time} to {end_time}")
print("="*80)

# Filter df_numeric for this period
df_tags_period = df_numeric[
    (df_numeric['Time'] >= start_time) & 
    (df_numeric['Time'] <= end_time)
].copy().sort_values('Time')

print(f"\nTag records in this period: {len(df_tags_period):,}")

# Filter df_stages for corridas that overlap with this period
# A corrida overlaps if: inicio_fea <= end_time AND final_fea >= start_time
df_stages_period = df_stages[
    (df_stages['inicio_fea'] <= end_time) & 
    (df_stages['final_fea'] >= start_time)
].copy()

print(f"Corridas with FEA in this period: {len(df_stages_period)}")
print(f"\nCorridas:")
for idx, row in df_stages_period.iterrows():
    print(f"  Corrida {row['corrida']}: {row['inicio_fea']} → {row['final_fea']}")

# COMMAND ----------

# DBTITLE 1,Plot time series with FEA stage markers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print(f"Creating time series plots with FEA markers...")
print("="*80)

# Get numeric columns (excluding Time and Date)
numeric_cols = [col for col in df_tags_period.columns if col not in ['Time', 'Date']]

print(f"Total plots to create: {len(numeric_cols)}")
print()

# Plot each variable individually
for idx, col in enumerate(numeric_cols, 1):
    print(f"[{idx}/{len(numeric_cols)}] Plotting {col}...")
    
    # Create individual figure
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Plot time series
    ax.plot(df_tags_period['Time'], df_tags_period[col], 
            linewidth=1.5, alpha=0.8, color='steelblue', label='Tag values')
    
    # Add vertical lines for FEA inicio and final for each corrida
    for _, row in df_stages_period.iterrows():
        corrida_num = row['corrida']
        
        # Inicio FEA (green line)
        if pd.notna(row['inicio_fea']):
            ax.axvline(x=row['inicio_fea'], color='green', linestyle='--', 
                      linewidth=2, alpha=0.7)
            ax.text(row['inicio_fea'], ax.get_ylim()[1]*0.98, f"{corrida_num}", 
                   rotation=90, va='top', ha='right', fontsize=9, color='green', fontweight='bold')
        
        # Final FEA (red line)
        if pd.notna(row['final_fea']):
            ax.axvline(x=row['final_fea'], color='red', linestyle='--', 
                      linewidth=2, alpha=0.7)
            ax.text(row['final_fea'], ax.get_ylim()[1]*0.98, f"{corrida_num}", 
                   rotation=90, va='top', ha='left', fontsize=9, color='red', fontweight='bold')
    
    # Formatting
    ax.set_title(f'{col} - Time Series with FEA Stages ({selected_date})', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Format x-axis to show time nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue', linewidth=2, label='Tag values'),
        Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Início FEA'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Final FEA')
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
    
    plt.tight_layout()
    display(fig)
    plt.close(fig)  # Close to free memory

print(f"\n{'='*80}")
print(f"✓ All {len(numeric_cols)} time series plots created successfully!")
print(f"\nLegend:")
print(f"  Green dashed lines (--): Início FEA")
print(f"  Red dashed lines (--): Final FEA")
print(f"  Numbers on lines: Corrida ID")

# COMMAND ----------

# DBTITLE 1,Adjust df_numeric to UTC-3 and create plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print(f"Adjusting df_numeric timestamps to UTC-3...")
print("="*80)

# Create a copy of df_numeric with UTC-3 adjustment (subtract 3 hours)
df_numeric_utc3 = df_numeric.copy()
df_numeric_utc3['Time'] = df_numeric_utc3['Time'] - pd.Timedelta(hours=3)
df_numeric_utc3['Date'] = df_numeric_utc3['Time'].dt.date

print(f"Original time range: {df_numeric['Time'].min()} to {df_numeric['Time'].max()}")
print(f"UTC-3 time range: {df_numeric_utc3['Time'].min()} to {df_numeric_utc3['Time'].max()}")
print()

# Select the same date and filter
selected_date_utc3 = sorted(df_numeric_utc3['Date'].unique())[0]
print(f"Selected date (UTC-3): {selected_date_utc3}")

# Define 5-hour window
start_time_utc3 = pd.to_datetime(f"{selected_date_utc3} 08:00:00")
end_time_utc3 = start_time_utc3 + pd.Timedelta(hours=5)

print(f"Time window (UTC-3): {start_time_utc3} to {end_time_utc3}")
print("="*80)

# Filter df_numeric_utc3 for this period
df_tags_period_utc3 = df_numeric_utc3[
    (df_numeric_utc3['Time'] >= start_time_utc3) & 
    (df_numeric_utc3['Time'] <= end_time_utc3)
].copy().sort_values('Time')

print(f"\nTag records in this period: {len(df_tags_period_utc3):,}")

# Filter df_stages for corridas that overlap with this period
df_stages_period_utc3 = df_stages[
    (df_stages['inicio_fea'] <= end_time_utc3) & 
    (df_stages['final_fea'] >= start_time_utc3)
].copy()

print(f"Corridas with FEA in this period: {len(df_stages_period_utc3)}")
print(f"\nCorridas:")
for idx, row in df_stages_period_utc3.iterrows():
    print(f"  Corrida {row['corrida']}: {row['inicio_fea']} → {row['final_fea']}")
print()

# Get numeric columns
numeric_cols_utc3 = [col for col in df_tags_period_utc3.columns if col not in ['Time', 'Date']]

print(f"Total plots to create: {len(numeric_cols_utc3)}")
print()

# Plot each variable individually
for idx, col in enumerate(numeric_cols_utc3, 1):
    print(f"[{idx}/{len(numeric_cols_utc3)}] Plotting {col} (UTC-3)...")
    
    # Create individual figure
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Plot time series
    ax.plot(df_tags_period_utc3['Time'], df_tags_period_utc3[col], 
            linewidth=1.5, alpha=0.8, color='steelblue', label='Tag values')
    
    # Add vertical lines for FEA inicio and final for each corrida
    for _, row in df_stages_period_utc3.iterrows():
        corrida_num = row['corrida']
        
        # Inicio FEA (green line)
        if pd.notna(row['inicio_fea']):
            ax.axvline(x=row['inicio_fea'], color='green', linestyle='--', 
                      linewidth=2, alpha=0.7)
            ax.text(row['inicio_fea'], ax.get_ylim()[1]*0.98, f"{corrida_num}", 
                   rotation=90, va='top', ha='right', fontsize=9, color='green', fontweight='bold')
        
        # Final FEA (red line)
        if pd.notna(row['final_fea']):
            ax.axvline(x=row['final_fea'], color='red', linestyle='--', 
                      linewidth=2, alpha=0.7)
            ax.text(row['final_fea'], ax.get_ylim()[1]*0.98, f"{corrida_num}", 
                   rotation=90, va='top', ha='left', fontsize=9, color='red', fontweight='bold')
    
    # Formatting
    ax.set_title(f'{col} - Time Series with FEA Stages (UTC-3) - {selected_date_utc3}', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Time (UTC-3)', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Format x-axis to show time nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue', linewidth=2, label='Tag values (UTC-3)'),
        Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Início FEA'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Final FEA')
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
    
    plt.tight_layout()
    display(fig)
    plt.close(fig)  # Close to free memory

print(f"\n{'='*80}")
print(f"✓ All {len(numeric_cols_utc3)} time series plots (UTC-3) created successfully!")
print(f"\nLegend:")
print(f"  Green dashed lines (--): Início FEA")
print(f"  Red dashed lines (--): Final FEA")
print(f"  Numbers on lines: Corrida ID")
print(f"\nNote: Tag timestamps adjusted to UTC-3 (subtracted 3 hours)")

# COMMAND ----------

# DBTITLE 1,Investigate ACI@FEA_GERAL_NUMCORR data and plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("Investigating ACI@FEA_GERAL_NUMCORR...")
print("="*80)

# Check if column exists
col_name = 'ACI@FEA_GERAL_NUMCORR'

if col_name in df_tags_period.columns:
    print(f"✓ Column '{col_name}' found in df_tags_period")
    print()
    
    # Analyze the data
    print("Data statistics:")
    print(f"  Count: {df_tags_period[col_name].count()}")
    print(f"  Non-null: {df_tags_period[col_name].notna().sum()}")
    print(f"  Null: {df_tags_period[col_name].isna().sum()}")
    print(f"  Min: {df_tags_period[col_name].min()}")
    print(f"  Max: {df_tags_period[col_name].max()}")
    print(f"  Mean: {df_tags_period[col_name].mean():.2f}")
    print(f"  Std: {df_tags_period[col_name].std():.2f}")
    print()
    
    # Show unique values if there are few
    unique_vals = df_tags_period[col_name].unique()
    print(f"Unique values: {len(unique_vals)}")
    if len(unique_vals) <= 20:
        print(f"  Values: {sorted(unique_vals)}")
    print()
    
    # Show sample data
    print("Sample data:")
    display(df_tags_period[['Time', col_name]].head(20))
    print()
    
    # Create the plot
    print("Creating plot...")
    fig, ax = plt.subplots(figsize=(16, 6))  # Increased height
    
    # Plot time series
    ax.plot(df_tags_period['Time'], df_tags_period[col_name], 
            linewidth=1.5, alpha=0.8, color='steelblue', label='Tag values', marker='o', markersize=3)
    
    # Add vertical lines for FEA inicio and final for each corrida
    for _, row in df_stages_period.iterrows():
        corrida_num = row['corrida']
        
        # Inicio FEA (green line)
        if pd.notna(row['inicio_fea']):
            ax.axvline(x=row['inicio_fea'], color='green', linestyle='--', 
                      linewidth=2, alpha=0.7)
            # Position text higher to avoid overlap
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(row['inicio_fea'], y_pos, f"{corrida_num}", 
                   rotation=90, va='top', ha='right', fontsize=9, color='green', fontweight='bold')
        
        # Final FEA (red line)
        if pd.notna(row['final_fea']):
            ax.axvline(x=row['final_fea'], color='red', linestyle='--', 
                      linewidth=2, alpha=0.7)
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(row['final_fea'], y_pos, f"{corrida_num}", 
                   rotation=90, va='top', ha='left', fontsize=9, color='red', fontweight='bold')
    
    # Formatting
    ax.set_title(f'{col_name} - Time Series with FEA Stages ({selected_date})', 
                fontsize=14, fontweight='bold', pad=20)  # Increased padding
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Format x-axis to show time nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue', linewidth=2, label='Tag values'),
        Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Início FEA'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Final FEA')
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
    
    # Use subplots_adjust instead of tight_layout to avoid the warning
    plt.subplots_adjust(top=0.92, bottom=0.15, left=0.08, right=0.95)
    display(fig)
    plt.close(fig)
    
    print("\n✓ Plot created successfully!")
    print("\nObservation: This variable likely contains corrida numbers (discrete values)")
    print("that change at specific times, which is why the plot might look different.")
    
else:
    print(f"✗ Column '{col_name}' NOT found in df_tags_period")
    print(f"\nAvailable columns: {list(df_tags_period.columns)}")

# COMMAND ----------

# DBTITLE 1,Function to detect inclination cycles
def detect_inclination_cycles(df, time_col, value_col, start_threshold, end_threshold, tolerance=0.5):
    """
    Detecta ciclos baseados em cruzamentos de limiar de inclinação.
    
    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame com dados de série temporal
    time_col : str
        Nome da coluna de timestamp
    value_col : str
        Nome da coluna de valor a analisar
    start_threshold : float
        Valor de limiar para iniciar um novo ciclo (ex: 3)
    end_threshold : float
        Valor de limiar para finalizar um ciclo (ex: -6 ou -7)
    tolerance : float
        Tolerância para comparação de limiar (padrão: 0.5)
    
    Retorna:
    --------
    pandas.DataFrame
        DataFrame com colunas: corrida, inicio_time, inicio_value, final_time, final_value
    """
    
    # Ordenar por tempo
    df_sorted = df[[time_col, value_col]].copy().sort_values(time_col).reset_index(drop=True)
    
    # Remover valores NaN
    df_sorted = df_sorted.dropna(subset=[value_col])
    
    cycles = []
    corrida_num = 1
    state = 'waiting_start'  # Estados: 'waiting_start', 'waiting_end'
    current_cycle = {}
    
    print(f"Detectando ciclos com start_threshold={start_threshold}, end_threshold={end_threshold}")
    print(f"Total de registros para analisar: {len(df_sorted):,}")
    print("="*80)
    
    for idx, row in df_sorted.iterrows():
        value = row[value_col]
        time = row[time_col]
        
        if state == 'waiting_start':
            # Verifica se o valor cruza o limiar de início (ex: >= 3)
            if value >= (start_threshold - tolerance):
                current_cycle = {
                    'corrida': corrida_num,
                    'inicio_time': time,
                    'inicio_value': value
                }
                state = 'waiting_end'
                print(f"Corrida {corrida_num}: INÍCIO detectado em {time} (valor={value:.2f}°)")
        
        elif state == 'waiting_end':
            # Verifica se o valor cruza o limiar de fim (ex: <= -6)
            if value <= (end_threshold + tolerance):
                current_cycle['final_time'] = time
                current_cycle['final_value'] = value
                cycles.append(current_cycle)
                print(f"Corrida {corrida_num}: FIM detectado em {time} (valor={value:.2f}°)")
                print()
                
                # Resetar para o próximo ciclo
                corrida_num += 1
                state = 'waiting_start'
                current_cycle = {}
    
    # Se houver um ciclo incompleto, reportar
    if state == 'waiting_end':
        print(f"⚠ Corrida {corrida_num}: Iniciada mas não completada (nenhum limiar de fim detectado)")
        print(f"   Iniciada em: {current_cycle['inicio_time']} (valor={current_cycle['inicio_value']:.2f}°)")
    
    print("="*80)
    print(f"Total de ciclos detectados: {len(cycles)}")
    
    # Converter para DataFrame
    df_cycles = pd.DataFrame(cycles)
    
    return df_cycles

print("✓ Função 'detect_inclination_cycles' criada com sucesso!")

# COMMAND ----------

# DBTITLE 1,Detect cycles for ACI@FEA_GERAL_INCLINOM for entire dataset
# Parâmetros
var_name = 'ACI@FEA_GERAL_INCLINOM'
start_threshold = 3.0
end_threshold = -6.0

print("Analisando TODO o dataset df_numeric")
print("="*80)

# Usar todo o df_numeric (sem filtrar por dia)
df_full_dataset = df_numeric.copy().sort_values('Time')

print(f"Total de registros no dataset: {len(df_full_dataset):,}")
print(f"Intervalo de tempo: {df_full_dataset['Time'].min()} até {df_full_dataset['Time'].max()}")
print(f"Período: {(df_full_dataset['Time'].max() - df_full_dataset['Time'].min()).days} dias")
print()

# Verificar se a variável existe
if var_name not in df_full_dataset.columns:
    print(f"✗ ERRO: Variável '{var_name}' não encontrada no dataset!")
    print(f"Colunas disponíveis: {list(df_full_dataset.columns)}")
else:
    print(f"✓ Variável '{var_name}' encontrada")
    print(f"  Valores não-nulos: {df_full_dataset[var_name].notna().sum():,}")
    print(f"  Intervalo de valores: {df_full_dataset[var_name].min():.2f} até {df_full_dataset[var_name].max():.2f}")
    print()
    
    # Detectar ciclos em TODO o dataset
    df_detected_cycles = detect_inclination_cycles(
        df=df_full_dataset,
        time_col='Time',
        value_col=var_name,
        start_threshold=start_threshold,
        end_threshold=end_threshold,
        tolerance=0.5
    )
    
    # Exibir resultados
    if len(df_detected_cycles) > 0:
        print("\nResumo dos ciclos detectados:")
        print(f"  Total de ciclos: {len(df_detected_cycles)}")
        print(f"  Primeira corrida: {df_detected_cycles['inicio_time'].min()}")
        print(f"  Última corrida: {df_detected_cycles['final_time'].max()}")
        print()
        print("Primeiros 10 ciclos:")
        display(df_detected_cycles.head(10))
        print()
        print("Últimos 10 ciclos:")
        display(df_detected_cycles.tail(10))
    else:
        print("\n⚠ Nenhum ciclo completo detectado com esses limiares")

# COMMAND ----------

# DBTITLE 1,Calculate cycle durations and create histogram
import matplotlib.pyplot as plt
import numpy as np

print("Análise de Duração das Corridas")
print("="*80)

# Calcular duração de cada corrida em minutos
if len(df_detected_cycles) > 0:
    df_detected_cycles['duracao_minutos'] = (
        df_detected_cycles['final_time'] - df_detected_cycles['inicio_time']
    ).dt.total_seconds() / 60
    
    # Estatísticas descritivas
    duracao = df_detected_cycles['duracao_minutos']
    
    print(f"Total de corridas analisadas: {len(duracao)}")
    print(f"\nEstatísticas de Duração (minutos):")
    print(f"  Média: {duracao.mean():.2f} min")
    print(f"  Mediana: {duracao.median():.2f} min")
    print(f"  Desvio Padrão: {duracao.std():.2f} min")
    print(f"  Mínimo: {duracao.min():.2f} min")
    print(f"  Máximo: {duracao.max():.2f} min")
    print(f"  Quartil 25%: {duracao.quantile(0.25):.2f} min")
    print(f"  Quartil 75%: {duracao.quantile(0.75):.2f} min")
    print()
    
    # Criar figura com 2 subplots (lado a lado)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    # ========== HISTOGRAMA 1: Completo (todos os dados) ==========
    n, bins, patches = ax1.hist(duracao, bins=15, color='steelblue', 
                               alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Adicionar linha vertical para média
    ax1.axvline(duracao.mean(), color='red', linestyle='--', 
               linewidth=2.5, alpha=0.8, label=f'Média: {duracao.mean():.2f} min')
    
    # Adicionar linha vertical para mediana
    ax1.axvline(duracao.median(), color='green', linestyle='--', 
               linewidth=2.5, alpha=0.8, label=f'Mediana: {duracao.median():.2f} min')
    
    # Formatação do histograma completo
    ax1.set_title(f'Distribuição Completa da Duração das Corridas\n' +
                f'Variável: {var_name} | Limiares: {start_threshold}° → {end_threshold}°',
                fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Duração (minutos)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequência (número de corridas)', fontsize=12, fontweight='bold')
    ax1.tick_params(labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    
    # Adicionar texto com estatísticas no gráfico completo
    stats_text = f"n = {len(duracao)} corridas\n"
    stats_text += f"μ = {duracao.mean():.2f} min\n"
    stats_text += f"σ = {duracao.std():.2f} min\n"
    stats_text += f"Min = {duracao.min():.2f} min\n"
    stats_text += f"Max = {duracao.max():.2f} min"
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========== HISTOGRAMA 2: Clipado (0-40 minutos) ==========
    # Clipar valores acima de 40 para 40
    duracao_clipped = duracao.copy()
    duracao_clipped[duracao_clipped > 40] = 40
    
    # Contar quantas corridas foram clipadas
    n_clipped = (duracao > 40).sum()
    
    # Criar bins de 0 a 40
    bins_clipped = np.arange(0, 42, 2.5)  # Bins de 2.5 minutos até 40
    
    n2, bins2, patches2 = ax2.hist(duracao_clipped, bins=bins_clipped, color='steelblue', 
                                   alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Adicionar linha vertical para média (se estiver no range)
    if duracao.mean() <= 40:
        ax2.axvline(duracao.mean(), color='red', linestyle='--', 
                   linewidth=2.5, alpha=0.8, label=f'Média: {duracao.mean():.2f} min')
    
    # Adicionar linha vertical para mediana (se estiver no range)
    if duracao.median() <= 40:
        ax2.axvline(duracao.median(), color='green', linestyle='--', 
                   linewidth=2.5, alpha=0.8, label=f'Mediana: {duracao.median():.2f} min')
    
    # Formatação do histograma clipado
    ax2.set_title(f'Distribuição Clipada (0-40 min)\n' +
                f'Valores > 40 min agrupados no último bin',
                fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Duração (minutos)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequência (número de corridas)', fontsize=12, fontweight='bold')
    ax2.tick_params(labelsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax2.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax2.set_xlim(0, 42)
    
    # Adicionar texto com informações do clipping
    clip_text = f"n = {len(duracao)} corridas\n"
    clip_text += f"μ = {duracao.mean():.2f} min\n"
    clip_text += f"Mediana = {duracao.median():.2f} min\n"
    clip_text += f"\nClipadas (>40): {n_clipped} ({n_clipped/len(duracao)*100:.1f}%)"
    
    ax2.text(0.98, 0.98, clip_text, transform=ax2.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    display(fig)
    plt.close(fig)
    
    print("\n✓ Histogramas criados com sucesso!")
    print(f"\nCorridas clipadas (>40 min): {n_clipped} de {len(duracao)} ({n_clipped/len(duracao)*100:.1f}%)")
    print("\nDistribuição por faixas de duração:")
    
    # Criar faixas de duração
    faixas = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 200)]
    for faixa_min, faixa_max in faixas:
        count = ((duracao >= faixa_min) & (duracao < faixa_max)).sum()
        percent = (count / len(duracao)) * 100
        if count > 0:
            if faixa_max >= 200:
                print(f"  {faixa_min:2d}+     min: {count:3d} corridas ({percent:5.1f}%)")
            else:
                print(f"  {faixa_min:2d}-{faixa_max:2d} min: {count:3d} corridas ({percent:5.1f}%)")
    
    print("\nTabela completa de corridas com duração:")
    display(df_detected_cycles[['corrida', 'inicio_time', 'final_time', 'duracao_minutos']].sort_values('duracao_minutos'))
    
else:
    print("⚠ Nenhum ciclo detectado para análise")

# COMMAND ----------

# DBTITLE 1,Function to plot inclination cycles with detection markers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_inclination_cycles(df, var_name, time_col='Time', 
                           start_date=None, end_date=None,
                           start_threshold=3.0, end_threshold=-6.0,
                           df_cycles=None, figsize=(20, 7),
                           hour_interval=1):
    """
    Plota série temporal de inclinação com marcadores de ciclos detectados.
    
    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame com dados de série temporal
    var_name : str
        Nome da variável/coluna a plotar
    time_col : str
        Nome da coluna de timestamp (padrão: 'Time')
    start_date : str ou datetime
        Data/hora inicial do período (opcional, se None usa todo o df)
    end_date : str ou datetime
        Data/hora final do período (opcional, se None usa todo o df)
    start_threshold : float
        Valor de limiar de início dos ciclos (padrão: 3.0)
    end_threshold : float
        Valor de limiar de fim dos ciclos (padrão: -6.0)
    df_cycles : pandas.DataFrame
        DataFrame com ciclos detectados (colunas: corrida, inicio_time, final_time)
        Se None, não plota marcadores de ciclos
    figsize : tuple
        Tamanho da figura (largura, altura) em polegadas (padrão: (20, 7))
    hour_interval : int
        Intervalo de horas para marcadores no eixo x (padrão: 1)
    
    Retorna:
    --------
    matplotlib.figure.Figure
        Objeto da figura criada
    """
    
    # Verificar se a variável existe
    if var_name not in df.columns:
        print(f"✗ ERRO: Variável '{var_name}' não encontrada no dataset!")
        print(f"Colunas disponíveis: {list(df.columns)}")
        return None
    
    # Filtrar por período se especificado
    df_plot = df.copy()
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df_plot = df_plot[df_plot[time_col] >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df_plot = df_plot[df_plot[time_col] <= end_date]
    
    df_plot = df_plot.sort_values(time_col)
    
    # Informações do período
    period_start = df_plot[time_col].min()
    period_end = df_plot[time_col].max()
    
    print(f"Criando gráfico para {var_name}")
    print("="*80)
    print(f"Período: {period_start} até {period_end}")
    print(f"Total de registros: {len(df_plot):,}")
    print(f"Limiar de início: {start_threshold}°")
    print(f"Limiar de fim: {end_threshold}°")
    
    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plotar série temporal
    ax.plot(df_plot[time_col], df_plot[var_name], 
            linewidth=1.5, alpha=0.8, color='steelblue', label=f'{var_name}')
    
    # Adicionar linhas horizontais para os limiares
    ax.axhline(y=start_threshold, color='green', linestyle=':', linewidth=2.5, alpha=0.7, 
               label=f'Limiar de Início ({start_threshold}°)')
    ax.axhline(y=end_threshold, color='red', linestyle=':', linewidth=2.5, alpha=0.7, 
               label=f'Limiar de Fim ({end_threshold}°)')
    
    # Adicionar linhas verticais para os ciclos detectados
    if df_cycles is not None and len(df_cycles) > 0:
        # Filtrar ciclos que estão no período plotado
        df_cycles_plot = df_cycles[
            (df_cycles['inicio_time'] >= period_start) & 
            (df_cycles['inicio_time'] <= period_end)
        ].copy()
        
        print(f"Ciclos no período: {len(df_cycles_plot)}")
        
        for _, row in df_cycles_plot.iterrows():
            corrida_num = row['corrida']
            
            # Início do ciclo (linha vertical verde)
            if pd.notna(row['inicio_time']):
                ax.axvline(x=row['inicio_time'], color='green', linestyle='--', 
                          linewidth=2.5, alpha=0.8)
                # Adicionar rótulo no topo
                y_pos_start = ax.get_ylim()[1] * 0.98
                ax.text(row['inicio_time'], y_pos_start, f"C{corrida_num}\nINÍCIO", 
                       rotation=0, va='top', ha='center', fontsize=9, 
                       color='green', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='green'))
            
            # Fim do ciclo (linha vertical vermelha)
            if pd.notna(row['final_time']) and row['final_time'] <= period_end:
                ax.axvline(x=row['final_time'], color='red', linestyle='--', 
                          linewidth=2.5, alpha=0.8)
                # Adicionar rótulo na parte inferior
                y_pos_end = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                ax.text(row['final_time'], y_pos_end, f"C{corrida_num}\nFIM", 
                       rotation=0, va='bottom', ha='center', fontsize=9, 
                       color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # Formatação
    period_label = f"{period_start.strftime('%Y-%m-%d %H:%M')} até {period_end.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(f'{var_name} - Detecção de Ciclos\n' + 
                f'Período: {period_label} | Limiar Início: {start_threshold}° | Limiar Fim: {end_threshold}°', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Horário', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inclinação (graus)', fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Formatar eixo x para mostrar o tempo de forma agradável
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Legenda
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    
    print("\n✓ Gráfico criado com sucesso!")
    
    # Estatísticas dos ciclos
    if df_cycles is not None and len(df_cycles_plot) > 0:
        print(f"\nDetalhes dos ciclos no período:")
        for _, row in df_cycles_plot.iterrows():
            if pd.notna(row['final_time']):
                duration = (row['final_time'] - row['inicio_time']).total_seconds() / 60
                print(f"  Corrida {row['corrida']}: {row['inicio_time'].strftime('%H:%M:%S')} → " +
                      f"{row['final_time'].strftime('%H:%M:%S')} (Duração: {duration:.1f} min)")
            else:
                print(f"  Corrida {row['corrida']}: {row['inicio_time'].strftime('%H:%M:%S')} → (em andamento)")
    
    return fig

print("✓ Função 'plot_inclination_cycles' criada com sucesso!")
print("\nExemplo de uso:")
print("  # Plotar dia completo")
print("  fig = plot_inclination_cycles(df_numeric, 'ACI@FEA_GERAL_INCLINOM', df_cycles=df_detected_cycles)")
print("  display(fig)")
print("  plt.close(fig)")
print("\n  # Plotar período específico (ex: 6 horas)")
print("  fig = plot_inclination_cycles(df_numeric, 'ACI@FEA_GERAL_INCLINOM',")
print("                               start_date='2025-11-11 08:00:00',")
print("                               end_date='2025-11-11 14:00:00',")
print("                               df_cycles=df_detected_cycles)")
print("  display(fig)")
print("  plt.close(fig)")

# COMMAND ----------

# DBTITLE 1,Function to detect TEMPO_VAZAMENTO peaks (wave cycles)
def detect_vazamento_peaks(df, time_col, value_col, start_threshold=5, end_threshold=5, min_peak_value=50):
    """
    Detecta ciclos de picos (ondas) na variável de tempo de vazamento.
    Cada ciclo começa quando o valor sobe acima do limiar inicial,
    atinge um pico, e termina quando volta abaixo do limiar final.
    
    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame com dados de série temporal
    time_col : str
        Nome da coluna de timestamp
    value_col : str
        Nome da coluna de valor a analisar
    start_threshold : float
        Valor de limiar para iniciar um novo ciclo (padrão: 5)
    end_threshold : float
        Valor de limiar para finalizar um ciclo (padrão: 5)
    min_peak_value : float
        Valor mínimo do pico para considerar um ciclo válido (padrão: 50)
    
    Retorna:
    --------
    pandas.DataFrame
        DataFrame com colunas: corrida, inicio_time, inicio_value, pico_time, pico_value, final_time, final_value
    """
    
    # Ordenar por tempo
    df_sorted = df[[time_col, value_col]].copy().sort_values(time_col).reset_index(drop=True)
    
    # Remover valores NaN
    df_sorted = df_sorted.dropna(subset=[value_col])
    
    cycles = []
    corrida_num = 1
    state = 'waiting_start'  # Estados: 'waiting_start', 'in_peak', 'waiting_end'
    current_cycle = {}
    current_peak_value = 0
    current_peak_time = None
    
    print(f"Detectando picos de vazamento com start_threshold={start_threshold}, end_threshold={end_threshold}")
    print(f"Valor mínimo de pico: {min_peak_value}")
    print(f"Total de registros para analisar: {len(df_sorted):,}")
    print("="*80)
    
    for idx, row in df_sorted.iterrows():
        value = row[value_col]
        time = row[time_col]
        
        if state == 'waiting_start':
            # Verifica se o valor cruza o limiar de início (começa a subir)
            if value > start_threshold:
                current_cycle = {
                    'corrida': corrida_num,
                    'inicio_time': time,
                    'inicio_value': value
                }
                current_peak_value = value
                current_peak_time = time
                state = 'in_peak'
                
        elif state == 'in_peak':
            # Atualizar o pico se encontrar valor maior
            if value > current_peak_value:
                current_peak_value = value
                current_peak_time = time
            
            # Verifica se o valor está descendo e cruzou o limiar de fim
            if value <= end_threshold:
                # Verificar se o pico atingiu o valor mínimo
                if current_peak_value >= min_peak_value:
                    current_cycle['pico_time'] = current_peak_time
                    current_cycle['pico_value'] = current_peak_value
                    current_cycle['final_time'] = time
                    current_cycle['final_value'] = value
                    cycles.append(current_cycle)
                    
                    print(f"Corrida {corrida_num}: INÍCIO={current_cycle['inicio_time'].strftime('%Y-%m-%d %H:%M:%S')} " +
                          f"| PICO={current_peak_value:.2f} em {current_peak_time.strftime('%H:%M:%S')} " +
                          f"| FIM={time.strftime('%H:%M:%S')}")
                    
                    corrida_num += 1
                else:
                    print(f"⚠ Ciclo descartado (pico={current_peak_value:.2f} < {min_peak_value}): " +
                          f"{current_cycle['inicio_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Resetar para o próximo ciclo
                state = 'waiting_start'
                current_cycle = {}
                current_peak_value = 0
                current_peak_time = None
    
    # Se houver um ciclo incompleto, reportar
    if state == 'in_peak':
        print(f"\n⚠ Corrida {corrida_num}: Iniciada mas não completada")
        print(f"   Iniciada em: {current_cycle['inicio_time']} (valor={current_cycle['inicio_value']:.2f})")
        print(f"   Pico atual: {current_peak_value:.2f} em {current_peak_time}")
    
    print("="*80)
    print(f"Total de ciclos detectados: {len(cycles)}")
    
    # Converter para DataFrame
    df_cycles = pd.DataFrame(cycles)
    
    return df_cycles

print("✓ Função 'detect_vazamento_peaks' criada com sucesso!")

# COMMAND ----------

# DBTITLE 1,Detect TEMPO_VAZAMENTO peaks for entire dataset
# Parâmetros para detecção de picos
var_vazamento = 'ACI@FEA_TEMPO_VAZAMENTO'
start_threshold_vaz = 5.0  # Valor acima do qual considera início do pico
end_threshold_vaz = 5.0    # Valor abaixo do qual considera fim do pico
min_peak_value_vaz = 50.0  # Valor mínimo do pico para ser considerado válido

print("Analisando TODO o dataset df_numeric para TEMPO_VAZAMENTO")
print("="*80)

# Usar todo o df_numeric
df_full_dataset_vaz = df_numeric.copy().sort_values('Time')

print(f"Total de registros no dataset: {len(df_full_dataset_vaz):,}")
print(f"Intervalo de tempo: {df_full_dataset_vaz['Time'].min()} até {df_full_dataset_vaz['Time'].max()}")
print(f"Período: {(df_full_dataset_vaz['Time'].max() - df_full_dataset_vaz['Time'].min()).days} dias")
print()

# Verificar se a variável existe
if var_vazamento not in df_full_dataset_vaz.columns:
    print(f"✗ ERRO: Variável '{var_vazamento}' não encontrada no dataset!")
    print(f"Colunas disponíveis: {list(df_full_dataset_vaz.columns)}")
else:
    print(f"✓ Variável '{var_vazamento}' encontrada")
    print(f"  Valores não-nulos: {df_full_dataset_vaz[var_vazamento].notna().sum():,}")
    print(f"  Intervalo de valores: {df_full_dataset_vaz[var_vazamento].min():.2f} até {df_full_dataset_vaz[var_vazamento].max():.2f}")
    print()
    
    # Detectar picos em TODO o dataset
    df_vazamento_cycles = detect_vazamento_peaks(
        df=df_full_dataset_vaz,
        time_col='Time',
        value_col=var_vazamento,
        start_threshold=start_threshold_vaz,
        end_threshold=end_threshold_vaz,
        min_peak_value=min_peak_value_vaz
    )
    
    # Exibir resultados
    if len(df_vazamento_cycles) > 0:
        print("\nResumo dos ciclos detectados:")
        print(f"  Total de ciclos: {len(df_vazamento_cycles)}")
        print(f"  Primeira corrida: {df_vazamento_cycles['inicio_time'].min()}")
        print(f"  Última corrida: {df_vazamento_cycles['final_time'].max()}")
        print()
        print("Primeiros 10 ciclos:")
        display(df_vazamento_cycles.head(10))
        print()
        print("Últimos 10 ciclos:")
        display(df_vazamento_cycles.tail(10))
    else:
        print("\n⚠ Nenhum ciclo completo detectado com esses limiares")

# COMMAND ----------

# DBTITLE 1,Calculate peak durations and create histogram for TEMPO_VAZAMENTO
import matplotlib.pyplot as plt
import numpy as np

print("Análise de Duração e Picos - TEMPO_VAZAMENTO")
print("="*80)

# Calcular duração e estatísticas de cada ciclo
if len(df_vazamento_cycles) > 0:
    # Duração total do ciclo (início até fim)
    df_vazamento_cycles['duracao_total_minutos'] = (
        df_vazamento_cycles['final_time'] - df_vazamento_cycles['inicio_time']
    ).dt.total_seconds() / 60
    
    # Tempo até o pico (início até pico)
    df_vazamento_cycles['tempo_ate_pico_minutos'] = (
        df_vazamento_cycles['pico_time'] - df_vazamento_cycles['inicio_time']
    ).dt.total_seconds() / 60
    
    # Tempo de descida (pico até fim)
    df_vazamento_cycles['tempo_descida_minutos'] = (
        df_vazamento_cycles['final_time'] - df_vazamento_cycles['pico_time']
    ).dt.total_seconds() / 60
    
    # Estatísticas
    duracao = df_vazamento_cycles['duracao_total_minutos']
    picos = df_vazamento_cycles['pico_value']
    
    print(f"Total de corridas analisadas: {len(duracao)}")
    print(f"\n📊 Estatísticas de Duração Total (minutos):")
    print(f"  Média: {duracao.mean():.2f} min")
    print(f"  Mediana: {duracao.median():.2f} min")
    print(f"  Desvio Padrão: {duracao.std():.2f} min")
    print(f"  Mínimo: {duracao.min():.2f} min")
    print(f"  Máximo: {duracao.max():.2f} min")
    print(f"  Quartil 25%: {duracao.quantile(0.25):.2f} min")
    print(f"  Quartil 75%: {duracao.quantile(0.75):.2f} min")
    
    print(f"\n🔺 Estatísticas de Valor de Pico:")
    print(f"  Média: {picos.mean():.2f}")
    print(f"  Mediana: {picos.median():.2f}")
    print(f"  Desvio Padrão: {picos.std():.2f}")
    print(f"  Mínimo: {picos.min():.2f}")
    print(f"  Máximo: {picos.max():.2f}")
    print(f"  Quartil 25%: {picos.quantile(0.25):.2f}")
    print(f"  Quartil 75%: {picos.quantile(0.75):.2f}")
    print()
    
    # Criar figura com 2 histogramas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    # ========== HISTOGRAMA 1: Duração Total ==========
    n1, bins1, patches1 = ax1.hist(duracao, bins=20, color='steelblue', 
                                   alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax1.axvline(duracao.mean(), color='red', linestyle='--', 
               linewidth=2.5, alpha=0.8, label=f'Média: {duracao.mean():.2f} min')
    ax1.axvline(duracao.median(), color='green', linestyle='--', 
               linewidth=2.5, alpha=0.8, label=f'Mediana: {duracao.median():.2f} min')
    
    ax1.set_title(f'Distribuição da Duração Total dos Picos\n' +
                f'Variável: {var_vazamento}',
                fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Duração Total (minutos)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequência (número de corridas)', fontsize=12, fontweight='bold')
    ax1.tick_params(labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    
    stats_text1 = f"n = {len(duracao)} corridas\n"
    stats_text1 += f"μ = {duracao.mean():.2f} min\n"
    stats_text1 += f"σ = {duracao.std():.2f} min\n"
    stats_text1 += f"Min = {duracao.min():.2f} min\n"
    stats_text1 += f"Max = {duracao.max():.2f} min"
    
    ax1.text(0.98, 0.98, stats_text1, transform=ax1.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========== HISTOGRAMA 2: Valor de Pico ==========
    n2, bins2, patches2 = ax2.hist(picos, bins=20, color='coral', 
                                   alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax2.axvline(picos.mean(), color='red', linestyle='--', 
               linewidth=2.5, alpha=0.8, label=f'Média: {picos.mean():.2f}')
    ax2.axvline(picos.median(), color='green', linestyle='--', 
               linewidth=2.5, alpha=0.8, label=f'Mediana: {picos.median():.2f}')
    
    ax2.set_title(f'Distribuição dos Valores de Pico\n' +
                f'Variável: {var_vazamento}',
                fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Valor de Pico', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequência (número de corridas)', fontsize=12, fontweight='bold')
    ax2.tick_params(labelsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax2.legend(fontsize=11, loc='upper right', framealpha=0.9)
    
    stats_text2 = f"n = {len(picos)} corridas\n"
    stats_text2 += f"μ = {picos.mean():.2f}\n"
    stats_text2 += f"σ = {picos.std():.2f}\n"
    stats_text2 += f"Min = {picos.min():.2f}\n"
    stats_text2 += f"Max = {picos.max():.2f}"
    
    ax2.text(0.98, 0.98, stats_text2, transform=ax2.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    display(fig)
    plt.close(fig)
    
    print("\n✓ Histogramas criados com sucesso!")
    
    print("\nTabela completa de corridas com duração e picos:")
    display(df_vazamento_cycles[['corrida', 'inicio_time', 'pico_time', 'pico_value', 
                                 'final_time', 'duracao_total_minutos', 
                                 'tempo_ate_pico_minutos', 'tempo_descida_minutos']].sort_values('duracao_total_minutos'))
    
else:
    print("⚠ Nenhum ciclo detectado para análise")

# COMMAND ----------

# DBTITLE 1,Plot TEMPO_VAZAMENTO peaks for a short period
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Selecionar um período curto para visualização (ex: 6 horas de um dia com várias corridas)
# Vamos pegar um dia que tenha vários ciclos
if len(df_vazamento_cycles) > 0:
    # Encontrar um dia com múltiplos ciclos
    df_vazamento_cycles['date'] = df_vazamento_cycles['inicio_time'].dt.date
    cycles_per_day = df_vazamento_cycles.groupby('date').size().sort_values(ascending=False)
    
    print("Dias com mais ciclos detectados:")
    print(cycles_per_day.head(10))
    print()
    
    # Pegar o dia com mais ciclos
    best_day = cycles_per_day.index[0]
    print(f"Plotando dia com mais ciclos: {best_day}")
    print()
    
    # Definir período de 12 horas nesse dia
    start_plot = pd.Timestamp(f"{best_day} 06:00:00")
    end_plot = pd.Timestamp(f"{best_day} 18:00:00")
    
    # Filtrar dados para o período
    df_plot_vaz = df_numeric[
        (df_numeric['Time'] >= start_plot) & 
        (df_numeric['Time'] <= end_plot)
    ].copy().sort_values('Time')
    
    # Filtrar ciclos para o período
    df_cycles_plot = df_vazamento_cycles[
        (df_vazamento_cycles['inicio_time'] >= start_plot) & 
        (df_vazamento_cycles['inicio_time'] <= end_plot)
    ].copy()
    
    print(f"Período: {start_plot} até {end_plot}")
    print(f"Total de registros: {len(df_plot_vaz):,}")
    print(f"Ciclos no período: {len(df_cycles_plot)}")
    print()
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Plotar série temporal
    ax.plot(df_plot_vaz['Time'], df_plot_vaz[var_vazamento], 
            linewidth=2, alpha=0.8, color='steelblue', label=var_vazamento)
    
    # Adicionar linhas horizontais para os limiares
    ax.axhline(y=start_threshold_vaz, color='green', linestyle=':', linewidth=2, alpha=0.5, 
               label=f'Limiar Início/Fim ({start_threshold_vaz})')
    ax.axhline(y=min_peak_value_vaz, color='orange', linestyle=':', linewidth=2, alpha=0.5, 
               label=f'Pico Mínimo ({min_peak_value_vaz})')
    
    # Adicionar marcadores para cada ciclo
    for _, row in df_cycles_plot.iterrows():
        corrida_num = row['corrida']
        
        # Início do ciclo (linha vertical verde)
        ax.axvline(x=row['inicio_time'], color='green', linestyle='--', 
                  linewidth=2, alpha=0.7)
        ax.text(row['inicio_time'], ax.get_ylim()[1] * 0.95, f"C{corrida_num}\nINÍCIO", 
               rotation=0, va='top', ha='center', fontsize=9, 
               color='green', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='green'))
        
        # Pico (marcador no topo)
        ax.plot(row['pico_time'], row['pico_value'], 'r*', markersize=20, 
               markeredgecolor='darkred', markeredgewidth=1.5, label='_nolegend_')
        ax.text(row['pico_time'], row['pico_value'] + 5, f"{row['pico_value']:.1f}", 
               rotation=0, va='bottom', ha='center', fontsize=10, 
               color='darkred', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='red'))
        
        # Fim do ciclo (linha vertical vermelha)
        ax.axvline(x=row['final_time'], color='red', linestyle='--', 
                  linewidth=2, alpha=0.7)
        ax.text(row['final_time'], ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
               f"C{corrida_num}\nFIM", 
               rotation=0, va='bottom', ha='center', fontsize=9, 
               color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # Formatação
    ax.set_title(f'{var_vazamento} - Detecção de Picos (Ondas)\n' + 
                f'Período: {start_plot.strftime("%Y-%m-%d %H:%M")} até {end_plot.strftime("%H:%M")} | ' +
                f'Limiar: {start_threshold_vaz} | Pico Mínimo: {min_peak_value_vaz}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Horário', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tempo de Vazamento', fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Formatar eixo x
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Legenda
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    display(fig)
    plt.close(fig)
    
    print("\n✓ Gráfico criado com sucesso!")
    
    # Estatísticas dos ciclos no período
    if len(df_cycles_plot) > 0:
        print(f"\nDetalhes dos ciclos no período:")
        for _, row in df_cycles_plot.iterrows():
            duration = row['duracao_total_minutos']
            print(f"  Corrida {row['corrida']}: {row['inicio_time'].strftime('%H:%M:%S')} → " +
                  f"PICO={row['pico_value']:.1f} em {row['pico_time'].strftime('%H:%M:%S')} → " +
                  f"{row['final_time'].strftime('%H:%M:%S')} (Duração: {duration:.1f} min)")
else:
    print("⚠ Nenhum ciclo detectado para plotar")

# COMMAND ----------

# DBTITLE 1,Example 1: Plot full day
# Exemplo 1: Plotar o dia completo
print("EXEMPLO 1: Dia completo")
print("="*80)

fig = plot_inclination_cycles(
    df=df_numeric,
    var_name='ACI@FEA_GERAL_INCLINOM',
    start_threshold=3.0,
    end_threshold=-6.0,
    df_cycles=df_detected_cycles,
    figsize=(20, 7),
    hour_interval=1
)

if fig is not None:
    display(fig)
    plt.close(fig)

# COMMAND ----------

# DBTITLE 1,Example 2: Plot specific time period (6 hours)
# Exemplo 2: Plotar um período específico de 6 horas
print("\nEXEMPLO 2: Período de 6 horas (08:00 - 14:00)")
print("="*80)

fig = plot_inclination_cycles(
    df=df_numeric,
    var_name='ACI@FEA_GERAL_INCLINOM',
    start_date='2025-11-11 08:00:00',
    end_date='2025-11-11 14:00:00',
    start_threshold=3.0,
    end_threshold=-6.0,
    df_cycles=df_detected_cycles,
    figsize=(18, 6),
    hour_interval=1
)

if fig is not None:
    display(fig)
    plt.close(fig)

# COMMAND ----------

# DBTITLE 1,Example 3: Plot short period (2 hours)
# Exemplo 3: Plotar um período curto de 2 horas
print("\nEXEMPLO 3: Período curto de 2 horas (10:00 - 12:00)")
print("="*80)

fig = plot_inclination_cycles(
    df=df_numeric,
    var_name='ACI@FEA_GERAL_INCLINOM',
    start_date='2025-11-11 10:00:00',
    end_date='2025-11-11 12:00:00',
    start_threshold=3.0,
    end_threshold=-6.0,
    df_cycles=df_detected_cycles,
    figsize=(16, 6),
    hour_interval=1  # Intervalo de 30 minutos para período curto
)

if fig is not None:
    display(fig)
    plt.close(fig)

# COMMAND ----------

# DBTITLE 1,Plot energia FEA - Sistema (sólido) vs Operador (tracejado)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# 1) CONFIGURAÇÃO DO PERÍODO
# ============================================================
start_plot = pd.to_datetime("2025-11-01 00:00:00")
end_plot   = pd.to_datetime("2025-11-01 17:00:00")

print(f"Período selecionado: {start_plot} até {end_plot}")
print("="*80)

# ============================================================
# 2) PREPARAR DADOS DE ENERGIA (df_numeric)
# ============================================================
df_base = df_numeric.copy()
df_base["Time"] = pd.to_datetime(df_base["Time"])
df_base["ACI@FEA_ELET_ENERGIA"] = pd.to_numeric(df_base["ACI@FEA_ELET_ENERGIA"], errors="coerce")

# Filtrar período
mask = (df_base["Time"] >= start_plot) & (df_base["Time"] <= end_plot)
df_plot = df_base.loc[mask].copy()

print(f"Registros de energia no período: {len(df_plot):,}")

# ============================================================
# 3) PREPARAR DADOS DO OPERADOR (df_final - converter para pandas)
# ============================================================
try:
    # Converter Spark DataFrame para pandas
    df_final_pd = df_final.toPandas()
    
    # Converter colunas de timestamp
    df_final_pd["inicio_fea"] = pd.to_datetime(df_final_pd["inicio_fea"])
    df_final_pd["final_fea"] = pd.to_datetime(df_final_pd["final_fea"])
    
    # Filtrar corridas que se sobrepõem ao período
    mask_op = (
        (df_final_pd["inicio_fea"] <= end_plot) &
        (df_final_pd["final_fea"] >= start_plot)
    )
    df_op_plot = df_final_pd.loc[mask_op].copy()
    print(f"Corridas do operador no período: {len(df_op_plot)}")
except Exception as e:
    print(f"Aviso: Não foi possível carregar df_final - {e}")
    df_op_plot = pd.DataFrame()

# ============================================================
# 4) PREPARAR DADOS DO SISTEMA (df_sys_fea_final)
# ============================================================
try:
    df_sys_plot = df_sys_fea_final.copy()
    df_sys_plot["inicio_fea_sys"] = pd.to_datetime(df_sys_plot["inicio_fea_sys"])
    df_sys_plot["final_fea_sys"] = pd.to_datetime(df_sys_plot["final_fea_sys"])
    
    # Filtrar corridas que se sobrepõem ao período
    mask_sys = (
        (df_sys_plot["inicio_fea_sys"] <= end_plot) &
        (df_sys_plot["final_fea_sys"] >= start_plot)
    )
    df_sys_plot = df_sys_plot.loc[mask_sys].copy()
    print(f"Corridas do sistema no período: {len(df_sys_plot)}")
except Exception as e:
    print(f"Aviso: Não foi possível carregar df_sys_fea_final - {e}")
    df_sys_plot = pd.DataFrame()

print()

# ============================================================
# 5) CRIAR GRÁFICO
# ============================================================
fig, ax = plt.subplots(figsize=(18, 7))

# Plotar curva de energia
ax.plot(
    df_plot["Time"],
    df_plot["ACI@FEA_ELET_ENERGIA"],
    color="steelblue",
    linewidth=2,
    alpha=0.9,
    label="Energia FEA"
)

# Obter limites do eixo Y para posicionar os textos
y_min, y_max = ax.get_ylim()
y_range = y_max - y_min

# Adicionar linhas TRACEJADAS do OPERADOR com labels
for idx, row in df_op_plot.iterrows():
    corrida = int(row["corrida"])
    
    if pd.notna(row["inicio_fea"]):
        ax.axvline(
            row["inicio_fea"],
            color="green",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Início OP" if idx == df_op_plot.index[0] else ""
        )
        # Label do início OP
        ax.text(
            row["inicio_fea"],
            y_max - y_range * 0.05,
            f"OP-I\n{corrida}",
            rotation=0,
            va="top",
            ha="center",
            fontsize=8,
            color="green",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="green", alpha=0.8)
        )
    
    if pd.notna(row["final_fea"]):
        ax.axvline(
            row["final_fea"],
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Fim OP" if idx == df_op_plot.index[0] else ""
        )
        # Label do fim OP
        ax.text(
            row["final_fea"],
            y_max - y_range * 0.05,
            f"OP-F\n{corrida}",
            rotation=0,
            va="top",
            ha="center",
            fontsize=8,
            color="red",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8)
        )

# Adicionar linhas SÓLIDAS do SISTEMA com labels
for idx, row in df_sys_plot.iterrows():
    corrida = int(row["corrida"])
    
    if pd.notna(row["inicio_fea_sys"]):
        ax.axvline(
            row["inicio_fea_sys"],
            color="darkgreen",
            linestyle="-",
            linewidth=2.5,
            alpha=0.9,
            label="Início SYS" if idx == df_sys_plot.index[0] else ""
        )
        # Label do início SYS
        ax.text(
            row["inicio_fea_sys"],
            y_max - y_range * 0.15,
            f"SYS-I\n{corrida}",
            rotation=0,
            va="top",
            ha="center",
            fontsize=8,
            color="darkgreen",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkgreen", alpha=0.9)
        )
    
    if pd.notna(row["final_fea_sys"]):
        ax.axvline(
            row["final_fea_sys"],
            color="darkred",
            linestyle="-",
            linewidth=2.5,
            alpha=0.9,
            label="Fim SYS" if idx == df_sys_plot.index[0] else ""
        )
        # Label do fim SYS
        ax.text(
            row["final_fea_sys"],
            y_max - y_range * 0.15,
            f"SYS-F\n{corrida}",
            rotation=0,
            va="top",
            ha="center",
            fontsize=8,
            color="darkred",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkred", alpha=0.9)
        )

# ============================================================
# 6) FORMATAÇÃO
# ============================================================
ax.set_title(
    "FEA – Energia Elétrica\nOperador (tracejado) vs Sistema (sólido)",
    fontsize=15,
    fontweight="bold",
    pad=20
)

ax.set_xlabel("Tempo", fontsize=12)
ax.set_ylabel("Energia Elétrica", fontsize=12)

ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="upper left")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

print("\n✓ Gráfico criado com sucesso!")
print("\nLegenda:")
print("  - Linhas TRACEJADAS (--): Operador")
print("  - Linhas SÓLIDAS (─): Sistema")
print("  - Verde: Início da corrida")
print("  - Vermelho: Fim da corrida")
print("  - OP-I/OP-F: Início/Fim do Operador + número da corrida")
print("  - SYS-I/SYS-F: Início/Fim do Sistema + número da corrida")

# COMMAND ----------

# DBTITLE 1,Comparação Sistema vs Operador por corrida
import pandas as pd

print("Criando tabela de comparação Sistema vs Operador...")
print("="*80)

# ============================================================
# 1) PREPARAR DADOS DO SISTEMA (REFERÊNCIA)
# ============================================================
df_sys = df_sys_fea_final.copy()
df_sys["corrida"] = df_sys["corrida"].astype(int)
df_sys["inicio_sys"] = pd.to_datetime(df_sys["inicio_fea_sys"])
df_sys["final_sys"] = pd.to_datetime(df_sys["final_fea_sys"])

# Selecionar apenas as colunas necessárias
df_sys = df_sys[["corrida", "inicio_sys", "final_sys"]]

print(f"Corridas do sistema (referência): {len(df_sys)}")
print(f"Primeira corrida do sistema: {df_sys['corrida'].min()}")
print(f"Última corrida do sistema: {df_sys['corrida'].max()}")
print()

# ============================================================
# 2) PREPARAR DADOS DO OPERADOR
# ============================================================
try:
    # Converter Spark DataFrame para pandas se necessário
    if hasattr(df_final, 'toPandas'):
        df_op = df_final.toPandas()
    else:
        df_op = df_final.copy()
    
    df_op["corrida"] = df_op["corrida"].astype(int)
    df_op["inicio_op"] = pd.to_datetime(df_op["inicio_fea"])
    df_op["final_op"] = pd.to_datetime(df_op["final_fea"])
    
    # Selecionar apenas as colunas necessárias
    df_op = df_op[["corrida", "inicio_op", "final_op"]]
    
    print(f"Total de corridas do operador (histórico completo): {len(df_op):,}")
    
    # Filtrar operador apenas para as corridas que existem no sistema
    corridas_sys = df_sys["corrida"].unique()
    df_op_filtrado = df_op[df_op["corrida"].isin(corridas_sys)].copy()
    
    print(f"Corridas do operador que coincidem com o sistema: {len(df_op_filtrado)}")
    
except Exception as e:
    print(f"Erro ao carregar dados do operador: {e}")
    df_op_filtrado = pd.DataFrame(columns=["corrida", "inicio_op", "final_op"])

print()

# ============================================================
# 3) MERGE (LEFT JOIN - SISTEMA COMO REFERÊNCIA)
# ============================================================
df_comparacao = pd.merge(
    df_sys,
    df_op_filtrado,
    on="corrida",
    how="left"  # Mantém todas as corridas do sistema
).sort_values("corrida").reset_index(drop=True)

# Calcular diferenças em minutos
# diff_inicio = inicio_op - inicio_sys (positivo = operador atrasado)
df_comparacao["diff_inicio"] = (
    (df_comparacao["inicio_op"] - df_comparacao["inicio_sys"])
    .dt.total_seconds() / 60
)

# diff_final = final_op - final_sys (positivo = operador atrasado)
df_comparacao["diff_final"] = (
    (df_comparacao["final_op"] - df_comparacao["final_sys"])
    .dt.total_seconds() / 60
)

# Reordenar colunas
df_comparacao = df_comparacao[[
    "corrida",
    "inicio_sys",
    "inicio_op",
    "diff_inicio",
    "final_sys",
    "final_op",
    "diff_final"
]]

print(f"Total de corridas na comparação: {len(df_comparacao)}")

# Verificar quantas corridas do sistema têm correspondência no operador
com_match = df_comparacao["inicio_op"].notna().sum()
sem_match = df_comparacao["inicio_op"].isna().sum()

print(f"  - Com correspondência no operador: {com_match}")
print(f"  - Sem correspondência no operador: {sem_match}")
print()

# ============================================================
# 4) ESTATÍSTICAS (apenas para corridas com correspondência)
# ============================================================
df_comp_valido = df_comparacao.dropna(subset=["diff_inicio", "diff_final"])

if len(df_comp_valido) > 0:
    print("ESTATÍSTICAS DAS DIFERENÇAS (em minutos)")
    print("="*80)
    print(f"Baseado em {len(df_comp_valido)} corridas com correspondência\n")

    print("Diferença no INÍCIO (inicio_op - inicio_sys):")
    print(f"  Média: {df_comp_valido['diff_inicio'].mean():.2f} min")
    print(f"  Mediana: {df_comp_valido['diff_inicio'].median():.2f} min")
    print(f"  Desvio padrão: {df_comp_valido['diff_inicio'].std():.2f} min")
    print(f"  Mínimo: {df_comp_valido['diff_inicio'].min():.2f} min")
    print(f"  Máximo: {df_comp_valido['diff_inicio'].max():.2f} min")

    print("\nDiferença no FIM (final_op - final_sys):")
    print(f"  Média: {df_comp_valido['diff_final'].mean():.2f} min")
    print(f"  Mediana: {df_comp_valido['diff_final'].median():.2f} min")
    print(f"  Desvio padrão: {df_comp_valido['diff_final'].std():.2f} min")
    print(f"  Mínimo: {df_comp_valido['diff_final'].min():.2f} min")
    print(f"  Máximo: {df_comp_valido['diff_final'].max():.2f} min")

    print("\n" + "="*80)
    print("\nInterpretação:")
    print("  - diff_inicio > 0: Operador registrou início DEPOIS do sistema")
    print("  - diff_inicio < 0: Operador registrou início ANTES do sistema")
    print("  - diff_final > 0: Operador registrou fim DEPOIS do sistema")
    print("  - diff_final < 0: Operador registrou fim ANTES do sistema")
    print()
else:
    print("⚠ Nenhuma corrida com correspondência encontrada!\n")

# ============================================================
# 5) EXIBIR TABELA
# ============================================================
print("\nTABELA DE COMPARAÇÃO (Sistema como referência):")
print("="*80)
display(df_comparacao)

print("\n✓ Tabela de comparação criada com sucesso!")
print("\nNota: Apenas corridas detectadas pelo SISTEMA são incluídas na comparação.")
print("      Corridas sem correspondência no operador terão valores NaN/null.")

# COMMAND ----------

# DBTITLE 1,Histogramas de análise - Sistema vs Operador
import matplotlib.pyplot as plt
import numpy as np

print("Criando histogramas de análise...")
print("="*80)

# Filtrar apenas corridas válidas (com correspondência)
df_valid = df_comparacao.dropna(subset=["diff_inicio", "diff_final"]).copy()

print(f"Analisando {len(df_valid)} corridas com dados completos\n")

# Calcular durações
df_valid["duracao_sys"] = (
    (df_valid["final_sys"] - df_valid["inicio_sys"]).dt.total_seconds() / 60
)
df_valid["duracao_op"] = (
    (df_valid["final_op"] - df_valid["inicio_op"]).dt.total_seconds() / 60
)
df_valid["diff_duracao"] = df_valid["duracao_op"] - df_valid["duracao_sys"]

# Calcular diferenças absolutas
df_valid["abs_diff_inicio"] = df_valid["diff_inicio"].abs()
df_valid["abs_diff_final"] = df_valid["diff_final"].abs()

# ============================================================
# CRIAR FIGURA COM SUBPLOTS
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Análise de Diferenças: Sistema vs Operador", fontsize=16, fontweight="bold", y=0.995)

# ============================================================
# 1) DIFERENÇA NO INÍCIO (com sinal)
# ============================================================
ax = axes[0, 0]
ax.hist(df_valid["diff_inicio"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero (sincronia perfeita)")
ax.axvline(df_valid["diff_inicio"].median(), color="green", linestyle="-", linewidth=2, 
           label=f"Mediana: {df_valid['diff_inicio'].median():.2f} min")
ax.set_xlabel("Diferença (minutos)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.set_title("Diferença no INÍCIO (inicio_op - inicio_sys)\nNegativo = Operador ANTES do Sistema", 
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# 2) DIFERENÇA NO FIM (com sinal)
# ============================================================
ax = axes[0, 1]
ax.hist(df_valid["diff_final"], bins=50, edgecolor="black", alpha=0.7, color="coral")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero (sincronia perfeita)")
ax.axvline(df_valid["diff_final"].median(), color="green", linestyle="-", linewidth=2,
           label=f"Mediana: {df_valid['diff_final'].median():.2f} min")
ax.set_xlabel("Diferença (minutos)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.set_title("Diferença no FIM (final_op - final_sys)\nNegativo = Operador ANTES do Sistema", 
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# 3) DIFERENÇA ABSOLUTA NO INÍCIO
# ============================================================
ax = axes[1, 0]
ax.hist(df_valid["abs_diff_inicio"], bins=50, edgecolor="black", alpha=0.7, color="lightblue")
ax.axvline(df_valid["abs_diff_inicio"].median(), color="green", linestyle="-", linewidth=2,
           label=f"Mediana: {df_valid['abs_diff_inicio'].median():.2f} min")
ax.set_xlabel("Diferença Absoluta (minutos)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.set_title("Diferença ABSOLUTA no Início\n(Magnitude do desvio, sem considerar direção)", 
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# 4) DIFERENÇA ABSOLUTA NO FIM
# ============================================================
ax = axes[1, 1]
ax.hist(df_valid["abs_diff_final"], bins=50, edgecolor="black", alpha=0.7, color="lightsalmon")
ax.axvline(df_valid["abs_diff_final"].median(), color="green", linestyle="-", linewidth=2,
           label=f"Mediana: {df_valid['abs_diff_final'].median():.2f} min")
ax.set_xlabel("Diferença Absoluta (minutos)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.set_title("Diferença ABSOLUTA no Fim\n(Magnitude do desvio, sem considerar direção)", 
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# 5) DURAÇÃO DAS CORRIDAS (Sistema vs Operador)
# ============================================================
ax = axes[2, 0]
ax.hist(df_valid["duracao_sys"], bins=40, alpha=0.6, label="Sistema", color="darkgreen", edgecolor="black")
ax.hist(df_valid["duracao_op"], bins=40, alpha=0.6, label="Operador", color="green", edgecolor="black")
ax.axvline(df_valid["duracao_sys"].median(), color="darkgreen", linestyle="--", linewidth=2)
ax.axvline(df_valid["duracao_op"].median(), color="green", linestyle="--", linewidth=2)
ax.set_xlabel("Duração (minutos)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.set_title(f"Duração das Corridas\nSistema: {df_valid['duracao_sys'].median():.1f} min (mediana) | Operador: {df_valid['duracao_op'].median():.1f} min (mediana)", 
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# 6) DIFERENÇA NA DURAÇÃO
# ============================================================
ax = axes[2, 1]
ax.hist(df_valid["diff_duracao"], bins=50, edgecolor="black", alpha=0.7, color="purple")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero (mesma duração)")
ax.axvline(df_valid["diff_duracao"].median(), color="green", linestyle="-", linewidth=2,
           label=f"Mediana: {df_valid['diff_duracao'].median():.2f} min")
ax.set_xlabel("Diferença (minutos)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.set_title("Diferença na DURAÇÃO (duracao_op - duracao_sys)\nPositivo = Operador registrou corrida mais longa", 
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# ESTATÍSTICAS RESUMIDAS
# ============================================================
print("\n" + "="*80)
print("RESUMO ESTATÍSTICO")
print("="*80)

print("\n1) DIFERENÇA NO INÍCIO (inicio_op - inicio_sys):")
print(f"   Média: {df_valid['diff_inicio'].mean():.2f} min")
print(f"   Mediana: {df_valid['diff_inicio'].median():.2f} min")
print(f"   Desvio padrão: {df_valid['diff_inicio'].std():.2f} min")
print(f"   Diferença absoluta média: {df_valid['abs_diff_inicio'].mean():.2f} min")

print("\n2) DIFERENÇA NO FIM (final_op - final_sys):")
print(f"   Média: {df_valid['diff_final'].mean():.2f} min")
print(f"   Mediana: {df_valid['diff_final'].median():.2f} min")
print(f"   Desvio padrão: {df_valid['diff_final'].std():.2f} min")
print(f"   Diferença absoluta média: {df_valid['abs_diff_final'].mean():.2f} min")

print("\n3) DURAÇÃO DAS CORRIDAS:")
print(f"   Sistema - Média: {df_valid['duracao_sys'].mean():.2f} min | Mediana: {df_valid['duracao_sys'].median():.2f} min")
print(f"   Operador - Média: {df_valid['duracao_op'].mean():.2f} min | Mediana: {df_valid['duracao_op'].median():.2f} min")
print(f"   Diferença média: {df_valid['diff_duracao'].mean():.2f} min")

print("\n4) OUTLIERS (diferenças extremas):")
outliers_inicio = df_valid[df_valid["abs_diff_inicio"] > 30]
outliers_final = df_valid[df_valid["abs_diff_final"] > 10]
print(f"   Corridas com |diff_inicio| > 30 min: {len(outliers_inicio)} ({len(outliers_inicio)/len(df_valid)*100:.1f}%)")
print(f"   Corridas com |diff_final| > 10 min: {len(outliers_final)} ({len(outliers_final)/len(df_valid)*100:.1f}%)")

if len(outliers_inicio) > 0:
    print(f"\n   Maiores desvios no INÍCIO:")
    top_outliers = df_valid.nlargest(5, "abs_diff_inicio")[["corrida", "diff_inicio", "abs_diff_inicio"]]
    for _, row in top_outliers.iterrows():
        print(f"      Corrida {int(row['corrida'])}: {row['diff_inicio']:.2f} min")

print("\n" + "="*80)
print("✓ Análise de histogramas concluída!")

# COMMAND ----------

# DBTITLE 1,Histogramas clipados (5%-95%) - Sem outliers extremos
import matplotlib.pyplot as plt
import numpy as np

print("Criando histogramas com dados clipados (5%-95%)...")
print("="*80)

# Filtrar apenas corridas válidas (com correspondência)
df_valid = df_comparacao.dropna(subset=["diff_inicio", "diff_final"]).copy()

print(f"Analisando {len(df_valid)} corridas com dados completos\n")

# Calcular durações
df_valid["duracao_sys"] = (
    (df_valid["final_sys"] - df_valid["inicio_sys"]).dt.total_seconds() / 60
)
df_valid["duracao_op"] = (
    (df_valid["final_op"] - df_valid["inicio_op"]).dt.total_seconds() / 60
)
df_valid["diff_duracao"] = df_valid["duracao_op"] - df_valid["duracao_sys"]

# Calcular diferenças absolutas
df_valid["abs_diff_inicio"] = df_valid["diff_inicio"].abs()
df_valid["abs_diff_final"] = df_valid["diff_final"].abs()

# ============================================================
# CLIPAR DADOS ENTRE 5% E 95%
# ============================================================
print("Calculando percentis para clipagem...")
print("="*80)

# Percentis para diff_inicio
p5_inicio = df_valid["diff_inicio"].quantile(0.05)
p95_inicio = df_valid["diff_inicio"].quantile(0.95)
print(f"diff_inicio: P5 = {p5_inicio:.2f} min, P95 = {p95_inicio:.2f} min")

# Percentis para diff_final
p5_final = df_valid["diff_final"].quantile(0.05)
p95_final = df_valid["diff_final"].quantile(0.95)
print(f"diff_final: P5 = {p5_final:.2f} min, P95 = {p95_final:.2f} min")

# Percentis para durações
p5_dur_sys = df_valid["duracao_sys"].quantile(0.05)
p95_dur_sys = df_valid["duracao_sys"].quantile(0.95)
p5_dur_op = df_valid["duracao_op"].quantile(0.05)
p95_dur_op = df_valid["duracao_op"].quantile(0.95)
print(f"duracao_sys: P5 = {p5_dur_sys:.2f} min, P95 = {p95_dur_sys:.2f} min")
print(f"duracao_op: P5 = {p5_dur_op:.2f} min, P95 = {p95_dur_op:.2f} min")

# Percentis para diff_duracao
p5_diff_dur = df_valid["diff_duracao"].quantile(0.05)
p95_diff_dur = df_valid["diff_duracao"].quantile(0.95)
print(f"diff_duracao: P5 = {p5_diff_dur:.2f} min, P95 = {p95_diff_dur:.2f} min")

# Clipar os dados
df_clipped = df_valid.copy()
df_clipped["diff_inicio_clip"] = df_clipped["diff_inicio"].clip(p5_inicio, p95_inicio)
df_clipped["diff_final_clip"] = df_clipped["diff_final"].clip(p5_final, p95_final)
df_clipped["duracao_sys_clip"] = df_clipped["duracao_sys"].clip(p5_dur_sys, p95_dur_sys)
df_clipped["duracao_op_clip"] = df_clipped["duracao_op"].clip(p5_dur_op, p95_dur_op)
df_clipped["diff_duracao_clip"] = df_clipped["diff_duracao"].clip(p5_diff_dur, p95_diff_dur)

# Contar quantos valores foram clipados
clipped_inicio = ((df_valid["diff_inicio"] < p5_inicio) | (df_valid["diff_inicio"] > p95_inicio)).sum()
clipped_final = ((df_valid["diff_final"] < p5_final) | (df_valid["diff_final"] > p95_final)).sum()
clipped_duracao = ((df_valid["diff_duracao"] < p5_diff_dur) | (df_valid["diff_duracao"] > p95_diff_dur)).sum()

print(f"\nValores clipados (removidos dos extremos):")
print(f"  diff_inicio: {clipped_inicio} corridas ({clipped_inicio/len(df_valid)*100:.1f}%)")
print(f"  diff_final: {clipped_final} corridas ({clipped_final/len(df_valid)*100:.1f}%)")
print(f"  diff_duracao: {clipped_duracao} corridas ({clipped_duracao/len(df_valid)*100:.1f}%)")
print()

# ============================================================
# CRIAR FIGURA COM SUBPLOTS
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Análise de Diferenças (Clipado 5%-95%): Sistema vs Operador\nSem outliers extremos", 
             fontsize=16, fontweight="bold", y=0.995)

# ============================================================
# 1) DIFERENÇA NO INÍCIO (clipado)
# ============================================================
ax = axes[0, 0]
ax.hist(df_clipped["diff_inicio_clip"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero (sincronia perfeita)")
ax.axvline(df_clipped["diff_inicio_clip"].median(), color="green", linestyle="-", linewidth=2, 
           label=f"Mediana: {df_clipped['diff_inicio_clip'].median():.2f} min")
ax.set_xlabel("Diferença (minutos)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.set_title(f"Diferença no INÍCIO (clipado P5-P95)\nNegativo = Operador ANTES | {clipped_inicio} outliers removidos", 
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# 2) DIFERENÇA NO FIM (clipado)
# ============================================================
ax = axes[0, 1]
ax.hist(df_clipped["diff_final_clip"], bins=50, edgecolor="black", alpha=0.7, color="coral")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero (sincronia perfeita)")
ax.axvline(df_clipped["diff_final_clip"].median(), color="green", linestyle="-", linewidth=2,
           label=f"Mediana: {df_clipped['diff_final_clip'].median():.2f} min")
ax.set_xlabel("Diferença (minutos)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.set_title(f"Diferença no FIM (clipado P5-P95)\nNegativo = Operador ANTES | {clipped_final} outliers removidos", 
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# 3) DIFERENÇA ABSOLUTA NO INÍCIO (usando dados clipados)
# ============================================================
ax = axes[1, 0]
abs_diff_inicio_clip = df_clipped["diff_inicio_clip"].abs()
ax.hist(abs_diff_inicio_clip, bins=50, edgecolor="black", alpha=0.7, color="lightblue")
ax.axvline(abs_diff_inicio_clip.median(), color="green", linestyle="-", linewidth=2,
           label=f"Mediana: {abs_diff_inicio_clip.median():.2f} min")
ax.set_xlabel("Diferença Absoluta (minutos)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.set_title("Diferença ABSOLUTA no Início (clipado)\n(Magnitude do desvio)", 
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# 4) DIFERENÇA ABSOLUTA NO FIM (usando dados clipados)
# ============================================================
ax = axes[1, 1]
abs_diff_final_clip = df_clipped["diff_final_clip"].abs()
ax.hist(abs_diff_final_clip, bins=50, edgecolor="black", alpha=0.7, color="lightsalmon")
ax.axvline(abs_diff_final_clip.median(), color="green", linestyle="-", linewidth=2,
           label=f"Mediana: {abs_diff_final_clip.median():.2f} min")
ax.set_xlabel("Diferença Absoluta (minutos)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.set_title("Diferença ABSOLUTA no Fim (clipado)\n(Magnitude do desvio)", 
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# 5) DURAÇÃO DAS CORRIDAS (clipado)
# ============================================================
ax = axes[2, 0]
ax.hist(df_clipped["duracao_sys_clip"], bins=40, alpha=0.6, label="Sistema", color="darkgreen", edgecolor="black")
ax.hist(df_clipped["duracao_op_clip"], bins=40, alpha=0.6, label="Operador", color="green", edgecolor="black")
ax.axvline(df_clipped["duracao_sys_clip"].median(), color="darkgreen", linestyle="--", linewidth=2)
ax.axvline(df_clipped["duracao_op_clip"].median(), color="green", linestyle="--", linewidth=2)
ax.set_xlabel("Duração (minutos)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.set_title(f"Duração das Corridas (clipado P5-P95)\nSistema: {df_clipped['duracao_sys_clip'].median():.1f} min | Operador: {df_clipped['duracao_op_clip'].median():.1f} min", 
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# 6) DIFERENÇA NA DURAÇÃO (clipado)
# ============================================================
ax = axes[2, 1]
ax.hist(df_clipped["diff_duracao_clip"], bins=50, edgecolor="black", alpha=0.7, color="purple")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero (mesma duração)")
ax.axvline(df_clipped["diff_duracao_clip"].median(), color="green", linestyle="-", linewidth=2,
           label=f"Mediana: {df_clipped['diff_duracao_clip'].median():.2f} min")
ax.set_xlabel("Diferença (minutos)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.set_title(f"Diferença na DURAÇÃO (clipado P5-P95)\nPositivo = Operador mais longo | {clipped_duracao} outliers removidos", 
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# ESTATÍSTICAS RESUMIDAS (DADOS CLIPADOS)
# ============================================================
print("\n" + "="*80)
print("RESUMO ESTATÍSTICO (DADOS CLIPADOS 5%-95%)")
print("="*80)

print("\n1) DIFERENÇA NO INÍCIO (clipado):")
print(f"   Média: {df_clipped['diff_inicio_clip'].mean():.2f} min")
print(f"   Mediana: {df_clipped['diff_inicio_clip'].median():.2f} min")
print(f"   Desvio padrão: {df_clipped['diff_inicio_clip'].std():.2f} min")
print(f"   Range: [{p5_inicio:.2f}, {p95_inicio:.2f}] min")

print("\n2) DIFERENÇA NO FIM (clipado):")
print(f"   Média: {df_clipped['diff_final_clip'].mean():.2f} min")
print(f"   Mediana: {df_clipped['diff_final_clip'].median():.2f} min")
print(f"   Desvio padrão: {df_clipped['diff_final_clip'].std():.2f} min")
print(f"   Range: [{p5_final:.2f}, {p95_final:.2f}] min")

print("\n3) DURAÇÃO DAS CORRIDAS (clipado):")
print(f"   Sistema - Média: {df_clipped['duracao_sys_clip'].mean():.2f} min | Mediana: {df_clipped['duracao_sys_clip'].median():.2f} min")
print(f"   Operador - Média: {df_clipped['duracao_op_clip'].mean():.2f} min | Mediana: {df_clipped['duracao_op_clip'].median():.2f} min")
print(f"   Diferença média: {df_clipped['diff_duracao_clip'].mean():.2f} min")

print("\n4) COMPARAÇÃO COM DADOS ORIGINAIS:")
print(f"   diff_inicio - Original: {df_valid['diff_inicio'].mean():.2f} min → Clipado: {df_clipped['diff_inicio_clip'].mean():.2f} min")
print(f"   diff_final - Original: {df_valid['diff_final'].mean():.2f} min → Clipado: {df_clipped['diff_final_clip'].mean():.2f} min")
print(f"   diff_duracao - Original: {df_valid['diff_duracao'].mean():.2f} min → Clipado: {df_clipped['diff_duracao_clip'].mean():.2f} min")

print("\n" + "="*80)
print("✓ Análise de histogramas clipados concluída!")
print("\nNota: Clipagem remove os 5% menores e 5% maiores valores,")
print("      focando na distribuição central (90% dos dados).")