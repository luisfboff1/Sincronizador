# Databricks notebook source
!pip install summarytools --upgrade

# COMMAND ----------

import datetime  # Basic date and time types
import functools  # Higher-order functions and operations on callable objects
import inspect  # Inspect live objects
import math  # Mathematical functions
import os  # Miscellaneous operating system interfaces
import re  # Regular expression operations
import sys  # System-specific parameters and functions
import time  # Time access and conversions
import typing  # Support for type hints
from typing import Callable, Literal, Optional, Union
import zoneinfo  # IANA time zone support

# display handles pyspark DataFrame, but not pandas DataFrame in some versions
from IPython.display import display as display_pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.sql.types import TimestampType
import summarytools

# COMMAND ----------

def print_grouped_df_columns_names(
    df: pd.DataFrame | pyspark.sql.dataframe.DataFrame, n_first_chars_to_group: int = 2
) -> None:
    last_first_letters = ""
    columns = sorted(df.columns)
    for column in columns:
        if last_first_letters == column[0:n_first_chars_to_group]:
            print(column, end=", ")
        else:
            print("\n" + column, end=", ")
        last_first_letters = column[0:n_first_chars_to_group]
    print(f"\n{len(columns)} columns")


def inspect_df(
    df: pd.DataFrame, 
    id_col: str = 'corrida', 
    date_col: Optional[str] = 'datasaidafp',
    date_format: str = '%Y-%m-%d'
) -> None:
    """
    Performs sanity checks on the DataFrame (duplicates) and prints a summary 
    of the ID range and Date range.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        id_col (str): The name of the identifier column (e.g., run/corrida ID). 
                      Defaults to 'corrida'.
        date_col (Optional[str]): The name of the date column. If provided, 
                                  min/max dates will be printed. Defaults to 'datasaidafp'.
        date_format (str): The output format for dates. Defaults to '%Y-%m-%d'.

    Raises:
        ValueError: If the DataFrame contains duplicate column names (sanity check).
        KeyError: If id_col is not found in the DataFrame.
    """
    
    # --- 1. Sanity Checks ---
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"🚨 CRITICAL: Duplicate column names detected: {dups}. Please deduplicate before proceeding.")

    # Warn about index duplicates
    if df.index.duplicated().any():
        print(f"\n⚠️  WARNING: Found {df.index.duplicated().sum()} duplicate index values.")

    if id_col not in df.columns:
        raise KeyError(f"Column '{id_col}' not found in DataFrame.")

    # --- 2. Data Preparation ---
    # Calculate ID stats
    min_id = df[id_col].min()
    max_id = df[id_col].max()
    
    # Calculate Date stats (handle both string and datetime objects)
    min_date_str = ""
    max_date_str = ""
    
    if date_col and date_col in df.columns:
        # We convert to datetime temporarily to ensure .min()/.max() works chronologically, 
        # not alphabetically (which happens with strings).
        # 'dayfirst=True' assumes DD/MM/YYYY format if parsing strings.
        temp_dates = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        
        if temp_dates.notna().any():
            min_date_str = f"({temp_dates.min().strftime(date_format)})"
            max_date_str = f"({temp_dates.max().strftime(date_format)})"
        else:
            min_date_str = "(Date parse error)"
            max_date_str = "(Date parse error)"

    # --- 3. Output ---
    print("-" * 60)
    print("DATASET INSPECTION SUMMARY")
    print("-" * 60)
    print("In the current dataset:")
    print(f"\t• Oldest {id_col}: {min_id} {min_date_str}")
    print(f"\t• Newest {id_col}: {max_id} {max_date_str}")
    print(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    print("-" * 60)


def custom_pd_display(
    df: pd.DataFrame, cols_to_show_first: Optional[dict[str, str]] = None, precision: int = 4
    ) -> None:
    """
    Custom function to display a DataFrame with:
    - All columns shown, replacing '_' by ' ' in column names to make it more readable
    - Float columns to the specified number of decimal places, but shown as integers
     if no decimals exist.
    - Integer columns displayed as integers.

    Parameters:
        df: The pandas DataFrame to display.
        cols_to_show_first: mapping of columns. These columns will be shown first.
        precision: The number of decimal places to show for float columns (default is 4).
    """
    # Custom function to format floats to 4 decimals and leave integers unchanged
    def format_floats(x):
        if isinstance(x, float):
            # Check if the float is essentially an integer
            if x.is_integer():
                return int(x)  # Display as integer
            else:
                return f"{x:.{precision}f}"  # Format with specified decimal places
        elif isinstance(x, list) and all(isinstance(i, float) for i in x):
            return [f"{i:.{precision}f}" for i in x]
        else:
            return x  # For non-float types, return the value as is
    
    df_shown = df.copy()

    if cols_to_show_first:
        # Reorder columns to have most important info before
        df_shown = df_shown[
            list([column for column in cols_to_show_first.keys() if column in df_shown.columns])
            + [column for column in df_shown.columns if column not in cols_to_show_first.keys()]
        ]
        df_shown = df_shown.rename(columns=cols_to_show_first)

    # Temporarily set display options for showing all columns and wide strings
    with pd.option_context(
        'display.max_columns', None, 'display.max_colwidth', None, 'display.float_format', '{:,.{precision}f}'.format
        ):
        # Replace '_' by ' ' to make it more readable
        df_shown.columns = [column.replace('_', ' ') for column in df_shown.columns]        
        # Apply the custom formatting function to the entire DataFrame
        display_pd(df_shown.applymap(format_floats))


def format_time_column(
    df: pyspark.sql.DataFrame, column_name: str, result_column_name: Optional[str] = None
    ) -> pyspark.sql.DataFrame:
    """
    This function takes a dataframe and a column name, and transforms the time data
    in the given column into the HH:MM format.
    If the value is in HHMM format, it will insert a colon between hours and minutes.
    
    df (pyspark.sql.DataFrame): The input dataframe
    column_name (str): The name of the column containing the time data
    result_column_name (str): The name of the result column containing the time data
    return: A pyspark.sql.DataFrame with the transformed column
    """
    if result_column_name is None:
        result_column_name = column_name

    return df.withColumn(
        f"{result_column_name}",
        F.when(
            # Checks if the value is in HHMM format (4 digits)
            F.col(column_name).rlike(r"^\d{4}$"),
            F.concat(
                F.col(column_name).substr(1, 2),
                F.lit(":"),
                F.col(column_name).substr(3, 2)
            )
        ).otherwise(
            # If already in HH:MM format, leave it unchanged
            F.col(column_name)
        )
    )

def assert_dataframe_equal(
    df_actual: pyspark.sql.DataFrame, data_expected: list, schema: list, order_col: str = None
    ):
    """
    Helper to compare actual DataFrame content with expected list of data.
    """
    if order_col:
        df_actual = df_actual.orderBy(order_col)
    
    actual_data = [tuple(row) for row in df_actual.collect()]
    
    if actual_data != data_expected:
        print("\n>>> EXPECTED DATA:")
        print(data_expected)
        print("\n>>> ACTUAL DATA:")
        print(actual_data)
    
    assert actual_data == data_expected
    

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType
from typing import Optional
 
def fill_nulls_with_last_valid(
    df: pyspark.sql.DataFrame, timestamp_column: str, columns: list[str]
    ) -> pyspark.sql.DataFrame:
    """
    Fill null values in specified columns with the last valid (non-null) value in each column.
    Args:
        df (DataFrame): Spark DataFrame to process.
        columns (list): List of column names to fill nulls.
    Returns:
        DataFrame: Updated DataFrame with nulls filled.
    """
    for column in columns:
        # Define a window specification ordered by the timestamp column
        window_spec = Window.orderBy(timestamp_column).rowsBetween(Window.unboundedPreceding, 0)
        # Build a list of expressions for the target columns
        # We construct the logic: If Col is Null, take Last(Col) over Window, else Col
        # This happens in the Spark plan definition, not during data execution.
        select_exprs = []
       
        # Add non-target columns to the selection list to preserve them
        target_cols_set = set(columns)
        for col_name in df.columns:
            if col_name in target_cols_set:
                # The logic: last(col, ignorenulls=True)
                select_exprs.append(
                    F.last(F.col(col_name), ignorenulls=True).over(window_spec).alias(col_name)
                )
            else:
                select_exprs.append(F.col(col_name))
 
        # Apply all transformations in a single pass
        return df.select(*select_exprs)
    # for column in columns:
    #     # Define a window specification ordered by the timestamp column
    #     window_spec = Window.orderBy(timestamp_column).rowsBetween(Window.unboundedPreceding, 0)
    #     # Create a new column with the last non-null value propagated forward
    #     df = df.withColumn(f'{column}_filled', F.last(F.col(column), ignorenulls=True).over(window_spec))
    #     # Replace original nulls with the filled values
    #     df = df.withColumn(column, F.coalesce(F.col(column), F.col(f'{column}_filled')))
    #     # Drop the auxiliary filled column
    #     df = df.drop(f'{column}_filled')
    # return df
 
def test_fill_nulls_with_last_valid():
    print("Running: test_fill_nulls_with_last_valid...", end=" ")
 
    # Scenario: Intermittent nulls should be filled with previous valid values.
    data = [
        (1, 10, "Unchanged"),
        (2, None, "Filled"),  # Should become 10
        (3, 20, "Unchanged"),
        (4, None, "Filled")   # Should become 20
    ]
    schema = ["timestamp", "value", "class"]
    df = spark.createDataFrame(data, schema)
 
    df_result = fill_nulls_with_last_valid(df, "timestamp", ["value"])
   
    expected_data = [
        (1, 10, "Unchanged"),
        (2, 10, "Filled"),
        (3, 20, "Unchanged"),
        (4, 20, "Filled")
    ]
   
    assert_dataframe_equal(df_result, expected_data, schema, order_col="timestamp")
    print("✅ Passed")
 
# test_fill_nulls_with_last_valid()
 
def fill_first_row_with_first_valid(
    df: pyspark.sql.DataFrame, columns: list[str]
    ) -> pyspark.sql.DataFrame:
    """
    Fill the first row of specified columns with the first valid (non-null) value in each column.
    Args:
        df (DataFrame): Spark DataFrame to process.
        columns (list): List of column names to fill.
    Returns:
        DataFrame: Updated DataFrame with the first row filled.
    """
    # Build a list of expressions for the target columns
    # We construct the logic: If Col is Null, take Last(Col) over Window, else Col
    # This happens in the Spark plan definition, not during data execution.
    select_exprs = []
 
    # Add non-target columns to the selection list to preserve them
    target_cols_set = set(columns)
    for col_name in df.columns:
        if col_name in target_cols_set:
            # Get the first valid (non-null) value for the column
            first_valid_value = df.filter(F.col(col_name).isNotNull()).select(col_name).first()[0]
            # Fill the first row with the first valid value
            select_exprs.append(
                F.when(
                    F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) == 1,
                    first_valid_value
                ).otherwise(
                    F.col(col_name)
                ).alias(col_name)
            )
        else:
            select_exprs.append(F.col(col_name))
 
    # 3. Apply all transformations in a single pass
    return df.select(*select_exprs)
 
    # for column in columns:
    #     # Get the first valid (non-null) value for the column
    #     first_valid_value = df.filter(F.col(column).isNotNull()).select(column).first()[0]
    #     # Fill the first row with the first valid value
    #     df = df.withColumn(
    #         column,
    #         F.when(F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) == 1, first_valid_value)
    #         .otherwise(F.col(column))
    #     )
    # return df
 
def test_fill_first_row_with_first_valid():
    print("Running: test_fill_first_row_with_first_valid...", end=" ")
 
    # Scenario: First row is Null. It should take the first valid value (50).
    data = [
        (None, "A", "Filled"),
        (50, "B", "Unchanged"),
        (None, "C", "Unchanged")
    ]
    schema = ["val", "id", "class"]
    df = spark.createDataFrame(data, schema)
 
    df_result = fill_first_row_with_first_valid(df, ["val"])
   
    expected_data = [
        (50, "A", "Filled"),     # Filled
        (50, "B", "Unchanged"),  # Unchanged
        (None, "C", "Unchanged") # Unchanged (not the first row)
    ]
   
    # Manual extraction for assertion to handle unordered local comparison easily
    rows = df_result.collect()
    result_tuples = [(r['val'], r['id'], r['class']) for r in rows]
   
    assert result_tuples == expected_data
    print("✅ Passed")
 
# test_fill_first_row_with_first_valid()
 
 
def repair_data_glitches(
    df: pyspark.sql.DataFrame,
    column: str,
    active_threshold: int = 1,
    lookahead_steps: int = 3
) -> pyspark.sql.DataFrame:
    """
    Identifies sudden drops to zero that return to a high value quickly (glitches)
    and repairs them using a forward fill strategy.
 
    This function distinguishes between a real 'end of cycle' (where data drops
    to zero and stays zero) and sensor noise (where data drops to zero but
    bounces back immediately).
 
    Args:
        df (pyspark.sql.DataFrame): The input PySpark DataFrame containing time-series data.
        column (str): The name of the data column to be repaired.
        active_threshold (int): The minimum data value to consider the state high/'active'.
                                Values below this (e.g., < 1) are treated as potential zeros.
                                Defaults to 1.
        lookahead_steps (int): The number of future rows (steps) to check to confirm
                               if the data returns to a high level.
                               Defaults to 3.
 
    Returns:
        pyspark.sql.DataFrame: A new DataFrame with the specified column repaired,
                               replacing glitch zeros with the last valid value.
    """
   
    # Define Windows for time-series analysis
    # w_lag: Used to check the PREVIOUS state (was it active before the drop?)
    w_lag = Window.orderBy("timestamp")
   
    # w_lookahead: Used to check the FUTURE state (does it become active again soon?)
    # We look from the next row (1) up to 'lookahead_steps' rows ahead.
    w_lookahead = Window.orderBy("timestamp").rowsBetween(1, lookahead_steps)
   
    # w_ffill: Used to propagate the last valid value forward to fill nulls (glitches)
    w_ffill = Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, Window.currentRow)
   
    # 1. Identify the Glitch
    # Condition:
    #   (a) Current value is effectively zero (e.g., < 1)
    #   (b) Previous value was active (High) -> Indicates a drop occurred
    #   (c) Future max value (within lookahead) is active (High) -> Indicates it's just a temporary drop
    is_glitch_cond = (
        (F.col(column) < 1) &
        (F.lag(column, 1, 0).over(w_lag) > active_threshold) &
        (F.max(column).over(w_lookahead) > active_threshold)
    )
   
    # 2. Create a temporary column where Glitches are masked as NULL
    # If it's a glitch, we set it to NULL so we can fill it later.
    # Otherwise, keep the original value.
    col_clean = f"{column}_temp_clean"
    df_masked = df.select(
        "*",
        F.when(is_glitch_cond, F.lit(None)).otherwise(F.col(column)).alias(col_clean)
    )
   
    # 3. Apply Forward Fill
    # Use the 'last' function with ignorenulls=True to carry over the previous valid value
    # covering the gap created by the glitch.
    df_fixed = df_masked.withColumn(
        column, # Overwrite original column with repaired data
        F.last(col_clean, ignorenulls=True).over(w_ffill)
    ).drop(col_clean)
    # df_fixed = df_masked.select(
    #     *[c for c in df_masked.columns if c != column and c != col_clean],
    #     F.last(col_clean, ignorenulls=True).over(w_ffill).alias(column)
    # )
   
    return df_fixed
 
def test_repair_data_glitches():
    print("Running: test_repair_data_glitches...", end=" ")
 
    # Scenario:
    # ts=20 is a GLITCH (drops to 0, returns high at ts=30). Should be repaired.
    # ts=50 is a REAL SHUTDOWN (drops to 0, stays 0). Should remain 0.
    data = [
        (10, 100),
        (20, 0),   # Glitch -> Expect 100
        (30, 100),
        (40, 100),
        (50, 0),   # Shutdown -> Expect 0
        (60, 0)
    ]
    schema = ["timestamp", "sensor"]
    df = spark.createDataFrame(data, schema)
 
    df_result = repair_data_glitches(df, "sensor", active_threshold=1)
   
    expected_data = [
        (10, 100),
        (20, 100), # Repaired
        (30, 100),
        (40, 100),
        (50, 0),   # Kept as is
        (60, 0)
    ]
   
    assert_dataframe_equal(df_result, expected_data, schema, order_col="timestamp")
    print("✅ Passed")
 
# test_repair_data_glitches()
 
def preprocess_pims_data(
    spark_df: pyspark.sql.DataFrame,
    first_timestamp: str,
    last_timestamp: str,
    timestamp_column: str,
    columns_to_fill: list[str],
) -> pyspark.sql.DataFrame:
    """
    Filters a Spark DataFrame by timestamp range and fills missing values
 
    Parameters
    ----------
    spark_df : pyspark.sql.DataFrame
        Input Spark DataFrame containing unstructured process data
        (timeseries OPC / PIMS data).
 
    first_timestamp : str
        Lower bound of the filtering time window (inclusive).
        Must be a string convertible to pandas.Timestamp.
 
    last_timestamp : str
        Upper bound of the filtering time window (inclusive).
 
    timestamp_column : str
        Name of the timestamp column used to filter the Spark DataFrame.
 
    columns_to_fill : list[str]
        List of columns whose missing values should be forward-filled
        and then backwards-filled (first valid).
 
    Returns
    -------
    pyspark.sql.DataFrame
        Processed Spark DataFrame filtered to the time window,
        with missing values handled
    """
    # 1) Filter Spark DataFrame by time window
    filtered_tb = spark_df.filter(
        (F.col(timestamp_column) >= F.lit(first_timestamp)) &
        (F.col(timestamp_column) <= F.lit(last_timestamp))
    )
 
   # 2) Fill nulls with last valid value (forward fill)
    filled_filtered_tb = fill_nulls_with_last_valid(
        filtered_tb.orderBy(timestamp_column, ascending=True),
        timestamp_column=timestamp_column,
        columns=columns_to_fill,
    )
 
    # 3) Fill first row with first valid non-null entry (backfill start)
    filled_filtered_tb = fill_first_row_with_first_valid(
        filled_filtered_tb,
        columns=columns_to_fill,
    )
 
    # 4) Repair Sensor/ Data Flow Glitches
    # Fix quick drops to zero that are likely noise (sudden drops to zero that return to a high value quickly)
    for column in columns_to_fill:
        filled_filtered_tb = repair_data_glitches(
            filled_filtered_tb,
            column=column,
            active_threshold=1,
            lookahead_steps=2
        )
 
    return filled_filtered_tb
 
 
#### VD stert/end Volume Gas Tag logic
 
def detect_tank_usage_events(
    df: pyspark.sql.DataFrame,
    initial_heat_number: int,
    initial_timestamp: str,
    predicted_tank: Optional[int] = None,
    minimal_duration: int = 10,
    debug: bool = False
) -> pyspark.sql.DataFrame:
    """
    Detects usage cycles (events) for Tank 1 and Tank 2 based on cumulative gas volume data.
   
    The function identifies the active periods of each vessel by detecting:
    - Rising Edge: Transition from 0 to a positive volume (Start of usage).
    - Falling Edge: Transition from a positive volume back to 0 (End of usage).
   
    It consolidates Argon and Nitrogen volumes to determine the total tank activity.
 
    Args:
        df (pyspark.sql.DataFrame): Input PySpark DataFrame containing columns:
                        'timestamp',
                        'ACI@VD_VASO1_VOLUMEAR', 'ACI@VD_VASO1_VOLUMEN2',
                        'ACI@VD_VASO2_VOLUMEAR', 'ACI@VD_VASO2_VOLUMEN2'.
        initial_heat_number (int): The identifier for the heat (corrida) to associate with these events.
        initial_timestamp (str): ISO format timestamp string to filter the start of the analysis.
        predicted_tank (Optional[int]): The expected tank to be used (1 or 2). Added as metadata to the output.
        minimal_duration (int): Minimum duration in seconds for an event to be considered valid.
        debug (bool): If True, prints additional information for debugging purposes.
 
    Returns:
        pyspark.sql.DataFrame: A DataFrame containing the detected events with the following schema:
                   [tank_id, start_time, end_time, duration_seconds, max_volume, heat_number, predicted_tank]
    """
 
    # 1. Pre-processing: Filter by start time to optimize and fill nulls
    # We explicitly select and rename columns to avoid syntax issues with special chars like '@'
    # and handle nulls which would break the math.
 
    clean_df = (
        df
        .filter(F.col("timestamp") >= F.lit(initial_timestamp))
        .fillna(0, subset=[
            "ACI@VD_VASO1_VOLUMEAR", "ACI@VD_VASO1_VOLUMEN2",
            "ACI@VD_VASO2_VOLUMEAR", "ACI@VD_VASO2_VOLUMEN2"
        ])
    )
 
    # 2. Calculate Total Volume per Tank
    # Logic: Tank is active if either Ar or N2 volume is increasing.
    df_vol = clean_df.select(
        "*",
        (F.col("ACI@VD_VASO1_VOLUMEAR") + F.col("ACI@VD_VASO1_VOLUMEN2")).alias("vol_total_v1"),
        (F.col("ACI@VD_VASO2_VOLUMEAR") + F.col("ACI@VD_VASO2_VOLUMEN2")).alias("vol_total_v2"),
    )
 
    if debug:
        displayHTML("df_vol")
        display(df_vol)
 
    # 3. Define Window for Time Series Analysis
    w = pyspark.sql.Window.orderBy("timestamp")
 
    # 4. Detect Edges using Lag (previous) and Lead (next)
    # We calculate the previous value to detect starts, and look ahead to detect abrupt stops.
    df_edges = df_vol.select(
        "*",
        (F.lag("vol_total_v1", 1, default=0).over(w)).alias("prev_vol_v1"),
        (F.lag("vol_total_v2", 1, default=0).over(w)).alias("prev_vol_v2"),
        # No default value to detect abrupt stop of data flow
        (F.lead("vol_total_v1", 1, default=None).over(w)).alias("next_vol_v1"),
        (F.lead("vol_total_v2", 1, default=None).over(w)).alias("next_vol_v2"),
        # Add a global row number to identify the dataframe boundary
        (F.row_number().over(w)).alias("global_row_number"),
    )
 
    if debug:
        displayHTML("df_edges")
        display(df_edges)
 
    # 5. Flag Active Periods (State Calculation)
    # Start Condition: Previous was 0 (or null treated as 0), Current > 0.
    # End Condition: Current > 0, Next is 0 (Abrupt reset).
    # We verify Vaso 1 and Vaso 2 separately.
   
    # Unpivot (stack) the dataframe to handle Vaso 1 and Vaso 2 in a single generic logic
    # This makes the code cleaner than duplicating logic for columns.
   
    # Create a structure to stack: Tank ID, Timestamp, Current Vol, Prev Vol, Next Vol
    v1_struct = F.struct(
        F.lit(1).alias("tank_id"),
        F.col("timestamp"),
        F.col("vol_total_v1").alias("vol"),
        F.col("prev_vol_v1").alias("prev"),
        F.col("next_vol_v1").alias("next"),
        F.col("global_row_number")
    )
   
    v2_struct = F.struct(
        F.lit(2).alias("tank_id"),
        F.col("timestamp"),
        F.col("vol_total_v2").alias("vol"),
        F.col("prev_vol_v2").alias("prev"),
        F.col("next_vol_v2").alias("next"),
        F.col("global_row_number")
    )
 
    # Explode array to get rows for Tank 1 and Tank 2
    df_stacked = (
        df_edges
        .select(F.array(v1_struct, v2_struct).alias("data"))
        .select(F.explode("data").alias("d"))
        .select(
            "d.tank_id",
            "d.timestamp",
            "d.vol",
            "d.prev",
            "d.next",
            "d.global_row_number",
        )
    )
 
    if debug:
        displayHTML("df_stacked")
        display(df_stacked)
 
    # 6. Determine Start and End Flags based on stacked data
    # Threshold is 1 to account for floating point noise, though 0 usually works for registers.
    THRESHOLD = 0.99
   
    df_flagged = df_stacked.select(
        "*",
        (
            (F.col("prev") <= THRESHOLD) & (F.col("vol") > THRESHOLD) & (F.col("global_row_number") > 1)
        ).alias("is_start"),
        (
            (F.col("vol") > THRESHOLD)
        ).alias("is_active"),
        (
            (F.col("vol") > THRESHOLD) & (F.col("next") <= THRESHOLD)
        ).alias("is_clean_finish"),
    )
 
    if debug:
        displayHTML("df_flagged")
        display(df_flagged)
 
    # 7. Create Session IDs
    # We do a running sum of 'is_start' to assign a unique ID to each continuous block of activity per tank.
    w_tank = pyspark.sql.Window.partitionBy("tank_id").orderBy("timestamp")
   
    df_sessions = (
        df_flagged
        .filter(F.col("is_active"))
        .withColumn("session_id", F.sum(F.col("is_start").cast("long")).over(w_tank))
    )
 
    if debug:
        displayHTML("df_sessions")
        display(df_sessions)
 
    # 8. Aggregate to find Start and End of each session
    agg_df = (
        df_sessions
        .filter(F.col("session_id") > 0)
        .groupBy("tank_id", "session_id")
        .agg(
            F.min("timestamp").alias("start_time"),
            F.max("timestamp").alias("last_seen_timestamp"), # This is the last timestamp WITH volume
            F.max("vol").alias("max_volume"),
            # If ANY row in the session has a clean finish, the session is complete.
            F.max("is_clean_finish").alias("has_clean_finish")
        )
        # .withColumn("duration_seconds", F.col("end_time").cast("long") - F.col("start_time").cast("long"))
        # .withColumn("heat_number", F.lit(initial_heat_number))
        # .withColumn("predicted_tank_input", F.lit(predicted_tank))
    )
 
    # 9. Filter and final Calculations: Duration, Null End Time, Heat Increment
 
    # Window to order events globally (across both tanks) to assign incremental heat numbers
    w_global_order = Window.orderBy("start_time", "tank_id")
 
    events_df = agg_df.select(
        "*",
        (
            F.when(
                F.col("has_clean_finish") == True,
                F.col("last_seen_timestamp")
            ).otherwise(F.lit(None)) # Return None if it didn't finish cleanly
        ).alias("end_time"),
        (
            F.col("last_seen_timestamp").cast("long") - F.col("start_time").cast("long")
        ).alias("duration_seconds"),
    )
 
    events_df = (
        events_df
        # Filter to keep only events that are longer than the minimum threshold BEFORE incrementing
        # heat numbers (to avoid incrementing for noise)
        .filter(F.col("duration_seconds") >= minimal_duration)
        .select(
            "*",
            # Increment heat numbers
            (
                F.row_number().over(w_global_order) - 1
            ).alias("heat_increment"),
            (
                F.lit(initial_heat_number) + F.col("heat_increment")
            ).alias("corrida"),
            F.lit(predicted_tank).alias("predicted_tank_input")
        )
    )
 
    # # 9. Filter to keep only events that are longer than the minimum threshold
    # events_df = events_df.filter(F.col("duration_seconds") >= minimal_duration)
 
    # Optional: Order by time for readability
    return events_df.orderBy("start_time", "tank_id").select(
        "corrida", "tank_id", "start_time", "end_time", "duration_seconds", "max_volume", "predicted_tank_input"
    )

# COMMAND ----------

MIN_TIME_TRANSPORT_FEA_FP = 3  # minutes;
MIN_TIME_FP = 19  # minutes; 3 minutos para homogenização, 10 minutos entre P6 e P7; adição de liga
MIN_TIME_TRANSPORT_FP_VD = 3  # minutes; Média é 5 min
MIN_TIME_VD = 2

TIMESTAMP_COLUMN = 'timestamp'
# List of columns to fill
COLUMNS_TO_FILL = [
    'ACI@VD_VASO1_VOLUMEN2',
    'ACI@VD_VASO2_VOLUMEN2',
    'ACI@VD_VASO1_VOLUMEAR',
    'ACI@VD_VASO2_VOLUMEAR',
]

# COMMAND ----------

df_vd = spark.read.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_vd")

# COMMAND ----------

tb_aciaria_vd_filled_filtered = preprocess_pims_data(
    spark_df=df_vd,
    first_timestamp="2025-12-02 19:01:00",
    last_timestamp="2025-12-02 23:20:00",
    timestamp_column=TIMESTAMP_COLUMN,
    columns_to_fill=COLUMNS_TO_FILL,
)

# COMMAND ----------

result = detect_tank_usage_events(
    tb_aciaria_vd_filled_filtered,
    initial_heat_number=130470,
    initial_timestamp="2025-12-02 19:01:00",
    predicted_tank=1,
    minimal_duration=MIN_TIME_VD*60,
    debug=False
)

# COMMAND ----------

result.display()

# COMMAND ----------

fp_df  = spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida")
vd_df  = spark.table("industrial_iatemperaturascha_refined_prd.computed_features.tb_corridatempos")

vd_joined = (
    vd_df.alias("vd")
    .join(fp_df.select("corrida", "TTTVD"), "corrida", "left")
)

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


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F


def debug_vd_period(
    spark_df_vd_pims,          # tb_aciaria_vd (PIMS / OPC)
    spark_df_vd_official,      # vd_joined (com inicio_vd / final_vd)
    start_ts: str,
    end_ts: str,
    initial_corrida: int,
    predicted_tank: int | None = None,
    minimal_duration: int = 10,
    debug: bool = False,
):
    """
    Debug visual do VD:
      - Volume total por vaso
      - Início/Fim Oficial
      - Início/Fim Detectado
    """

    # =========================================================
    # 1. PRÉ-PROCESSAMENTO (igual você já usa)
    # =========================================================
    df_vd_filled = preprocess_pims_data(
        spark_df=spark_df_vd_pims,
        first_timestamp=start_ts,
        last_timestamp=end_ts,
        timestamp_column="timestamp",
        columns_to_fill=[
            "ACI@VD_VASO1_VOLUMEAR",
            "ACI@VD_VASO1_VOLUMEN2",
            "ACI@VD_VASO2_VOLUMEAR",
            "ACI@VD_VASO2_VOLUMEN2",
        ],
    )

    # =========================================================
    # 2. DETECÇÃO VD (NOSSA LÓGICA)
    # =========================================================
    df_detected = detect_tank_usage_events(
        df=df_vd_filled,
        initial_heat_number=initial_corrida,
        initial_timestamp=start_ts,
        predicted_tank=predicted_tank,
        minimal_duration=minimal_duration,
        debug=debug,
    ).toPandas()

    # =========================================================
    # 3. DADOS PARA PLOT (PANDAS)
    # =========================================================
    df_plot = (
        df_vd_filled
        .filter((F.col("timestamp") >= start_ts) & (F.col("timestamp") <= end_ts))
        .select(
            "timestamp",
            (F.col("ACI@VD_VASO1_VOLUMEAR") + F.col("ACI@VD_VASO1_VOLUMEN2")).alias("vol_v1"),
            (F.col("ACI@VD_VASO2_VOLUMEAR") + F.col("ACI@VD_VASO2_VOLUMEN2")).alias("vol_v2"),
        )
        .orderBy("timestamp")
        .toPandas()
    )

    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])

    # =========================================================
    # 4. VD OFICIAL
    # =========================================================
    df_official = (
        spark_df_vd_official
        .filter(
            (F.col("inicio_vd") >= start_ts) &
            (F.col("final_vd") <= end_ts)
        )
        .select("corrida", "inicio_vd", "final_vd")
        .toPandas()
    )

    # =========================================================
    # 5. PLOT
    # =========================================================
    plt.figure(figsize=(15, 6))

    plt.plot(df_plot["timestamp"], df_plot["vol_v1"], label="VD Vaso 1", color="tab:blue")
    plt.plot(df_plot["timestamp"], df_plot["vol_v2"], label="VD Vaso 2", color="tab:orange")

    y_max = max(df_plot["vol_v1"].max(), df_plot["vol_v2"].max()) * 1.05

    # ---- OFICIAL
    for _, r in df_official.iterrows():
        c = int(r["corrida"])

        plt.axvline(r["inicio_vd"], color="green", linestyle="--", linewidth=2)
        plt.text(r["inicio_vd"], y_max * 0.95, f"Início OP {c}", rotation=90,
                 color="green", ha="right")

        plt.axvline(r["final_vd"], color="red", linestyle="--", linewidth=2)
        plt.text(r["final_vd"], y_max * 0.90, f"Fim OP {c}", rotation=90,
                 color="red", ha="left")

    # ---- DETECTADO
    for _, r in df_detected.iterrows():
        c = int(r["corrida"])

        plt.axvline(r["start_time"], color="blue", linestyle=":", linewidth=2)
        plt.text(r["start_time"], y_max * 0.85, f"Início {c}", rotation=90,
                 color="blue", ha="right")

        if pd.notna(r["end_time"]):
            plt.axvline(r["end_time"], color="orange", linestyle=":", linewidth=2)
            plt.text(r["end_time"], y_max * 0.80, f"Fim {c}", rotation=90,
                     color="orange", ha="left")

    plt.title("VD — Volume Total | Oficial x Detectado")
    plt.legend()
    plt.grid(True)
    plt.show()

    return df_detected


# COMMAND ----------

df_vd_detected = debug_vd_period(
    spark_df_vd_pims=df_vd,
    spark_df_vd_official=vd_joined,
    start_ts="2025-12-02 19:01:00",
    end_ts="2025-12-02 23:50:00",
    initial_corrida=130469,
    predicted_tank=None,
    minimal_duration=10,
    debug=False,
)

display(df_vd_detected)


# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F


def get_vd_events_comparison(
    spark_df_vd_pims,
    spark_df_vd_official,
    start_ts: str,
    end_ts: str,
    initial_corrida: int,
    predicted_tank: int | None = None,
    minimal_duration: int = 10,
):
    """
    Retorna DataFrame Pandas com:
      - corrida
      - start_official
      - end_official
      - start_detected
      - end_detected
      - delta_start_min
      - delta_end_min
    """

    # =========================================================
    # 1. DETECÇÃO VD (NOSSA LÓGICA)
    # =========================================================
    df_detected = (
        detect_tank_usage_events(
            df=spark_df_vd_pims,
            initial_heat_number=initial_corrida,
            initial_timestamp=start_ts,
            predicted_tank=predicted_tank,
            minimal_duration=minimal_duration,
            debug=False,
        )
        .select(
            "corrida",
            F.col("start_time").alias("start_detected"),
            F.col("end_time").alias("end_detected"),
        )
        .toPandas()
    )

    # =========================================================
    # 2. VD OFICIAL
    # =========================================================
    df_official = (
        spark_df_vd_official
        .filter(
            (F.col("inicio_vd") >= start_ts) &
            (F.col("final_vd") <= end_ts)
        )
        .select(
            F.col("corrida"),
            F.col("inicio_vd").alias("start_official"),
            F.col("final_vd").alias("end_official"),
        )
        .toPandas()
    )

    # =========================================================
    # 3. NORMALIZAR DATETIME
    # =========================================================
    for c in [
        "start_detected", "end_detected",
        "start_official", "end_official",
    ]:
        if c in df_detected.columns:
            df_detected[c] = pd.to_datetime(df_detected[c])
        if c in df_official.columns:
            df_official[c] = pd.to_datetime(df_official[c])

    # =========================================================
    # 4. MERGE FINAL
    # =========================================================
    df = (
        df_detected
        .merge(df_official, on="corrida", how="outer")
        .sort_values("corrida")
        .reset_index(drop=True)
    )

    # =========================================================
    # 5. DELTAS (MINUTOS)
    # =========================================================
    df["delta_start_min"] = (
        (df["start_detected"] - df["start_official"])
        .dt.total_seconds() / 60
    )

    df["delta_end_min"] = (
        (df["end_detected"] - df["end_official"])
        .dt.total_seconds() / 60
    )

    return df


# COMMAND ----------

df_vd_events = get_vd_events_comparison(
    spark_df_vd_pims=tb_aciaria_vd_filled_filtered,
    spark_df_vd_official=vd_joined,
    start_ts="2025-12-09 19:01:00",
    end_ts="2025-12-23:20:00",
    initial_corrida=130469,
    predicted_tank=None,
    minimal_duration=10,
)

display(df_vd_events)
