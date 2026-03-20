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

def detect_fp_events_gas_based(
    spark_df: pyspark.sql.DataFrame,
    initial_heat_number: Optional[int],
    initial_timestamp: str,
    last_timestamp: Optional[str] = None,
    gas_threshold: float = 1.0,
    min_peak_total_gas: float = 500.0,       # pico mínimo pra considerar FP “de verdade”
    minimal_duration_seconds: int = 60,      # duração mínima do FP
    end_fraction_of_peak: float = 0.90,      # quanto precisa cair em relação ao pico
    min_start_rel_increase: float = 0.01,    # subida mínima (1% = 0.01) pra considerar início
    debug: bool = False,
) -> pyspark.sql.DataFrame:
    """
    Detecta ciclos de FP (início/fim) usando APENAS o volume total de gás:
 
        gas_total = gas_carro1 + gas_carro2
 
    e usa os carros separados só para escolher `carro_fp` (quem teve maior volume).
 
    Início do FP:
        1ª vez que gas_total:
            - sobe de <= gas_threshold p/ > gas_threshold
            - e tem aumento relativo suficiente
              (rel_increase_prev >= min_start_rel_increase
               ou rel_increase_next >= min_start_rel_increase)
 
    Fim do FP (dinâmico):
        dentro da sessão, achamos o pico de gas_total;
        fp_end é a 1ª vez APÓS o pico em que:
            gas_total <= end_fraction_of_peak * peak_total_gas.
        Se não houver essa queda, usa o último timestamp da sessão.
 
    Saída:
        [corrida, fp_start, fp_end, duration_seconds,
         carro_fp, peak_total_gas, peak_gas_carro1, peak_gas_carro2]
    """
 
    # -------------------------------------------------------------------------
    # 0) Colunas fixas do FP
    # -------------------------------------------------------------------------
    TIMESTAMP_COL = "timestamp"
    COL_CORRIDA   = "ACI@FP_NUMERO_CORRIDA"
    COL_AR_C1     = "ACI@FP_Volume_Argonio_Carro_1"
    COL_N2_C1     = "ACI@FP_Volume_Nitrogenio_Carro_1"
    COL_AR_C2     = "ACI@FP_Volume_Argonio_Carro_2"
    COL_N2_C2     = "ACI@FP_Volume_Nitrogenio_Carro_2"
 
    if last_timestamp is None:
        last_timestamp = "9999-12-31 23:59:59"
 
    # -------------------------------------------------------------------------
    # 1) Pré-processamento PIMS
    # -------------------------------------------------------------------------
    columns_to_fill = [COL_AR_C1, COL_N2_C1, COL_AR_C2, COL_N2_C2]
 
    df_fp = preprocess_pims_data(
        spark_df=spark_df,
        first_timestamp=initial_timestamp,
        last_timestamp=last_timestamp,
        timestamp_column=TIMESTAMP_COL,
        columns_to_fill=columns_to_fill,
    ).orderBy(TIMESTAMP_COL)
 
    # -------------------------------------------------------------------------
    # 2) Corrida raw + last_corrida_before ("online")
    # -------------------------------------------------------------------------
    df_fp = df_fp.withColumn(
        "corrida_raw",
        F.when(F.col(COL_CORRIDA).cast("int") > 0,
               F.col(COL_CORRIDA).cast("int"))
         .otherwise(F.lit(None).cast("int"))
    )
 
    w_before = Window.orderBy(TIMESTAMP_COL).rowsBetween(Window.unboundedPreceding, -1)
 
    df_fp = df_fp.withColumn(
        "last_corrida_before",
        F.last("corrida_raw", ignorenulls=True).over(w_before)
    )
 
    if initial_heat_number is not None:
        df_fp = df_fp.withColumn(
            "last_corrida_before",
            F.when(
                F.col("last_corrida_before").isNull(),
                F.lit(initial_heat_number - 1)
            ).otherwise(F.col("last_corrida_before"))
        )
 
    # -------------------------------------------------------------------------
    # 3) Volumes de gás: carro 1, carro 2 e total
    # -------------------------------------------------------------------------
    df_fp = df_fp.select(
        "*",
        (F.col(COL_AR_C1).cast("double") + F.col(COL_N2_C1).cast("double")).alias("gas_carro1"),
        (F.col(COL_AR_C2).cast("double") + F.col(COL_N2_C2).cast("double")).alias("gas_carro2"),
    )
 
    df_fp = df_fp.withColumn(
        "gas_carro1",
        F.coalesce(F.col("gas_carro1"), F.lit(0.0))
    ).withColumn(
        "gas_carro2",
        F.coalesce(F.col("gas_carro2"), F.lit(0.0))
    )

    df_fp = df_fp.withColumn(
        "gas_total",
        F.col("gas_carro1") + F.col("gas_carro2")
    )

 
    # -------------------------------------------------------------------------
    # 4) Início / ativo + subida mínima
    # -------------------------------------------------------------------------
    w_order = Window.orderBy(TIMESTAMP_COL)
 
    df_fp = df_fp.select(
        "*",
        F.lag("gas_total", 1, 0.0).over(w_order).alias("prev_gas_total"),
        F.lead("gas_total", 1, 0.0).over(w_order).alias("next_gas_total"),
    )
 
    # variações relativas
    df_fp = df_fp.select(
        "*",
        F.when(
            F.col("prev_gas_total") > 0,
            (F.col("gas_total") - F.col("prev_gas_total")) / F.col("prev_gas_total")
        ).otherwise(F.lit(1.0)).alias("rel_increase_prev"),
        F.when(
            F.col("gas_total") > 0,
            (F.col("next_gas_total") - F.col("gas_total")) / F.col("gas_total")
        ).otherwise(F.lit(0.0)).alias("rel_increase_next"),
    )
 
    df_fp = df_fp.select(
        "*",
        (
            (F.col("prev_gas_total") <= gas_threshold) &
            (F.col("gas_total")      >  gas_threshold) &
            (
                (F.col("rel_increase_prev") >= min_start_rel_increase) |
                (F.col("rel_increase_next") >= min_start_rel_increase)
            )
        ).alias("is_start_fp"),
        (F.col("gas_total") > gas_threshold).alias("is_active_fp"),
    )
 
    # -------------------------------------------------------------------------
    # 4.5) session_id em TODO o df_fp (não só onde is_active_fp)
    # -------------------------------------------------------------------------
    df_fp = df_fp.withColumn(
        "session_id",
        F.sum(F.col("is_start_fp").cast("long")).over(w_order)
    )
 
    # Trabalhamos só com sessões > 0 (após primeiro início), mas mantendo zeros
    df_sessions = df_fp.filter(F.col("session_id") > 0)
 
    if debug:
        print("===== df_sessions (com session_id, ainda com zeros) =====")
        display(df_sessions.orderBy("session_id", TIMESTAMP_COL))
 
    # -------------------------------------------------------------------------
    # 5) Corrida no momento do start + corrida_evento por sessão
    # -------------------------------------------------------------------------
    df_sessions = df_sessions.withColumn(
        "corrida_at_fp_start",
        F.when(
            F.col("is_start_fp"),
            F.when(F.col("corrida_raw").isNotNull(), F.col("corrida_raw"))
             .otherwise(F.col("last_corrida_before") + F.lit(1))
        )
    )
 
    w_sess_full = (
        Window
        .partitionBy("session_id")
        .orderBy(TIMESTAMP_COL)
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )
 
    corrida_from_fp_col = F.max("corrida_raw").over(w_sess_full)
    corrida_from_start  = F.max("corrida_at_fp_start").over(w_sess_full)
 
    df_sessions = df_sessions.withColumn(
        "corrida_evento",
        F.when(corrida_from_fp_col.isNotNull(), corrida_from_fp_col)
         .otherwise(corrida_from_start)
    )
 
    # -------------------------------------------------------------------------
    # 6) Pico + fim dinâmico (agora vendo também os zeros)
    # -------------------------------------------------------------------------
    df_sessions = df_sessions.withColumn(
        "peak_total_gas",
        F.max("gas_total").over(w_sess_full)
    )
 
    df_sessions = df_sessions.withColumn(
        "is_peak",
        F.col("gas_total") == F.col("peak_total_gas")
    )
 
    df_sessions = df_sessions.withColumn(
        "peak_ts",
        F.max(F.when(F.col("is_peak"), F.col(TIMESTAMP_COL))).over(w_sess_full)
    )
 
    df_sessions = df_sessions.withColumn(
        "dynamic_end_level",
        F.col("peak_total_gas") * F.lit(end_fraction_of_peak)
    )
 
    df_sessions = df_sessions.withColumn(
        "is_dynamic_end_candidate",
        (F.col(TIMESTAMP_COL) >= F.col("peak_ts")) &
        (F.col("gas_total") <= F.col("dynamic_end_level"))
    )
 
    df_sessions = df_sessions.withColumn(
        "dynamic_end_ts",
        F.when(F.col("is_dynamic_end_candidate"), F.col(TIMESTAMP_COL))
    )
 
    if debug:
        print("===== df_sessions (com pico e fim dinâmico) =====")
        display(df_sessions.orderBy("session_id", TIMESTAMP_COL))
 
    # -------------------------------------------------------------------------
    # 7) Agregação por sessão
    # -------------------------------------------------------------------------
    agg_df = (
        df_sessions
        .groupBy("session_id")
        .agg(
            F.min(TIMESTAMP_COL).alias("fp_start"),
            F.max(TIMESTAMP_COL).alias("fp_last_seen"),
            F.max("peak_total_gas").alias("peak_total_gas"),
            F.max("gas_carro1").alias("peak_gas_carro1"),
            F.max("gas_carro2").alias("peak_gas_carro2"),
            F.max("corrida_evento").alias("corrida"),
            F.min("dynamic_end_ts").alias("fp_end_dynamic"),
        )
    )
 
    agg_df = agg_df.withColumn(
        "fp_end",
        F.coalesce(F.col("fp_end_dynamic"), F.col("fp_last_seen"))
    )
 
    agg_df = agg_df.withColumn(
        "duration_seconds",
        F.col("fp_end").cast("long") - F.col("fp_start").cast("long")
    )
 
    if debug:
        print("===== agg_df (antes dos filtros) =====")
        display(agg_df.orderBy("fp_start"))
 
    # -------------------------------------------------------------------------
    # 8) Filtros de ruído
    # -------------------------------------------------------------------------
    events_df = agg_df.filter(
        (F.col("duration_seconds") >= minimal_duration_seconds) &
        (F.col("peak_total_gas")   >= min_peak_total_gas)
    )

    # carro FP
    events_df = events_df.withColumn(
        "carro_fp",
        F.when(F.col("peak_gas_carro1") >= F.col("peak_gas_carro2"), F.lit(1))
        .otherwise(F.lit(2))
    )

    # -------------------------------------------------------------------------
    # 9) CORRIDA SEMPRE CRESCENTE (🔥 mudança chave)
    # -------------------------------------------------------------------------
    w_corrida = Window.orderBy("fp_start")

    events_df = events_df.withColumn(
        "corrida",
        F.lit(initial_heat_number) +
        (F.row_number().over(w_corrida) - 1)
    )

    # -------------------------------------------------------------------------
    # 10) Seleção final
    # -------------------------------------------------------------------------
    events_df = (
        events_df
        .orderBy("fp_start")
        .select(
            "corrida",
            "fp_start",
            "fp_end",
            "duration_seconds",
            "carro_fp",
            "peak_total_gas",
            "peak_gas_carro1",
            "peak_gas_carro2",
        )
    )

    if debug:
        print("===== events_df (resultado final FP) =====")
        display(events_df)

    return events_df


# COMMAND ----------

df_fp = spark.read.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_fp")

# COMMAND ----------

first_ts = "2025-11-24 20:55:00"
last_ts  = "2025-11-25 13:00:00"
fp_events = detect_fp_events_gas_based(
    spark_df=df_fp,
    initial_heat_number=130320,
    initial_timestamp=first_ts,
    last_timestamp=last_ts,
    gas_threshold=100.0,           # passou de 100 = FP real
    min_peak_total_gas=800.0,
    minimal_duration_seconds=60,
    end_fraction_of_peak=0.9,
    min_start_rel_increase=0.05,   # 5% de subida global
    debug=False,
)

display(fp_events)

# COMMAND ----------

def prepare_fp_gas_series(
    spark_df_fp,
    start_ts: str,
    end_ts: str,
):
    """
    Retorna DataFrame Pandas com:
      - timestamp
      - gas_carro1
      - gas_carro2
      - gas_total
    """

    COL_AR_C1 = "ACI@FP_Volume_Argonio_Carro_1"
    COL_N2_C1 = "ACI@FP_Volume_Nitrogenio_Carro_1"
    COL_AR_C2 = "ACI@FP_Volume_Argonio_Carro_2"
    COL_N2_C2 = "ACI@FP_Volume_Nitrogenio_Carro_2"

    # 🔹 usa o MESMO preprocessamento do detector
    df_fp_clean = preprocess_pims_data(
        spark_df=spark_df_fp,
        first_timestamp=start_ts,
        last_timestamp=end_ts,
        timestamp_column="timestamp",
        columns_to_fill=[COL_AR_C1, COL_N2_C1, COL_AR_C2, COL_N2_C2],
    )

    df_plot = (
        df_fp_clean
        .select(
            "timestamp",
            (F.col(COL_AR_C1).cast("double") + F.col(COL_N2_C1).cast("double")).alias("gas_carro1"),
            (F.col(COL_AR_C2).cast("double") + F.col(COL_N2_C2).cast("double")).alias("gas_carro2"),
        )
        .withColumn("gas_carro1", F.coalesce(F.col("gas_carro1"), F.lit(0.0)))
        .withColumn("gas_carro2", F.coalesce(F.col("gas_carro2"), F.lit(0.0)))
        .withColumn("gas_total", F.col("gas_carro1") + F.col("gas_carro2"))
        .orderBy("timestamp")
        .toPandas()
    )

    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
    return df_plot


# COMMAND ----------

def convert_datetime(df, date_col, hour_col, new_col):
    return (
        df.withColumn(
            "date_clean",
            F.to_date(F.col(date_col))
        )
        .withColumn(
            "hour_clean",
            F.when(
                F.length(F.col(hour_col)) == 4,
                F.concat_ws(
                    ":",
                    F.col(hour_col).substr(1, 2),
                    F.col(hour_col).substr(3, 2)
                )
            ).otherwise(F.col(hour_col))
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

def debug_fp_period(
    spark_df_fp_raw,
    spark_df_fp_official,
    start_ts: str,
    end_ts: str,
    detect_kwargs: dict,
):
    """
    Debug visual do FP:
      - gás total (PIMS tratado)
      - início/fim oficial (MES)
      - início/fim detectado (lógica)
    """

    # ==========================
    # 1. DETECÇÃO (PIMS tratado)
    # ==========================
    df_detected = detect_fp_events_gas_based(
        spark_df=spark_df_fp_raw,
        initial_timestamp=start_ts,
        last_timestamp=end_ts,
        **detect_kwargs,
    ).toPandas()

    # ==========================
    # 2. PRÉ-PROCESSAMENTO PIMS (🔥 NOVO)
    # ==========================
    df_pims = preprocess_pims_data(
        spark_df=spark_df_fp_raw,
        first_timestamp=start_ts,
        last_timestamp=end_ts,
        timestamp_column="timestamp",
        columns_to_fill=[
            "ACI@FP_Volume_Argonio_Carro_1",
            "ACI@FP_Volume_Nitrogenio_Carro_1",
            "ACI@FP_Volume_Argonio_Carro_2",
            "ACI@FP_Volume_Nitrogenio_Carro_2",
        ],
    )

    # ==========================
    # 3. SÉRIE TEMPORAL (AGORA LIMPA)
    # ==========================
    df_plot = (
        df_pims
        .select(
            "timestamp",
            (
                F.col("ACI@FP_Volume_Argonio_Carro_1") +
                F.col("ACI@FP_Volume_Nitrogenio_Carro_1") +
                F.col("ACI@FP_Volume_Argonio_Carro_2") +
                F.col("ACI@FP_Volume_Nitrogenio_Carro_2")
            ).alias("gas_total")
        )
        .orderBy("timestamp")
        .toPandas()
    )

    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])

    # ==========================
    # 4. FP OFICIAL (MES)
    # ==========================
    df_official = (
        spark_df_fp_official
        .filter(
            (F.col("inicio_fp") >= start_ts) &
            (F.col("final_fp") <= end_ts)
        )
        .select("corrida", "inicio_fp", "final_fp")
        .toPandas()
    )

    # ==========================
    # 5. PLOT
    # ==========================
    plt.figure(figsize=(14, 5))
    plt.plot(df_plot["timestamp"], df_plot["gas_total"], label="Gás Total FP")

    y_max = df_plot["gas_total"].max()

    # ---- Oficial
    for _, r in df_official.iterrows():
        c = int(r["corrida"])
        plt.axvline(r["inicio_fp"], color="green", linestyle="--")
        plt.text(r["inicio_fp"], y_max * 0.95, f"Início OP {c}",
                 rotation=90, color="green", ha="right")

        plt.axvline(r["final_fp"], color="red", linestyle="--")
        plt.text(r["final_fp"], y_max * 0.90, f"Fim OP {c}",
                 rotation=90, color="red", ha="left")

    # ---- Detectado
    for _, r in df_detected.iterrows():
        c = int(r["corrida"])
        plt.axvline(r["fp_start"], color="blue", linestyle=":")
        plt.text(r["fp_start"], y_max * 0.85, f"Início {c}",
                 rotation=90, color="blue", ha="right")

        if pd.notna(r["fp_end"]):
            plt.axvline(r["fp_end"], color="orange", linestyle=":")
            plt.text(r["fp_end"], y_max * 0.80, f"Fim {c}",
                     rotation=90, color="orange", ha="left")

    plt.title("FP — Oficial x Detectado (PIMS tratado)")
    plt.grid(True)
    plt.legend()
    plt.show()

    return df_detected


# COMMAND ----------

df_fp = spark.read.table(
    "industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_fp"
)
fp_df = spark.read.table(
    "industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida"
)

# Chegada / Saída FP
fp_df = convert_datetime(fp_df, "DATACHEGADAFP", "HORACHEGADAFP", "inicio_fp")
fp_df = convert_datetime(fp_df, "DATASAIDAFP", "HORASAIDAFP", "final_fp")

# Normalização de nomes
fp_df = (
    fp_df
    .withColumn("inicio_fp", F.col("inicio_fp"))
    .withColumn("final_fp", F.col("final_fp"))
)



# COMMAND ----------

fp_debug_params = dict(
    initial_heat_number=130469,
    gas_threshold=100.0,
    min_peak_total_gas=800.0,
    minimal_duration_seconds=60,
    end_fraction_of_peak=0.9,
    min_start_rel_increase=0.05,
)

df_fp_detected = debug_fp_period(
    spark_df_fp_raw=df_fp,
    spark_df_fp_official=fp_df,   # tabela MES oficial FP
    start_ts="2025-12-02 18:01:00",
    end_ts="2025-12-02 23:50:00",
    detect_kwargs=fp_debug_params,
)




# COMMAND ----------

from pyspark.sql import functions as F


def build_fp_comparison_table(
    spark_fp_official: pyspark.sql.DataFrame,
    spark_fp_detected: pyspark.sql.DataFrame,
    start_ts: str,
    end_ts: str,
) -> pyspark.sql.DataFrame:
    """
    Tabela FP: Oficial x Detectado com deltas em minutos.
    - Filtra janela
    - Remove corrida 0 / nula
    - Deduplica para 1 linha por corrida (min start, max end)
    """

    # --------------------------
    # 1) FP OFICIAL (MES) — 1 linha por corrida
    # --------------------------
    fp_official = (
        spark_fp_official
        .filter(
            (F.col("inicio_fp").isNotNull()) &
            (F.col("final_fp").isNotNull()) &
            (F.col("inicio_fp") >= F.lit(start_ts)) &
            (F.col("inicio_fp") <= F.lit(end_ts))
        )
        .withColumn("corrida", F.col("corrida").cast("int"))
        .filter(F.col("corrida") > 0)
        .groupBy("corrida")
        .agg(
            F.min("inicio_fp").alias("fp_start_oficial"),
            F.max("final_fp").alias("fp_end_oficial"),
        )
    )

    # --------------------------
    # 2) FP DETECTADO (PIMS) — 1 linha por corrida
    # --------------------------
    fp_detected = (
        spark_fp_detected
        .filter(
            (F.col("fp_start").isNotNull()) &
            (F.col("fp_start") >= F.lit(start_ts)) &
            (F.col("fp_start") <= F.lit(end_ts))
        )
        .withColumn("corrida", F.col("corrida").cast("int"))
        .filter(F.col("corrida") > 0)
        .groupBy("corrida")
        .agg(
            F.min("fp_start").alias("fp_start_detectado"),
            F.max("fp_end").alias("fp_end_detectado"),
        )
    )

    # --------------------------
    # 3) JOIN 1:1 por corrida (sem explosão)
    # --------------------------
    df_cmp = fp_official.join(fp_detected, on="corrida", how="full")

    # --------------------------
    # 4) Deltas em minutos
    # --------------------------
    df_cmp = (
        df_cmp
        .withColumn(
            "delta_start_min",
            (F.col("fp_start_detectado").cast("long") - F.col("fp_start_oficial").cast("long")) / 60.0
        )
        .withColumn(
            "delta_end_min",
            (F.col("fp_end_detectado").cast("long") - F.col("fp_end_oficial").cast("long")) / 60.0
        )
    )

    return (
        df_cmp
        .orderBy("corrida")
        .select(
            "corrida",
            "fp_start_oficial",
            "fp_start_detectado",
            "delta_start_min",
            "fp_end_oficial",
            "fp_end_detectado",
            "delta_end_min",
        )
    )


# COMMAND ----------

start_ts = "2025-11-24 22:55:00"
end_ts   = "2025-11-25 03:00:00"

fp_detected = detect_fp_events_gas_based(
    spark_df=df_fp,
    initial_heat_number=130334,
    initial_timestamp=start_ts,
    last_timestamp=end_ts,
    gas_threshold=100.0,
    min_peak_total_gas=800.0,
    minimal_duration_seconds=60,
    end_fraction_of_peak=0.9,
    min_start_rel_increase=0.05,
    debug=False,
)

fp_comparison = build_fp_comparison_table(
    spark_fp_official=fp_df,         # MES (já com inicio_fp / final_fp)
    spark_fp_detected=fp_detected,   # PIMS detectado
    start_ts=start_ts,
    end_ts=end_ts,
)

display(fp_comparison)
