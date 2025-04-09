import pandas as pd
import os
import json
from datetime import datetime
from typing import Optional


STUDY_START = datetime(2025, 3, 14).date()
STUDY_END = datetime(2025, 3, 27).date()
COFFEE_END = datetime(2025, 3, 20).date()
DATA_FOLDER = './raw_data/'
APPLY_INTERPOLATION = True
APPLY_SMOOTHING = False
WINDOW_SIZE = None

def load_and_merge(data_folder: str) -> pd.DataFrame:
    """Read and merge dataframes."""
    dfs = []

    for i, file in enumerate(os.listdir(data_folder)):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_folder, file))
            df = df.drop(columns=['Sid', 'UpdateTime'])
            df['Uid'] = i
            dfs.append(df)
    return pd.concat(dfs)

def convert_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df['Time'] = pd.to_datetime(df['Time'], unit='s')
    except ValueError:
        df['Time'] = pd.to_datetime(df['Time'])
    df['Date'] = df['Time'].dt.date
    return df

def limit_time_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[(df['Time'].dt.date >= STUDY_START) & (df['Time'].dt.date <= STUDY_END)]
    df['is_coffee'] = df['Time'].dt.date <= COFFEE_END
    return df

def extract_values(df: pd.DataFrame):
    df_bpm = df[df['Key'] == 'heart_rate'].copy()
    df_bpm["bpm"] = df_bpm["Value"].apply(lambda x: json.loads(x).get("bpm", None))

    df_sleep = df[df['Key'] == 'sleep'].copy()
    df_sleep["sleep_duration"] = df_sleep["Value"].apply(lambda x: json.loads(x).get("duration", None))
    df_sleep["sleep_deep_duration"] = df_sleep["Value"].apply(lambda x: json.loads(x).get("sleep_deep_duration", None))
    df_sleep["avg_hr"] = df_sleep["Value"].apply(lambda x: json.loads(x).get("avg_hr", None))
    df_sleep["sleep_rem_duration"] = df_sleep["Value"].apply(lambda x: json.loads(x).get("sleep_rem_duration", None))
    df_sleep["bedtime"] = df_sleep["Value"].apply(lambda x: json.loads(x).get("bedtime", None))
    df_sleep["bedtime"] = pd.to_datetime(df_sleep["bedtime"], unit='s')
    df_sleep["awake_count"] = df_sleep["Value"].apply(lambda x: json.loads(x).get("awake_count", None))
    # return df_bpm.drop(columns=['Value']), df_sleep.drop(columns=['Value'])
    return df_bpm, df_sleep

def filter_sleep(df_sleep: pd.DataFrame) -> pd.DataFrame:
    return df_sleep[df_sleep['sleep_deep_duration'] > 0]

def interpolate_bpm_data(df_bpm: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing timestamps in BPM data to ensure data points every minute.

    Args:
        df_bpm: DataFrame containing BPM data with 'Time' and 'bpm' columns

    Returns:
        DataFrame with interpolated BPM values for every minute in the time range
    """
    # Ensure we're working with a copy to avoid modifying the original
    df = df_bpm.copy()

    # Group by Uid to handle multiple users
    interpolated_dfs = []

    for uid, group in df.groupby('Uid'):
        if len(group) == 0:
            continue

        # Get min and max timestamps for this user
        min_time = group['Time'].min().floor('min')
        max_time = group['Time'].max().ceil('min')

        # Create a complete timestamp range at minute intervals
        complete_range = pd.date_range(start=min_time, end=max_time, freq='min')

        # Create a DataFrame with the complete range
        complete_df = pd.DataFrame({'Time': complete_range})
        complete_df['Date'] = complete_df['Time'].dt.date
        complete_df['Uid'] = uid
        complete_df['is_coffee'] = complete_df['Date'] <= COFFEE_END


        # Round original timestamps to the nearest minute for proper merging
        group['Time'] = group['Time'].dt.floor('min')

        # Merge with the original data
        merged = pd.merge(complete_df, group, on=['Time', 'Date', 'Uid', 'is_coffee'], how='left').reset_index()

        # For timestamps with multiple readings, take the mean
        agg_df = merged.groupby('Time').agg({
            'Uid': 'first',
            'Date': 'first',
            'is_coffee': 'first',
            'bpm': 'mean'
        }).reset_index()

        # Interpolate missing BPM values
        agg_df['bpm'] = agg_df['bpm'].interpolate(method='linear')
        agg_df['bpm_delta'] = agg_df['bpm'].diff()
        agg_df = agg_df.dropna(subset=['bpm_delta'])

        interpolated_dfs.append(agg_df)

    # Combine all users' data
    if interpolated_dfs:
        return pd.concat(interpolated_dfs)
    else:
        return pd.DataFrame()

def apply_moving_average(df_bpm: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    Apply a moving average to BPM data to reduce noise.

    Args:
        df_bpm: DataFrame containing BPM data with 'Time' and 'bpm' columns
        window_size: Size of the moving average window (default: 5 minutes)

    Returns:
        DataFrame with an additional 'bpm_smooth' column containing smoothed BPM values
    """
    # Ensure we're working with a copy to avoid modifying the original
    df = df_bpm.copy()

    # Apply moving average for each user separately
    for uid, group in df.groupby('Uid'):
        # Sort by time to ensure proper smoothing
        group = group.sort_values('Time')

        # Apply the moving average and update the original DataFrame
        smoothed_bpm = group['bpm'].rolling(window=window_size, center=True, min_periods=1).mean()
        df.loc[group.index, 'bpm_smooth'] = smoothed_bpm

    return df

def prepare_data(data_folder: str, apply_interpolation: bool = True, apply_smoothing: bool = True, window_size: Optional[int] = 5) -> tuple:
    """
    Prepare BPM and sleep data with optional interpolation and smoothing.

    Args:
        data_folder: Path to the folder containing data files
        apply_interpolation: Whether to apply timestamp interpolation (default: True)
        apply_smoothing: Whether to apply moving average smoothing (default: True)
        window_size: Size of the moving average window (default: 5 minutes)

    Returns:
        Tuple of (df_bpm, df_sleep) DataFrames
    """
    df = load_and_merge(data_folder)
    df = convert_time(df)
    df = limit_time_period(df)
    df_bpm, df_sleep = extract_values(df)
    df_sleep = filter_sleep(df_sleep)

    if apply_interpolation:
        df_bpm = interpolate_bpm_data(df_bpm)

    if apply_smoothing:
        df_bpm = apply_moving_average(df_bpm, window_size)

    df_bpm.to_csv(os.path.join(data_folder, 'bpm_data.csv'), index=False)
    df_sleep.to_csv(os.path.join(data_folder, 'sleep_data.csv'), index=False)

    return df_bpm, df_sleep



if __name__ == "__main__":
    prepare_data(
        DATA_FOLDER,
        apply_interpolation=APPLY_INTERPOLATION,
        apply_smoothing=APPLY_SMOOTHING,
        window_size=WINDOW_SIZE,
    )
