"""
Sleep and Heart Rate Analysis Script

This script analyzes relationships between caffeine intake, sleep patterns, and heart rate data.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from typing import Optional, Tuple, List, Dict, Union
import numpy as np
from matplotlib.ticker import FuncFormatter


def load_data(heart_rate_path: str = 'raw_data/bpm_data.csv',
              sleep_data_path: str = 'raw_data/sleep_data.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load heart rate and sleep data from CSV files.

    Args:
        heart_rate_path: Path to the heart rate data CSV file
        sleep_data_path: Path to the sleep data CSV file

    Returns:
        Tuple containing heart rate DataFrame and sleep DataFrame
    """
    df_bpm = pd.read_csv(heart_rate_path)
    df_sleep = pd.read_csv(sleep_data_path)
    return df_bpm, df_sleep


def visualize_bpm_comparison(df_bpm: pd.DataFrame, uid: Optional[int] = None) -> None:
    """
    Visualize original BPM data against the smoothed version.

    Args:
        df_bpm: DataFrame containing BPM data with 'bpm' and 'bpm_smooth' columns
        uid: User ID to filter by (optional)
    """
    if uid is not None:
        plot_data = df_bpm[df_bpm['Uid'] == uid].copy()
    else:
        plot_data = df_bpm.copy()

    if len(plot_data) == 0:
        print("No data available for the specified user.")
        return

    plot_data = plot_data.sort_values('Time')

    plt.figure(figsize=(15, 6))
    plt.plot(plot_data['Time'], plot_data['bpm'], 'b-', alpha=0.5, label='Original BPM')
    plt.plot(plot_data['Time'], plot_data['bpm_smooth'], 'r-', label='Smoothed BPM')
    plt.xlabel('Time')
    plt.ylabel('Heart Rate (BPM)')
    plt.title('Original vs. Smoothed Heart Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()


def analyze_deep_sleep_vs_caffeine(sleep_data: pd.DataFrame,
                                   save_path: str = "plots/deep_sleep_vs_caffeine.png") -> None:
    """
    Analyze and visualize relationship between caffeine intake and deep sleep duration.

    Args:
        sleep_data: DataFrame containing sleep data
        save_path: Path to save the generated plot
    """
    grouped_sleep_data = sleep_data.groupby(['Uid', 'Date', 'is_coffee']).agg(
        {'sleep_duration': 'sum', 'sleep_deep_duration': 'sum'}
    ).reset_index()

    deep_caff = grouped_sleep_data[grouped_sleep_data["is_coffee"] == True]["sleep_deep_duration"]
    deep_nocaff = grouped_sleep_data[grouped_sleep_data["is_coffee"] == False]["sleep_deep_duration"]
    print(f"Deep Sleep Duration - Caffeine: {deep_caff.mean():.2f} min, No Caffeine: {deep_nocaff.mean():.2f} min")
    t_deep, p_deep = ttest_ind(deep_nocaff, deep_caff, equal_var=False)
    print(f"Deep Sleep - t: {t_deep:.2f}, p: {p_deep:.4f}")

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=grouped_sleep_data, x="is_coffee", y="sleep_deep_duration")
    plt.xticks([0, 1], ['No Caffeine', 'Caffeine'])
    plt.title("Deep Sleep Duration vs Caffeine")
    plt.ylabel("Minutes of Deep Sleep")
    plt.xlabel("Caffeine Intake")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()


def visualize_hr_density_by_student(df_bpm: pd.DataFrame, save_path: str = "plots/hr_density_per_student.png") -> None:
    """
    Generate heart rate density plots for each student, comparing caffeine vs no caffeine.

    Args:
        df_bpm: DataFrame containing heart rate data
        save_path: Path to save the generated plot
    """
    unique_students = df_bpm["Uid"].unique()
    num_students = len(unique_students)

    fig, axes = plt.subplots(num_students, 1, figsize=(10, 3 * num_students))

    # Handle case with single student
    if num_students == 1:
        axes = [axes]

    # Iterate over each unique student (Uid) and subplot
    for i, student_id in enumerate(unique_students):
        # Filter the data for the specific student
        df_student = df_bpm[df_bpm["Uid"] == student_id]

        # Split the data into caffeine and non-caffeine for this student
        bpm_caff = df_student[df_student["is_coffee"] == True]["bpm"]
        bpm_nocaff = df_student[df_student["is_coffee"] == False]["bpm"]

        # Plot KDE for caffeine and no caffeine for this student on the corresponding subplot
        sns.kdeplot(bpm_caff, label=f"Caffeine - Student {student_id}", fill=True,
                    color="deepskyblue", ax=axes[i], alpha=0.6)
        sns.kdeplot(bpm_nocaff, label=f"No Caffeine - Student {student_id}", fill=True,
                    color="orange", ax=axes[i], alpha=0.6)

        # Set plot title and labels for each subplot
        axes[i].set_title(f"Heart Rate Density — Caffeine vs No Caffeine (Student {student_id})")
        axes[i].set_xlabel("Heart Rate (bpm)")
        axes[i].set_ylabel("Density")
        axes[i].grid(True)
        axes[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()


def analyze_sleep_hr_by_caffeine(sleep_data: pd.DataFrame,
                                 save_path: str = 'plots/hr_density_sleep_per_person.png') -> None:
    """
    Analyze heart rate during sleep, comparing caffeine vs no caffeine conditions.

    Args:
        sleep_data: DataFrame containing sleep data with heart rate information
        save_path: Path to save the generated plot
    """
    unique_students = sleep_data["Uid"].unique()
    num_students = len(unique_students)

    fig, axes = plt.subplots(num_students, 1, figsize=(10, 3 * num_students))

    # Handle case with single student
    if num_students == 1:
        axes = [axes]

    # Iterate over each unique student (Uid) and subplot
    for i, student_id in enumerate(unique_students):
        df_student = sleep_data[sleep_data["Uid"] == student_id]

        hr_caff = df_student[df_student["is_coffee"] == True]["avg_hr"]
        hr_nocaff = df_student[df_student["is_coffee"] == False]["avg_hr"]

        t_hr, p_hr = ttest_ind(hr_nocaff, hr_caff, equal_var=False)
        print(f"Uid: {student_id}")
        print(f"HR Density During Sleep - t: {t_hr:.2f}, p: {p_hr:.4f}")
        print()

        sns.kdeplot(hr_caff, label="Caffeine", fill=True, ax=axes[i])
        sns.kdeplot(hr_nocaff, label="No Caffeine", fill=True, ax=axes[i])
        axes[i].set_title(f"Heart Rate During Sleep — Caffeine vs No Caffeine for student {student_id}")
        axes[i].set_xlabel("Heart Rate (bpm)")
        axes[i].set_ylabel("Density")
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()


def visualize_sleep_stages(sleep_data: pd.DataFrame, save_path: str = 'plots/sleep_stages.png') -> None:
    """
    Visualize the proportion of different sleep stages (deep vs light) by caffeine intake.

    Args:
        sleep_data: DataFrame containing sleep data
        save_path: Path to save the generated plot
    """
    # Calculate light sleep duration
    sleep_data_with_light = sleep_data.copy()
    sleep_data_with_light['sleep_light_duration'] = sleep_data_with_light['sleep_duration'] - sleep_data_with_light[
        'sleep_deep_duration']

    # Compute means grouped by caffeine
    grouped = sleep_data_with_light.groupby("is_coffee")[["sleep_deep_duration", "sleep_light_duration"]].mean()

    # Convert to percentages
    grouped_percent = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # Plot as stacked bar chart of percentages
    ax = grouped_percent.plot(kind="bar", stacked=True, figsize=(6, 4))

    # Add annotations
    for i, (idx, row) in enumerate(grouped_percent.iterrows()):
        cumulative = 0
        for stage in row.index:
            value = row[stage]
            if value > 5:  # Only label segments bigger than 5% to avoid clutter
                ax.text(
                    i,
                    cumulative + value / 2,  # Middle of the segment
                    f"{value:.1f}%",
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=10,
                    fontweight='bold'
                )
            cumulative += value

    plt.xticks([0, 1], ['No Caffeine', 'Caffeine'], rotation=0)
    plt.title("Proportion of Sleep Stages (Deep vs Light)")
    plt.ylabel("Percentage of Total Sleep (%)")
    plt.xlabel("Caffeine Intake")
    plt.ylim(0, 120)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()


def visualize_hr_fluctuation_by_student(df_bpm: pd.DataFrame,
                                        save_path: str = "plots/hr_density_fluctuation_per_student.png") -> None:
    """
    Visualize heart rate fluctuations (delta) for each student, comparing caffeine vs no caffeine.

    Args:
        df_bpm: DataFrame containing heart rate data with bpm_delta column
        save_path: Path to save the generated plot
    """
    unique_students = df_bpm["Uid"].unique()
    num_students = len(unique_students)

    fig, axes = plt.subplots(num_students, 1, figsize=(10, 3 * num_students))

    # Handle case with single student
    if num_students == 1:
        axes = [axes]

    # Iterate over each unique student (Uid) and subplot
    for i, student_id in enumerate(unique_students):
        # Filter the data for the specific student
        df_student = df_bpm[df_bpm["Uid"] == student_id]

        # Split the data into caffeine and non-caffeine for this student
        bpm_caff = df_student[df_student["is_coffee"] == True]["bpm_delta"]
        bpm_nocaff = df_student[df_student["is_coffee"] == False]["bpm_delta"]

        # Plot KDE for caffeine and no caffeine for this student on the corresponding subplot
        sns.kdeplot(bpm_caff, label=f"Caffeine - Student {student_id}", fill=True,
                    color="deepskyblue", ax=axes[i], alpha=0.6)
        sns.kdeplot(bpm_nocaff, label=f"No Caffeine - Student {student_id}", fill=True,
                    color="orange", ax=axes[i], alpha=0.6)

        # Set plot title and labels for each subplot
        axes[i].set_title(f"Heart Rate Fluctuation — Caffeine vs No Caffeine (Student {student_id})")
        axes[i].set_xlabel("Heart Rate Change (bpm)")
        axes[i].set_ylabel("Density")
        axes[i].grid(True)
        axes[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()


def analyze_awake_count(sleep_data: pd.DataFrame, save_path: str = 'plots/awake_count_per_person.png') -> None:
    """
    Analyze and visualize number of awake periods during sleep, comparing caffeine vs no caffeine.

    Args:
        sleep_data: DataFrame containing sleep data with awake_count column
        save_path: Path to save the generated plot
    """
    unique_students = sleep_data["Uid"].unique()
    num_students = len(unique_students)

    fig, axes = plt.subplots(num_students, 1, figsize=(10, 3 * num_students))

    # Handle case with single student
    if num_students == 1:
        axes = [axes]

    # Iterate over each unique student (Uid) and subplot
    for i, student_id in enumerate(unique_students):
        df_student = sleep_data[sleep_data["Uid"] == student_id]

        hr_caff = df_student[df_student["is_coffee"] == True]["awake_count"]
        hr_nocaff = df_student[df_student["is_coffee"] == False]["awake_count"]

        # Plot histograms for both caffeine and no caffeine groups
        axes[i].hist(hr_caff, bins=10, alpha=0.5, label="Caffeine", color="blue")
        axes[i].hist(hr_nocaff, bins=10, alpha=0.5, label="No Caffeine", color="orange")

        # Set titles and labels
        axes[i].set_title(f"Number of awakes during sleep — Caffeine vs No Caffeine for student {student_id}")
        axes[i].set_xlabel("Number of awakes")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()


def plot_sleep_duration(df_sleep):
    # Create figure and axis objects explicitly
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create the bar plot
    sns_plot = sns.barplot(data=df_sleep, x="is_coffee", y="sleep_duration", ax=ax)

    # Set the x-axis labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Caffeine', 'Caffeine'])

    # Set plot titles and labels
    ax.set_title("Average Sleep Duration with and without Caffeine")
    ax.set_ylabel("Sleep Duration (minutes)")
    ax.set_xlabel("Caffeine Intake")

    # Add annotations in the center of each bar
    for i, bar in enumerate(ax.patches):
        # Get the height of the bar
        height = bar.get_height()
        # Calculate position for text (center of bar)
        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = height / 2  # Middle of the bar

        # Add the text
        ax.text(
            x_pos, y_pos,
            f'{height:.1f}',
            ha='center',
            va='center',
            color='white',  # White text for better visibility
            fontweight='bold'  # Make the text bold for better visibility
        )

    # Ensure tight layout and save
    plt.tight_layout()
    plt.savefig('plots/sleep_comparison.png', dpi=300)
    plt.close()
    # plt.show()


def analyze_bedtime_distribution(sleep_data: pd.DataFrame,
                                 save_path: str = 'plots/bedtime_distribution_by_caffeine.png') -> None:
    """
    Analyze distribution of bedtimes comparing caffeine vs no caffeine conditions.

    Args:
        sleep_data: DataFrame containing sleep data with bedtime column
        save_path: Path to save the generated plot
    """
    # Create a copy to avoid modifying the original data
    sleep_data_with_bedtime = sleep_data.copy()
    sleep_data_with_bedtime['bedtime'] = pd.to_datetime(sleep_data_with_bedtime['bedtime'])

    # Convert bedtime to seconds since midnight for numeric representation
    sleep_data_with_bedtime['bedtime_of_day'] = (sleep_data_with_bedtime['bedtime'].dt.hour * 3600 +
                                                 sleep_data_with_bedtime['bedtime'].dt.minute * 60 +
                                                 sleep_data_with_bedtime['bedtime'].dt.second)

    unique_students = sleep_data_with_bedtime["Uid"].unique()
    num_students = len(unique_students)

    fig, axes = plt.subplots(num_students, 1, figsize=(10, 5 * num_students))

    # Handle case with single student
    if num_students == 1:
        axes = [axes]

    # Format x-axis to show times instead of seconds
    def format_time(x, pos):
        hours = int(x // 3600)
        minutes = int((x % 3600) // 60)
        return f'{hours:02d}:{minutes:02d}'

    # Iterate over each unique student (Uid) and subplot
    for i, student_id in enumerate(unique_students):
        df_student = sleep_data_with_bedtime[sleep_data_with_bedtime["Uid"] == student_id]

        bedtime_caff = df_student[df_student["is_coffee"] == True]["bedtime_of_day"]
        bedtime_nocaff = df_student[df_student["is_coffee"] == False]["bedtime_of_day"]

        # Using numeric data for KDE plot
        sns.kdeplot(bedtime_caff, label="Caffeine", fill=True, ax=axes[i])
        sns.kdeplot(bedtime_nocaff, label="No Caffeine", fill=True, ax=axes[i])

        axes[i].xaxis.set_major_formatter(FuncFormatter(format_time))
        axes[i].set_title(f"Bedtime Distribution — Caffeine vs No Caffeine for student {student_id}")
        axes[i].set_xlabel("Bedtime")
        axes[i].set_ylabel("Density")
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()


def daily_bpm_overall(df_bpm):
    plt.figure(figsize=(10, 6))

    # Split into caffeine and no caffeine for all students combined
    bpm_caff = df_bpm[df_bpm["is_coffee"] == True]["bpm"]
    bpm_nocaff = df_bpm[df_bpm["is_coffee"] == False]["bpm"]

    # Plot KDEs
    sns.kdeplot(bpm_caff, label="Caffeine", fill=True, color="deepskyblue", alpha=0.6)
    sns.kdeplot(bpm_nocaff, label="No Caffeine", fill=True, color="orange", alpha=0.6)

    # Titles and labels
    plt.title("Heart Rate Density — Caffeine vs No Caffeine (All Students)", fontsize=14)
    plt.xlabel("Heart Rate (bpm)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/hr_density_aggregated.png", dpi=300)
    # plt.show()

def sleep_bpm_overall(sleep_data):
    plt.figure(figsize=(10, 6))

    # Split into caffeine and no caffeine for all students combined
    bpm_caff = sleep_data[sleep_data["is_coffee"] == True]["avg_hr"]
    bpm_nocaff = sleep_data[sleep_data["is_coffee"] == False]["avg_hr"]
    print(f'!!! Heart Rate During Sleep - Caffeine: {bpm_caff.mean():.2f} bpm, No Caffeine: {bpm_nocaff.mean():.2f} bpm')

    # Plot KDEs
    sns.kdeplot(bpm_caff, label="Caffeine", fill=True)
    sns.kdeplot(bpm_nocaff, label="No Caffeine", fill=True)

    # Titles and labels
    plt.title("Heart Rate During Sleep — Caffeine vs No Caffeine (All Students)", fontsize=14)
    plt.xlabel("Heart Rate (bpm)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/sleep_hr_density_overall.png", dpi=300)
    # plt.show()


def filter_sleep(sleep_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process sleep data for analysis.
    This function was referenced in the original code but not defined.

    Args:
        sleep_data: Raw sleep data DataFrame

    Returns:
        Processed sleep data DataFrame
    """
    # As the original function wasn't defined, this is a placeholder
    # Implement according to your specific filtering requirements
    return sleep_data


def main() -> None:
    """
    Main function to run all analyses.
    """
    # Load data
    df_bpm, df_sleep = load_data()

    # Run analyses
    plot_sleep_duration(df_sleep)
    daily_bpm_overall(df_bpm)
    sleep_bpm_overall(df_sleep)
    analyze_deep_sleep_vs_caffeine(df_sleep)
    visualize_hr_density_by_student(df_bpm)
    analyze_sleep_hr_by_caffeine(filter_sleep(df_sleep))
    visualize_sleep_stages(df_sleep)
    visualize_hr_fluctuation_by_student(df_bpm)
    analyze_awake_count(df_sleep)
    analyze_bedtime_distribution(df_sleep)


if __name__ == "__main__":
    main()