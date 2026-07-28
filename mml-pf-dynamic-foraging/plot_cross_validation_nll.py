import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_chronological_cv(csv_filepath="data/chronological_walk_forward_cv.csv", save_path="chronological_cv_curve.png"):
    """
    Reads the chronological CV results and generates the walk-forward learning curve.
    """
    # 1. Load the Data
    df = pd.read_csv(csv_filepath)
    
    # 2. Initialize the Figure
    plt.figure(figsize=(10, 6))
    
    # 3. Plot Individual Subject Trajectories
    sns.lineplot(
        data=df, 
        x='test_session_number', 
        y='avg_nll_per_trial', 
        hue='subject_id', 
        marker='o', 
        palette='viridis'
    )
    
    # 4. Calculate and Plot the Cohort Average
    avg_df = df.groupby('test_session_number')['avg_nll_per_trial'].mean().reset_index()
    plt.plot(
        avg_df['test_session_number'], 
        avg_df['avg_nll_per_trial'], 
        color='red', 
        linewidth=3, 
        linestyle='--', 
        label='Cohort Average'
    )
    
    # 5. Formatting and Aesthetics
    plt.title('Chronological Walk-Forward CV: Out-of-Sample NLL vs Session', fontsize=14)
    plt.xlabel('Test Session Number (Chronological)', fontsize=12)
    plt.ylabel('Average Out-of-Sample NLL per Trial\n(Lower is Better)', fontsize=12)
    
    # Ensure x-axis ticks represent discrete session numbers
    plt.xticks(sorted(df['test_session_number'].unique()))
    
    plt.legend(title='Subject ID')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 6. Export
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to '{save_path}'")
    plt.show()

if __name__ == "__main__":
    plot_chronological_cv()