import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob

# Force matplotlib to not use any Xwindows backend for headless HPC plotting
matplotlib.use('Agg')

def merge_and_plot():
    print("Locating and merging MMLPF data...")
    mml_files = glob.glob('mmlpf_cv_*.csv')
    mml_df = pd.concat([pd.read_csv(f) for f in mml_files], ignore_index=True)
    
    print("Locating and merging PsyTrack data...")
    psy_files = glob.glob('psytrack_cv_*.csv')
    psy_df = pd.concat([pd.read_csv(f) for f in psy_files], ignore_index=True)

    print(f"Successfully loaded {len(mml_files)} MMLPF subjects and {len(psy_files)} PsyTrack subjects.")

    # Group by session number to calculate the cohort averages
    mml_avg = mml_df.groupby('test_session_number')['avg_nll_per_trial'].mean().reset_index()
    psy_avg = psy_df.dropna(subset=['avg_nll_per_trial']).groupby('test_session_number')['avg_nll_per_trial'].mean().reset_index()

    # Generate the presentation plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(mml_avg['test_session_number'], mml_avg['avg_nll_per_trial'], 
             color='#5b5cf0', linewidth=3, marker='o', label='MMLPF Architecture')
    plt.plot(psy_avg['test_session_number'], psy_avg['avg_nll_per_trial'], 
             color='gray', linewidth=3, marker='s', linestyle='--', label='PsyTrack Baseline')
    
    plt.title('Chronological Walk-Forward CV: MMLPF vs PsyTrack (10-Mouse Cohort)', fontsize=14, fontweight='bold')
    plt.xlabel('Test Session Number', fontsize=12)
    plt.ylabel('Average Out-of-Sample NLL per Trial\n(Lower is Better)', fontsize=12)
    
    # Ensure all session integers are plotted neatly on the X-axis
    all_sessions = sorted(list(set(mml_avg['test_session_number']).union(set(psy_avg['test_session_number']))))
    plt.xticks(all_sessions)
    
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the final image
    save_path = "final_cohort_comparison_plot.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot successfully generated and saved to '{save_path}'")

if __name__ == "__main__":
    merge_and_plot()