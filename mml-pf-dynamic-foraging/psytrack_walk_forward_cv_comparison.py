import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psytrack
import aind_dynamic_foraging_database as db
from psytrack.helper.helperFunctions import read_input

# ==========================================
# 1. PsyTrack Data Formatting
# ==========================================
def build_psytrack_dict(choices, rewards):
    """
    Formats raw choice and reward arrays into the specific dictionary structure 
    PsyTrack requires, ensuring strict (N, 1) matrix dimensions.
    """
    # Shift 0/1 choices to 1/2 for PsyTrack's internal math
    y = choices + 1 
    
    # Calculate the Win-Stay/Lose-Shift feature
    past_choice = np.insert(choices[:-1], 0, 0)
    past_reward = np.insert(rewards[:-1], 0, 0)
    
    reward_feature = np.zeros(len(choices))
    reward_feature[(past_choice == 1) & (past_reward == 1)] = 1
    reward_feature[(past_choice == 0) & (past_reward == 1)] = -1
    
    # FIX: PsyTrack strictly requires input features to be 2D arrays (N, 1)
    # Using np.ones((N, 1)) and .reshape(-1, 1) forces this alignment.
    inputs = {
        'bias': np.ones((len(y), 1)),
        'reward_history': reward_feature.reshape(-1, 1)
    }
    
    return {'y': y, 'inputs': inputs, 'dayLength': np.array([len(y)])}

# ==========================================
# 2. Chronological Walk-Forward Execution (PsyTrack)
# ==========================================
def execute_psytrack_chronological_cv(session_data_list, weight_dict, hyper_guess, optList):
    """
    Trains PsyTrack on accumulated past sessions and tests its out-of-sample 
    Negative Log-Likelihood on the subsequent chronological session.
    """
    results = []
    cumulative_train_trials = 0
    
    # Start testing at Session index 1, so we have Session 0 to train on
    for i in range(1, len(session_data_list)):
        
        # --- Prepare Cumulative Training Data ---
        train_choices = np.concatenate([s[0] for s in session_data_list[:i]])
        train_rewards = np.concatenate([s[1] for s in session_data_list[:i]])
        cumulative_train_trials = len(train_choices)
        
        D_train = build_psytrack_dict(train_choices, train_rewards)
        
        # --- Prepare Upcoming Test Data ---
        test_choices, test_rewards = session_data_list[i]
        D_test = build_psytrack_dict(test_choices, test_rewards)
        
        print(f"  [Train: {i} Sessions | {cumulative_train_trials} Trials] --> Predicting Session {i+1} ({len(test_choices)} trials)")
        
        # --- Phase 1: Train PsyTrack ---
        try:
            hyp, evd, wMode, hess_info = psytrack.hyperOpt(
                D_train, 
                hyper_guess, 
                weight_dict, 
                optList, 
                showOpt=0  
            )
            
            # --- Phase 2: Test Out-of-Sample NLL ---
            w_last = wMode[:, -1]
            g = read_input(D_test, weight_dict)
            
            test_nll = 0.0
            for t in range(len(test_choices)):
                gw = g[t] @ w_last
                yt = int(D_test['y'][t]) - 1  
                
                logli = yt * gw - np.logaddexp(0, gw)
                test_nll -= logli
                
            avg_nll_per_trial = test_nll / len(test_choices)
            
        except Exception as e:
            print(f"  -> PsyTrack evaluation skipped on Session {i+1} due to: {e}")
            test_nll = np.nan
            avg_nll_per_trial = np.nan
        
        results.append({
            "test_session_number": i + 1,
            "cumulative_train_trials": cumulative_train_trials,
            "out_of_sample_nll": test_nll,
            "test_trials": len(test_choices),
            "avg_nll_per_trial": avg_nll_per_trial
        })
        
    return pd.DataFrame(results)

# ==========================================
# 3. Plotting & Comparison
# ==========================================
def plot_and_save_comparison(psy_csv_path, mml_csv_path, save_path="mml_vs_psytrack_chronological_comparison.png"):
    """
    Loads both output CSVs, calculates the cohort averages, and generates the comparison plot.
    """
    try:
        psy_df = pd.read_csv(psy_csv_path)
        mml_df = pd.read_csv(mml_csv_path)
        
        # Drop NaNs before aggregating to prevent singular matrix failures from breaking the line
        psy_avg = psy_df.dropna(subset=['avg_nll_per_trial']).groupby('test_session_number')['avg_nll_per_trial'].mean().reset_index()
        mml_avg = mml_df.groupby('test_session_number')['avg_nll_per_trial'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(
            mml_avg['test_session_number'], mml_avg['avg_nll_per_trial'], 
            color='blue', linewidth=3, marker='o', label='MMLPF'
        )
        plt.plot(
            psy_avg['test_session_number'], psy_avg['avg_nll_per_trial'], 
            color='red', linewidth=3, marker='s', linestyle='--', label='PsyTrack Baseline'
        )
        
        plt.title('Chronological Walk-Forward CV: MML vs PsyTrack (Cohort Average)', fontsize=14)
        plt.xlabel('Test Session Number (Chronological)', fontsize=12)
        plt.ylabel('Average Out-of-Sample NLL per Trial\n(Lower is Better)', fontsize=12)
        
        all_sessions = sorted(list(set(mml_avg['test_session_number']).union(set(psy_avg['test_session_number']))))
        plt.xticks(all_sessions)
        
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300)
        print(f"\nComparison plot successfully saved to '{save_path}'")
        plt.show()
    except Exception as e:
        print(f"\nError generating comparison plot: {e}")

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    print("Querying AIND Database for highly trained cohort...")
    
    cohort_query = "task LIKE '%Uncoupled%' AND foraging_eff > 0.8 AND finished_trials > 300"
    all_sessions = db.select_sessions(where=cohort_query)
    all_sessions = all_sessions.sort_values(by=['subject_id', 'session_date'])
    
    session_counts = all_sessions['subject_id'].value_counts()
    valid_subjects = session_counts[session_counts.between(4, 10)].index.unique()
    
    subjects_to_test = valid_subjects[:2]
    test_sessions = all_sessions[all_sessions['subject_id'].isin(subjects_to_test)]
    
    trials_df = db.fetch_trials(test_sessions, columns=["animal_response", "earned_reward"])
    valid_trials_df = trials_df[trials_df['animal_response'] != 2].copy()
    
    weight_dict = {'bias': 1, 'reward_history': 1}
    hyper_guess = {'sigma': [2**-5, 2**-5], 'sigInit': 2**5, 'sigDay': 2**5}
    optList = ['sigma']
    
    all_psytrack_results = []
    
    for subject_id, subject_data in valid_trials_df.groupby('subject_id'):
        print(f"\n==========================================")
        print(f"Executing PsyTrack Chronological CV for Subject: {subject_id}")
        print(f"==========================================")
        
        session_data_list = []
        for (session_date, session_id), session_data in subject_data.groupby(['session_date', 'session_id']):
            choices = session_data['animal_response'].astype(int).values
            rewards = session_data['earned_reward'].astype(int).values
            session_data_list.append((choices, rewards))
            
        print(f"Loaded {len(session_data_list)} consecutive sessions.")
        
        if len(session_data_list) > 1:
            subject_cv_df = execute_psytrack_chronological_cv(session_data_list, weight_dict, hyper_guess, optList)
            subject_cv_df.insert(0, 'subject_id', subject_id)
            all_psytrack_results.append(subject_cv_df)
            
    if all_psytrack_results:
        final_cv_df = pd.concat(all_psytrack_results, ignore_index=True)
        export_filename = "psytrack_chronological_cv.csv"
        final_cv_df.to_csv(export_filename, index=False)
        print(f"\nCross-validation complete! Results saved to '{export_filename}'")
        
        # Run comparison plot automatically
        mml_baseline_path = "data/chronological_walk_forward_cv.csv"
        plot_and_save_comparison(export_filename, mml_baseline_path)