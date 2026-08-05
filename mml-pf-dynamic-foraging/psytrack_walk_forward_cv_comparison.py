import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psytrack
import aind_dynamic_foraging_database as db
from psytrack.helper.helperFunctions import read_input

# Import the MMLPF walk-forward function to compartmentalize and reduce lines of code[cite: 4]
from cross_validation import execute_chronological_walk_forward

# ==========================================
# 1. PsyTrack Data Formatting
# ==========================================
def build_psytrack_dict(choices, rewards):
    """
    Formats raw choice and reward arrays into the specific dictionary structure 
    PsyTrack requires, ensuring strict (N, 1) matrix dimensions[cite: 3].
    """
    y = choices + 1 
    
    past_choice = np.insert(choices[:-1], 0, 0)
    past_reward = np.insert(rewards[:-1], 0, 0)
    
    reward_feature = np.zeros(len(choices))
    reward_feature[(past_choice == 1) & (past_reward == 1)] = 1
    reward_feature[(past_choice == 0) & (past_reward == 1)] = -1
    
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
    Negative Log-Likelihood on the subsequent chronological session[cite: 3].
    """
    results = []
    cumulative_train_trials = 0
    
    for i in range(1, len(session_data_list)):
        train_choices = np.concatenate([s[0] for s in session_data_list[:i]])
        train_rewards = np.concatenate([s[1] for s in session_data_list[:i]])
        cumulative_train_trials = len(train_choices)
        
        D_train = build_psytrack_dict(train_choices, train_rewards)
        
        test_choices, test_rewards = session_data_list[i]
        D_test = build_psytrack_dict(test_choices, test_rewards)
        
        print(f"  [PsyTrack Train: {i} Sessions | {cumulative_train_trials} Trials] --> Predicting Session {i+1} ({len(test_choices)} trials)")
        
        try:
            hyp, evd, wMode, hess_info = psytrack.hyperOpt(
                D_train, hyper_guess, weight_dict, optList, showOpt=0  
            )
            
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
    Loads both output CSVs, calculates the cohort averages, and generates the comparison plot[cite: 3].
    """
    try:
        psy_df = pd.read_csv(psy_csv_path)
        mml_df = pd.read_csv(mml_csv_path)
        
        psy_avg = psy_df.dropna(subset=['avg_nll_per_trial']).groupby('test_session_number')['avg_nll_per_trial'].mean().reset_index()
        mml_avg = mml_df.groupby('test_session_number')['avg_nll_per_trial'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(
            mml_avg['test_session_number'], mml_avg['avg_nll_per_trial'], 
            color='blue', linewidth=3, marker='o', label='MMLPF Architecture (Yours)'
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
    except Exception as e:
        print(f"\nError generating comparison plot: {e}")

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    print("Querying AIND Database for expanded HPC cohort...")
    
    # Relaxed parameters: 0.65 efficiency to capture a more representative population 
    cohort_query = "task LIKE '%Uncoupled%' AND foraging_eff > 0.65 AND finished_trials > 300"
    all_sessions = db.select_sessions(where=cohort_query)
    all_sessions = all_sessions.sort_values(by=['subject_id', 'session_date'])
    
    # Require at least 4 sessions, but remove the maximum cap for the HPC 
    session_counts = all_sessions['subject_id'].value_counts()
    valid_subjects = session_counts[session_counts >= 4].index.unique()
    
    # Expand testing pool to 20 mice for the cluster 
    subjects_to_test = valid_subjects[:20]
    test_sessions = all_sessions[all_sessions['subject_id'].isin(subjects_to_test)]
    
    trials_df = db.fetch_trials(test_sessions, columns=["animal_response", "earned_reward"])
    valid_trials_df = trials_df[trials_df['animal_response'] != 2].copy()
    
    weight_dict = {'bias': 1, 'reward_history': 1}
    hyper_guess = {'sigma': [2**-5, 2**-5], 'sigInit': 2**5, 'sigDay': 2**5}
    optList = ['sigma']
    
    all_psytrack_results = []
    all_mml_results = []
    
    for subject_id, subject_data in valid_trials_df.groupby('subject_id'):
        print(f"\n==========================================")
        print(f"Executing Dual CV for Subject: {subject_id}")
        print(f"==========================================")
        
        session_data_list = []
        for (session_date, session_id), session_data in subject_data.groupby(['session_date', 'session_id']):
            choices = session_data['animal_response'].astype(int).values
            rewards = session_data['earned_reward'].astype(int).values
            session_data_list.append((choices, rewards))
            
        print(f"Loaded {len(session_data_list)} consecutive sessions.\n")
        
        if len(session_data_list) > 1:
            # 1. Run MMLPF Walk-Forward[cite: 4]
            print(">>> Starting MMLPF Architecture Evaluation...")
            subject_mml_df = execute_chronological_walk_forward(session_data_list)
            subject_mml_df.insert(0, 'subject_id', subject_id)
            all_mml_results.append(subject_mml_df)
            
            # 2. Run PsyTrack Walk-Forward[cite: 3]
            print("\n>>> Starting PsyTrack Baseline Evaluation...")
            subject_psy_df = execute_psytrack_chronological_cv(session_data_list, weight_dict, hyper_guess, optList)
            subject_psy_df.insert(0, 'subject_id', subject_id)
            all_psytrack_results.append(subject_psy_df)
            
    if all_mml_results and all_psytrack_results:
        # Save MMLPF Results[cite: 4]
        final_mml_df = pd.concat(all_mml_results, ignore_index=True)
        mml_export = "mmlpf_chronological_walk_forward_cv.csv"
        final_mml_df.to_csv(mml_export, index=False)
        
        # Save PsyTrack Results[cite: 3]
        final_psy_df = pd.concat(all_psytrack_results, ignore_index=True)
        psy_export = "psytrack_chronological_walk_forward_cv.csv"
        final_psy_df.to_csv(psy_export, index=False)
        
        print(f"\nBatch processing complete for all subjects!")
        
        plot_and_save_comparison(psy_export, mml_export)