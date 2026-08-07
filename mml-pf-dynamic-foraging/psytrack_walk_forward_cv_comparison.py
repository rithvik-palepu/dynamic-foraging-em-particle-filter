import argparse
import sys
import numpy as np
import pandas as pd
import psytrack
import aind_dynamic_foraging_database as db

from cross_validation import execute_chronological_walk_forward

# ==========================================
# 1. PsyTrack Data Formatting
# ==========================================
def build_psytrack_dict(choices, rewards):
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
# 2. Chronological Walk-Forward Execution 
# ==========================================
def execute_psytrack_chronological_cv(session_data_list, weight_dict, hyper_guess, optList):
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
            g = psytrack.helper.helperFunctions.read_input(D_test, weight_dict)
            
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
# 3. Main Parallel Execution
# ==========================================
if __name__ == "__main__":
    # Setup command line argument for SLURM Job Array
    parser = argparse.ArgumentParser()
    parser.add_argument('--array_id', type=int, required=True, help="SLURM array task ID (e.g., 0-9)")
    args = parser.parse_args()
    
    print("Querying AIND Database for available subjects...")
    cohort_query = "task LIKE '%Uncoupled%' AND foraging_eff > 0.65 AND finished_trials > 300"
    all_sessions = db.select_sessions(where=cohort_query)
    all_sessions = all_sessions.sort_values(by=['subject_id', 'session_date'])
    
    # Require at least 20 sessions for robust longitudinal data
    session_counts = all_sessions['subject_id'].value_counts()
    valid_subjects = session_counts[session_counts >= 20].index.sort_values() 
    
    # Graceful exit if array size exceeds available mice
    if args.array_id >= len(valid_subjects):
        print(f"Error: Array ID {args.array_id} exceeds available valid subjects ({len(valid_subjects)}). Skipping.")
        sys.exit(0)
        
    subject_id = valid_subjects[args.array_id]
    print(f"\n==========================================")
    print(f"Executing Dual CV for Subject: {subject_id} (Array ID: {args.array_id})")
    print(f"==========================================")
    
    # Isolate target mouse and enforce 25-session maximum cap
    subject_sessions = all_sessions[all_sessions['subject_id'] == subject_id].head(25)
    
    trials_df = db.fetch_trials(subject_sessions, columns=["animal_response", "earned_reward"])
    valid_trials_df = trials_df[trials_df['animal_response'] != 2].copy()
    
    weight_dict = {'bias': 1, 'reward_history': 1}
    hyper_guess = {'sigma': [2**-5, 2**-5], 'sigInit': 2**5, 'sigDay': 2**5}
    optList = ['sigma']
    
    session_data_list = []
    for (session_date, session_id), session_data in valid_trials_df.groupby(['session_date', 'session_id']):
        choices = session_data['animal_response'].astype(int).values
        rewards = session_data['earned_reward'].astype(int).values
        session_data_list.append((choices, rewards))
        
    print(f"Loaded {len(session_data_list)} consecutive sessions (Capped at max 25).\n")
    
    if len(session_data_list) > 1:
        print(">>> Starting MMLPF Architecture Evaluation...")
        subject_mml_df = execute_chronological_walk_forward(session_data_list)
        subject_mml_df.insert(0, 'subject_id', subject_id)
        
        print("\n>>> Starting PsyTrack Baseline Evaluation...")
        subject_psy_df = execute_psytrack_chronological_cv(session_data_list, weight_dict, hyper_guess, optList)
        subject_psy_df.insert(0, 'subject_id', subject_id)
        
        # Save files with subject_id appended to prevent overwriting
        mml_export = f"mmlpf_cv_{subject_id}.csv"
        subject_mml_df.to_csv(mml_export, index=False)
        
        psy_export = f"psytrack_cv_{subject_id}.csv"
        subject_psy_df.to_csv(psy_export, index=False)
        
        print(f"\nParallel processing complete for Subject {subject_id}!")