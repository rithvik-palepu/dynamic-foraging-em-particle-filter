import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
import aind_dynamic_foraging_database as db
import psytrack

# Import your core MMLPF objective function directly
from empirical_drifting_agent_mml_pf import calculate_nll_fast

def split_data_clean(choices, rewards, F=5, seed=42):
    """
    Splits session choices and rewards into F folds for cross-validation 
    without relying on broken relative-import files.
    """
    np.random.seed(seed)
    N = len(choices)
    shuffled_array = np.arange(N)
    np.random.shuffle(shuffled_array)
    
    chunk = int(N / F)
    folds = []
    
    for k in range(F):
        test_inds = np.sort(shuffled_array[k * chunk : (k + 1) * chunk])
        train_inds = np.setdiff1d(np.arange(N), test_inds)
        
        folds.append({
            'train_choices': choices[train_inds],
            'train_rewards': rewards[train_inds],
            'test_choices': choices[test_inds],
            'test_rewards': rewards[test_inds],
            'test_inds': test_inds
        })
    return folds

def run_comparative_cv(session_choices, session_rewards, F=5):
    """
    Evaluates both PsyTrack and MMLPF across F cross-validation folds.
    """
    folds = split_data_clean(session_choices, session_rewards, F=F, seed=42)
    fold_results = []
    
    weight_dict = {'bias': 1, 'reward_history': 1}
    hyper_guess = {'sigma': [2**-5, 2**-5], 'sigInit': 2**5, 'sigDay': 2**5}
    optList = ['sigma']
    
    for f, fold in enumerate(folds):
        print(f"--- Evaluating Fold {f+1} of {F} ---")
        
        train_c, train_r = fold['train_choices'], fold['train_rewards']
        test_c, test_r = fold['test_choices'], fold['test_rewards']
        
        if len(test_c) == 0:
            continue
            
        # --- A. Evaluate PsyTrack ---
        # Format training data for PsyTrack dictionary requirements
        y_train = train_c + 1
        past_c_tr = np.insert(train_c[:-1], 0, 0)
        past_r_tr = np.insert(train_r[:-1], 0, 0)
        rew_feat_tr = np.zeros(len(train_c))
        rew_feat_tr[(past_c_tr == 1) & (past_r_tr == 1)] = 1
        rew_feat_tr[(past_c_tr == 0) & (past_r_tr == 1)] = -1
        
        D_train = {
            'y': y_train,
            'inputs': {
                'bias': np.ones((len(train_c), 1)),
                'reward_history': rew_feat_tr.reshape(-1, 1)
            },
            'dayLength': np.array([len(train_c)])
        }
        
        # Format test data for PsyTrack
        y_test = test_c + 1
        past_c_te = np.insert(test_c[:-1], 0, 0)
        past_r_te = np.insert(test_r[:-1], 0, 0)
        rew_feat_te = np.zeros(len(test_c))
        rew_feat_te[(past_c_te == 1) & (past_r_te == 1)] = 1
        rew_feat_te[(past_c_te == 0) & (past_r_te == 1)] = -1
        
        D_test = {
            'y': y_test,
            'inputs': {
                'bias': np.ones((len(test_c), 1)),
                'reward_history': rew_feat_te.reshape(-1, 1)
            },
            'dayLength': np.array([len(test_c)])
        }
        
        try:
            _, _, wMode, _ = psytrack.hyperOpt(
                D_train, hyper_guess, weight_dict, optList, showOpt=0
            )
            g = psytrack.helper.helperFunctions.read_input(D_test, weight_dict)
            w_last = wMode[:, -1]
            
            psy_nll = 0.0
            for t in range(len(test_c)):
                gw = g[t] @ w_last
                yt = int(D_test['y'][t]) - 1
                logli = yt * gw - np.logaddexp(0, gw)
                psy_nll -= logli
            psy_avg_nll = psy_nll / len(test_c)
        except Exception as e:
            print(f"PsyTrack failed on Fold {f+1}: {e}")
            psy_avg_nll = np.nan

        # --- B. Evaluate MMLPF (Using imported calculate_nll_fast) ---
        m_step_result = differential_evolution(
            func=calculate_nll_fast, 
            bounds=[(0.001, 0.2), (0.001, 0.2)], 
            args=(train_c, train_r),
            maxiter=15, popsize=6, tol=0.01, disp=False
        )
        opt_sa, opt_sb = m_step_result.x
        
        mml_nll = calculate_nll_fast([opt_sa, opt_sb], test_c, test_r, num_particles=300)
        mml_avg_nll = mml_nll / len(test_c)
        
        fold_results.append({
            "fold": f + 1,
            "PsyTrack_NLL": psy_avg_nll,
            "MMLPF_NLL": mml_avg_nll
        })
        
    return pd.DataFrame(fold_results)

def plot_and_save_comparison(results_df, save_path="mml_vs_psytrack_fold_comparison.png"):
    """
    Generates, saves, and displays the comparative k-fold validation performance plot.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(
        results_df['fold'], results_df['MMLPF_NLL'], 
        marker='o', linewidth=2.5, color='blue', label='MMLPF Architecture (Yours)'
    )
    
    # Isolate PsyTrack data and drop NaNs from singular matrix failures to ensure continuous plotting
    psy_plot_df = results_df.dropna(subset=['PsyTrack_NLL'])
    
    plt.plot(
        psy_plot_df['fold'], psy_plot_df['PsyTrack_NLL'], 
        marker='s', linestyle='--', linewidth=2.5, color='red', label='PsyTrack Baseline'
    )
    
    plt.title("Comparative k-Fold Cross-Validation: MMLPF vs PsyTrack", fontsize=14)
    plt.xlabel("Cross-Validation Fold", fontsize=12)
    plt.ylabel("Average Out-of-Sample NLL per Trial\n(Lower is Better)", fontsize=12)
    plt.xticks(results_df['fold'])
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"\nComparison plot successfully saved to '{save_path}'")
    plt.show()

if __name__ == "__main__":
    print("Querying AIND Database for sample session...")
    cohort_query = "task LIKE '%Uncoupled%' AND foraging_eff > 0.8 AND finished_trials > 300"
    all_sessions = db.select_sessions(where=cohort_query)
    
    sample_session = all_sessions.head(1)
    trials_df = db.fetch_trials(sample_session, columns=["animal_response", "earned_reward"])
    valid_trials = trials_df[trials_df['animal_response'] != 2].copy()
    
    choices = valid_trials['animal_response'].astype(int).values
    rewards = valid_trials['earned_reward'].astype(int).values
    
    print(f"Running comparative cross-validation across 5 folds on {len(choices)} trials...")
    results_df = run_comparative_cv(choices, rewards, F=5)
    
    print("\nCross-Validation Results Summary:")
    print(results_df)
    
    plot_and_save_comparison(results_df)