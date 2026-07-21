import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psytrack as psy

# ==========================================
# 1. Data Transformation
# ==========================================
def transform_to_psytrack_dict_t1(session_df):
    raw_choices = session_df['animal_response'].astype(int).values
    y = raw_choices + 1  # Map 0,1 to 1,2
    
    raw_rewards = session_df['earned_reward'].astype(int).values
    
    centered_choices = np.where(raw_choices == 1, 1, -1)
    centered_rewards = np.where(raw_rewards == 1, centered_choices, 0)
    
    num_trials = len(raw_choices)
    window_size = 1
    choice_history = np.zeros((num_trials, window_size))
    reward_history = np.zeros((num_trials, window_size))
    
    choice_history[1:, 0] = centered_choices[:-1]
    reward_history[1:, 0] = centered_rewards[:-1]
        
    inputs = {
        'choice_history': choice_history,
        'reward_history': reward_history
    }
    
    session_id = session_df['session_id'].iloc[0]
    return {
        'name': session_id,
        'y': y,
        'inputs': inputs
    }

# ==========================================
# 2. Main Execution & Overlay Plotting
# ==========================================
if __name__ == "__main__":
    print("Loading EM trajectory data...")
    df = pd.read_csv("batch_em_trajectories.csv")
    
    session_id = df['session_id'].unique()[0]
    session_df = df[df['session_id'] == session_id].copy()
    
    print(f"Transforming Session: {session_id} for PsyTrack (t-1 Only)...")
    psy_dict = transform_to_psytrack_dict_t1(session_df)
    
    window_size = 1
    weights_dict = {
        'bias': 1,
        'choice_history': window_size, 
        'reward_history': window_size
    }
    
    K = np.sum([weights_dict[k] for k in weights_dict.keys()])
    hyper_args = {
        'sigInit': 2**4.,      
        'sigma': [1/2**4] * K,
        'sigDay': None
    }
    
    print("Running PsyTrack Optimization & Fitting...")
    optList, evd, W, hess_info = psy.hyperOpt(psy_dict, hyper_args, weights_dict, ['sigma'])
    
    bias_weight = W[0, :]
    choice_weights = W[1:1+window_size, :]
    reward_weights = W[1+window_size:, :]
    agg_reward_weight = reward_weights[0, :]
    
    est_beta = session_df['est_beta'].values
    est_alpha = session_df['est_alpha'].values
    choices = session_df['animal_response'].values
    rewards = session_df['earned_reward'].values
    trials = np.arange(len(est_beta))
    
    print("Generating Benchmark Overlay Plot with Raw Behavior...")
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1.5, 1.5, 1.5]})
    
    # Panel 0: Raw Choices and Rewards
    axes[0].set_ylim(-0.5, 1.5)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Left', 'Right'])
    
    unrewarded = rewards == 0
    rewarded = rewards > 0
    axes[0].scatter(trials[unrewarded], choices[unrewarded], color='black', s=10, marker='.')
    axes[0].scatter(trials[rewarded], choices[rewarded], edgecolors='black', facecolors='none', s=20, marker='o')
    axes[0].set_title(f"Framework Comparison vs. Raw Behavior: {session_id}")

    # Panel 1: EM Beta vs PsyTrack Aggregate Reward Weight
    axes[1].plot(trials, est_beta, color='black', label=r'EM Inverse Temp ($\beta$)')
    axes[1].set_ylabel(r'EM $\beta$', color='black')
    axes[1].tick_params(axis='y', labelcolor='black')
    axes[1].set_yscale('log')
    
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(trials, agg_reward_weight, color='blue', alpha=0.7, label='PsyTrack t-1 Reward Weight')
    ax1_twin.set_ylabel('PsyTrack Reward Weight', color='blue')
    ax1_twin.tick_params(axis='y', labelcolor='blue')
    
    # Panel 2: PsyTrack Individual Reward Weights
    axes[2].plot(trials, reward_weights[0, :], color='teal', label='Lag t-1')
    axes[2].set_ylabel('Individual Reward Weights')
    axes[2].legend(loc='upper right', fontsize='small', frameon=False)
    
    # Panel 3: EM Alpha (Learning Rate)
    axes[3].plot(trials, est_alpha, color='black', label=r'EM Learning Rate ($\alpha$)')
    axes[3].set_ylabel(r'EM $\alpha$')
    axes[3].set_yscale('log')
    axes[3].set_xlabel('Trials')
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        if ax != axes[1]:  
            ax.spines['right'].set_visible(False)
            
    ax1_twin.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"benchmark_raw_t1_{session_id}.png", dpi=150)
    print(f"Benchmark plot saved to benchmark_raw_t1_{session_id}.png")