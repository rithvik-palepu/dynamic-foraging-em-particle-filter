import numpy as np
import matplotlib.pyplot as plt
import psytrack as psy
from psytrack.getMAP import getMAP
from scipy.optimize import differential_evolution

# Import the synthetic data generator
from synthetic_drifting_agent import generate_drifting_agent_data

# Import the EM Pipeline from your empirical script
from empirical_drifting_agent_em import calculate_nll_fast, execute_particle_smoother

# ==========================================
# 1. PsyTrack Formatting Helper
# ==========================================
def format_for_psytrack(choices, rewards, history_window=3):
    """Maps binary data into a symmetric GLM design matrix."""
    num_trials = len(choices)
    bipolar_choices = (choices * 2) - 1  # Map 0,1 to -1,+1 for symmetric weights
    
    prev_choices = np.zeros((num_trials, history_window))
    prev_rewards = np.zeros((num_trials, history_window))
    interactions = np.zeros((num_trials, history_window))
    
    for i in range(1, history_window + 1):
        shifted_c = np.concatenate([np.zeros(i), bipolar_choices[:-i]])
        shifted_r = np.concatenate([np.zeros(i), rewards[:-i]])
        
        prev_choices[:, i-1] = shifted_c
        prev_rewards[:, i-1] = shifted_r
        interactions[:, i-1] = shifted_c * shifted_r
        
    inputs = {
        'bias': np.ones((num_trials, 1)),
        'prev_choice': prev_choices,
        'prev_reward': prev_rewards,
        'interaction': interactions
    }
    
    return {
        'y': choices + 1,  # PsyTrack requires 1 and 2
        'inputs': inputs
    }, inputs

# ==========================================
# 2. Probability Conversion Helpers
# ==========================================
def get_psytrack_prob(wMode, inputs, history_window):
    """Converts raw GLM log-odds into choice probabilities via the logistic sigmoid function."""
    all_features = np.hstack([
        inputs['bias'],
        inputs['prev_choice'],
        inputs['prev_reward'],
        inputs['interaction']
    ])
    log_odds = np.sum(wMode.T * all_features, axis=1)
    return 1.0 / (1.0 + np.exp(-log_odds))

# ==========================================
# 3. Main Execution & Sweep
# ==========================================
if __name__ == "__main__":
    print("1. Generating Highly Volatile Synthetic Data...")
    data = generate_drifting_agent_data(num_trials=500, seed=42)
    choices = data['choices']
    rewards = data['rewards']
    
    # Ground Truth Choice Probabilities: P(Choose Right) = 1 / (1 + exp(-beta * (Q_R - Q_L)))
    true_q_diff = data['true_Q'][:, 1] - data['true_Q'][:, 0]
    true_beta = data['true_beta']
    true_prob_right = 1.0 / (1.0 + np.exp(-true_beta * true_q_diff))
    
    print("\n2. Formatting for PsyTrack (History Window = 3)...")
    history_window = 3
    psy_dat, raw_inputs = format_for_psytrack(choices, rewards, history_window=history_window)
    
    weights_dict = {
        'bias': 1, 'prev_choice': history_window, 
        'prev_reward': history_window, 'interaction': history_window
    }
    K = sum(weights_dict.values())
    
    # ---------------------------------------------------------
    # A. The Baseline: Standard PsyTrack (hyperOpt)
    # ---------------------------------------------------------
    print("\n3. Running Standard PsyTrack (hyperOpt Baseline)...")
    hyper_args = {'sigma': [1/2**4] * K, 'sigInit': 2**4., 'sigDay': None}
    
    optList, _, wMode_baseline, _ = psy.hyperOpt(psy_dat, hyper_args, weights_dict, ['sigma'])
    prob_baseline = get_psytrack_prob(wMode_baseline, raw_inputs, history_window)
    corr_baseline = np.corrcoef(prob_baseline, true_prob_right)[0, 1]
    print(f"   -> hyperOpt recovered sigma: {np.mean(optList['sigma']):.6f}")
    print(f"   -> Baseline Probability Correlation: {corr_baseline:.3f}")

    # ---------------------------------------------------------
    # B. The Advantage: Manual Sigma Sweep (Bypassing hyperOpt)
    # ---------------------------------------------------------
    print("\n4. Running Manual Sigma Sweep (Forcing Volatility)...")
    sigma_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_corr = -1.0
    best_prob = None
    best_sigma = None
    
    for sig in sigma_grid:
        forced_hyper = {'sigma': np.array([sig] * K), 'sigInit': np.array([2**4.] * K)}
        
        # Use getMAP directly to calculate weights at the fixed sigma, bypassing evidence optimization
        wMode_sweep, _, _, _ = getMAP(psy_dat, forced_hyper, weights_dict)
        prob_sweep = get_psytrack_prob(wMode_sweep, raw_inputs, history_window)
        
        corr = np.corrcoef(prob_sweep, true_prob_right)[0, 1]
        print(f"   -> Forced Sigma: {sig:<7} | Probability Correlation: {corr:.3f}")
        
        if corr > best_corr:
            best_corr = corr
            best_prob = prob_sweep
            best_sigma = sig

    # ---------------------------------------------------------
    # C. Your Model: EM Particle Filter
    # ---------------------------------------------------------
    print("\n5. Running EM Particle Filter Pipeline...")
    
    # M-Step: Optimize Volatility (Using exact bounds from empirical script)
    m_step_result = differential_evolution(
        func=calculate_nll_fast, 
        bounds=[(0.001, 0.2), (0.001, 0.2)], 
        args=(choices, rewards),
        maxiter=30, popsize=15, tol=0.01, 
        workers=-1, disp=False  
    )
    opt_sigma_alpha, opt_sigma_beta = m_step_result.x
    print(f"   -> M-Step Converged. Sigma Alpha={opt_sigma_alpha:.4f}, Sigma Beta={opt_sigma_beta:.4f}")
    
    # E-Step: Extract Smoothed Trajectories
    est_Q, est_alpha, est_beta = execute_particle_smoother(
        choices, rewards, sigma_alpha=opt_sigma_alpha, sigma_beta=opt_sigma_beta
    )
    
    # Reconstruct EM Choice Probabilities via Softmax: P(Right) = 1 / (1 + exp(-est_beta * (est_Q_R - est_Q_L)))
    em_q_diff = est_Q[:, 1] - est_Q[:, 0]
    prob_em = 1.0 / (1.0 + np.exp(-est_beta * em_q_diff))
    corr_em = np.corrcoef(prob_em, true_prob_right)[0, 1]
    print(f"   -> EM Probability Correlation: {corr_em:.3f}")
    
    # ---------------------------------------------------------
    # 6. Plotting the Defense (Probability Space)
    # ---------------------------------------------------------
    print("\n6. Generating Final Benchmark Plot...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    axes[0].plot(true_prob_right, color='black', lw=2, label=r'True $P(\text{Choose Right})$')
    axes[0].plot(prob_baseline, color='blue', alpha=0.7, label=rf'PsyTrack hyperOpt (r={corr_baseline:.2f})')
    axes[0].set_title("Standard Optimization: Over-regularized to a Static Model")
    axes[0].legend(loc='upper right', frameon=False)
    axes[0].set_ylim(-0.05, 1.05)
    
    axes[1].plot(true_prob_right, color='black', lw=2, label=r'True $P(\text{Choose Right})$')
    axes[1].plot(best_prob, color='teal', alpha=0.7, label=rf'PsyTrack Best Sweep $\sigma={best_sigma}$ (r={best_corr:.2f})')
    axes[1].set_title("Manual Override: Forcing Maximum Volatility")
    axes[1].legend(loc='upper right', frameon=False)
    axes[1].set_ylim(-0.05, 1.05)
    
    axes[2].plot(true_prob_right, color='black', lw=2, label=r'True $P(\text{Choose Right})$')
    axes[2].plot(prob_em, color='darkred', alpha=0.7, label=rf'EM Particle Filter (r={corr_em:.2f})')
    axes[2].set_title("Biological Architecture: Expectation-Maximization Pipeline")
    axes[2].legend(loc='upper right', frameon=False)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("Trials")
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel(r"$P(\text{Choose Right})$")
        
    plt.tight_layout()
    plt.savefig("synthetic_model_benchmark.png", dpi=150)
    print("Benchmark complete. Plot saved to 'synthetic_model_benchmark.png'")