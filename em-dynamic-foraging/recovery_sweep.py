import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm 
from scipy.optimize import differential_evolution

# Import your EM functions from your main file (ensure these functions have the updated matched priors)
# from empirical_drifting_agent_em_2 import calculate_nll_fast, execute_particle_smoother

# ==========================================
# 1. EM Functions (Inlined for the sweep)
# ==========================================
def calculate_nll_fast(hyperparams, choices, rewards, num_particles=500):
    np.random.seed(42)  # Temporarily converts objective function into a deterministic function for DE optimization
    sigma_alpha, sigma_beta = hyperparams
    num_trials = len(choices)
    
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.uniform(0, 1, num_particles) 
    particles[:, 1] = np.random.uniform(0, 1, num_particles) 
    
    # MATCHED PRIOR INITIALIZATION
    particles[:, 2] = np.random.normal(np.log(0.3), 0.5, num_particles) 
    particles[:, 3] = np.random.normal(np.log(15.0), 0.5, num_particles)  
    
    nll = 0.0

    for t in range(num_trials):
        particles[:, 2] += np.random.normal(0, sigma_alpha, num_particles)
        particles[:, 3] += np.random.normal(0, sigma_beta, num_particles)

        alpha_vals = np.clip(np.exp(particles[:, 2]), 1e-4, 0.99)
        beta_vals = np.clip(np.exp(particles[:, 3]), 0.01, 100.0) 
        Q_vals = particles[:, :2]

        max_Q = np.max(Q_vals, axis=1, keepdims=True)
        exp_Q = np.exp(beta_vals[:, None] * (Q_vals - max_Q))
        probs = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
        
        p_choice = probs[np.arange(num_particles), choices[t]]
        nll -= np.log(np.mean(p_choice) + 1e-16)
        
        weights = p_choice / (np.sum(p_choice) + 1e-16)
        if np.sum(weights) < 1e-20:
             weights = np.ones(num_particles) / num_particles
        weights /= np.sum(weights)

        idx = np.random.choice(num_particles, size=num_particles, p=weights)
        particles = particles[idx].copy()
        
        alpha_vals_resampled = np.clip(np.exp(particles[:, 2]), 1e-4, 0.99)
        particles[:, choices[t]] += alpha_vals_resampled * (rewards[t] - particles[:, choices[t]])
        
    return nll

def execute_particle_smoother(choices, rewards, sigma_alpha, sigma_beta, num_particles=1500, lag=15):
    num_trials = len(choices)
    
    particles_hist = np.zeros((num_trials, num_particles, 4))
    parent_indices = np.zeros((num_trials, num_particles), dtype=int)
    
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.uniform(0, 1, num_particles) 
    particles[:, 1] = np.random.uniform(0, 1, num_particles) 
    
    # MATCHED PRIOR INITIALIZATION
    particles[:, 2] = np.random.normal(np.log(0.3), 0.5, num_particles) 
    particles[:, 3] = np.random.normal(np.log(15.0), 0.5, num_particles)  

    # --- Phase 1: Forward Pass ---
    for t in range(num_trials):
        particles[:, 2] += np.random.normal(0, sigma_alpha, num_particles)
        particles[:, 3] += np.random.normal(0, sigma_beta, num_particles)

        alpha_vals = np.clip(np.exp(particles[:, 2]), 1e-4, 0.99)
        beta_vals = np.clip(np.exp(particles[:, 3]), 0.01, 100.0) 
        Q_vals = particles[:, :2]

        max_Q = np.max(Q_vals, axis=1, keepdims=True)
        exp_Q = np.exp(beta_vals[:, None] * (Q_vals - max_Q))
        probs = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
        
        p_choice = probs[np.arange(num_particles), choices[t]]
        weights = p_choice / (np.sum(p_choice) + 1e-16)
        if np.sum(weights) < 1e-20:
            weights = np.ones(num_particles) / num_particles
        weights /= np.sum(weights)
        
        particles_hist[t] = particles.copy()

        idx = np.random.choice(num_particles, size=num_particles, p=weights)
        parent_indices[t] = idx
        particles = particles[idx].copy()
        
        alpha_vals_resampled = np.clip(np.exp(particles[:, 2]), 1e-4, 0.99)
        particles[:, choices[t]] += alpha_vals_resampled * (rewards[t] - particles[:, choices[t]])
        
    # --- Phase 2: Fixed-Lag Smoothing (Prevents Lineage Collapse) ---
    smoothed_particles = np.zeros((num_trials, num_particles, 4))
    
    for t in range(num_trials):
        target_t = min(t + lag, num_trials - 1)
        current_idx = np.arange(num_particles)
        
        for step in range(target_t - 1, t - 1, -1):
            current_idx = parent_indices[step][current_idx]
            
        smoothed_particles[t] = particles_hist[t][current_idx]
        
    est_Q = np.mean(smoothed_particles[:, :, :2], axis=1)
    
    # Return the full un-aggregated particles for alpha and beta to calculate Confidence Intervals
    return est_Q, smoothed_particles[:, :, 2], smoothed_particles[:, :, 3]

# ==========================================
# 2. Synthetic Generator
# ==========================================
def generate_aind_coupled_data(true_sigma_alpha, true_sigma_beta, seed, num_trials=500):
    """Simulates a drifting agent in an AIND-matched coupled baiting task."""
    np.random.seed(seed) 
    
    true_alpha = np.zeros(num_trials)
    true_beta = np.zeros(num_trials)
    true_Q = np.zeros((num_trials, 2))
    choices = np.zeros(num_trials, dtype=int)
    rewards = np.zeros(num_trials, dtype=int)
    
    current_alpha_logit = np.log(0.3)
    current_beta_logit = np.log(15.0)
    current_Q = np.array([0.5, 0.5])
    
    # AIND Family 3 probabilities (coupled, summing to 1.0)
    prob_family = [
        [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], 
        [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]
    ]
    
    p_reward = np.zeros((num_trials, 2))
    current_trial = 0
    
    while current_trial < num_trials:
        block_len = int(np.random.exponential(scale=40))
        block_len = np.clip(block_len, 20, 80) 
        
        idx = np.random.choice(len(prob_family))
        probs = prob_family[idx]
        
        end_trial = min(current_trial + block_len, num_trials)
        p_reward[current_trial:end_trial] = probs
        current_trial = end_trial
        
    baited_rewards = np.array([0, 0])
    
    for t in range(num_trials):
        current_alpha_logit += np.random.normal(0, true_sigma_alpha)
        current_beta_logit += np.random.normal(0, true_sigma_beta)
        
        alpha_t = np.clip(np.exp(current_alpha_logit), 1e-4, 0.99)
        beta_t = np.clip(np.exp(current_beta_logit), 0.01, 100.0)
        
        true_alpha[t] = alpha_t
        true_beta[t] = beta_t
        true_Q[t] = current_Q.copy()
        
        max_Q = np.max(current_Q)
        exp_Q = np.exp(beta_t * (current_Q - max_Q))
        choice_probs = exp_Q / np.sum(exp_Q)
        
        choice = np.random.choice([0, 1], p=choice_probs)
        choices[t] = choice
        
        new_rewards = (np.random.rand(2) < p_reward[t]).astype(int)
        baited_rewards = np.clip(baited_rewards + new_rewards, 0, 1)
        
        reward = baited_rewards[choice]
        rewards[t] = reward
        baited_rewards[choice] = 0 
        
        current_Q[choice] += alpha_t * (reward - current_Q[choice])
        
    return choices, rewards, true_alpha, true_beta

# ==========================================
# 3. Visualization and Execution Loop
# ==========================================
def plot_recovery_results(df, true_s_alpha, true_s_beta, t_alpha, e_alpha_particles, t_beta, e_beta_particles):
    """Generates a 2x3 grid showing hyperparameters, point estimates, and swarm confidence intervals."""
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Hyperparameter Recovery (Histograms)
    axs[0, 0].hist(df['rec_sigma_alpha'], bins=10, color='skyblue', edgecolor='black')
    axs[0, 0].axvline(true_s_alpha, color='red', linestyle='dashed', linewidth=2, label=f'True ({true_s_alpha})')
    axs[0, 0].set_title('Global Hyperparameter Recovery: Sigma Alpha')
    axs[0, 0].set_xlabel('Recovered Value')
    axs[0, 0].set_ylabel('Frequency (Iterations)')
    axs[0, 0].legend()
    
    axs[1, 0].hist(df['rec_sigma_beta'], bins=10, color='lightgreen', edgecolor='black')
    axs[1, 0].axvline(true_s_beta, color='red', linestyle='dashed', linewidth=2, label=f'True ({true_s_beta})')
    axs[1, 0].set_title('Global Hyperparameter Recovery: Sigma Beta')
    axs[1, 0].set_xlabel('Recovered Value')
    axs[1, 0].legend()
    
    # Calculate Latent Means
    alpha_mean = np.mean(e_alpha_particles, axis=1)
    beta_mean = np.mean(e_beta_particles, axis=1)
    
    # 2. Latent State Point Estimation (Mean Trajectories)
    axs[0, 1].plot(t_alpha, label='True Alpha Trajectory', color='black', alpha=0.7)
    axs[0, 1].plot(alpha_mean, label='Recovered Smoothed Alpha', color='red', alpha=0.8)
    axs[0, 1].set_title('Latent Trajectory Recovery: Learning Rate')
    axs[0, 1].set_xlabel('Trial Number')
    axs[0, 1].legend()
    
    axs[1, 1].plot(t_beta, label='True Beta Trajectory', color='black', alpha=0.7)
    axs[1, 1].plot(beta_mean, label='Recovered Smoothed Beta', color='blue', alpha=0.8)
    axs[1, 1].set_title('Latent Trajectory Recovery: Temperature')
    axs[1, 1].set_xlabel('Trial Number')
    axs[1, 1].legend()

    # Calculate Confidence Intervals (5th and 95th percentiles of the swarm)
    alpha_p5 = np.percentile(e_alpha_particles, 5, axis=1)
    alpha_p95 = np.percentile(e_alpha_particles, 95, axis=1)
    beta_p5 = np.percentile(e_beta_particles, 5, axis=1)
    beta_p95 = np.percentile(e_beta_particles, 95, axis=1)
    
    # 3. Particle Swarm Distributions (Confidence Intervals)
    axs[0, 2].plot(t_alpha, label='True Alpha Trajectory', color='black', alpha=0.7)
    axs[0, 2].plot(alpha_mean, label='Mean Estimate', color='red')
    axs[0, 2].fill_between(range(len(t_alpha)), alpha_p5, alpha_p95, color='red', alpha=0.2, label='90% CI Swarm Bounds')
    axs[0, 2].set_title('Particle Distribution Confidence: Alpha')
    axs[0, 2].set_xlabel('Trial Number')
    axs[0, 2].legend()

    axs[1, 2].plot(t_beta, label='True Beta Trajectory', color='black', alpha=0.7)
    axs[1, 2].plot(beta_mean, label='Mean Estimate', color='blue')
    axs[1, 2].fill_between(range(len(t_beta)), beta_p5, beta_p95, color='blue', alpha=0.2, label='90% CI Swarm Bounds')
    axs[1, 2].set_title('Particle Distribution Confidence: Beta')
    axs[1, 2].set_xlabel('Trial Number')
    axs[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig("full_recovery_plot_with_distributions.png", dpi=300)
    print("\nVisualizations saved to 'full_recovery_plot_with_distributions.png'")
    plt.show()

def run_em_recovery_study(num_iterations=20):
    results = []
    seeds = range(num_iterations)
    
    TRUE_SIGMA_ALPHA = 0.08
    TRUE_SIGMA_BETA = 0.05
    
    print(f"Starting AIND EM Parameter & Latent Recovery Study across {num_iterations} seeds...")
    
    # Variables to hold the final session's latent states for the trajectory plot
    sample_true_alpha, sample_alpha_particles = None, None
    sample_true_beta, sample_beta_particles = None, None
    
    for s in tqdm(seeds):
        choices, rewards, true_alpha, true_beta = generate_aind_coupled_data(
            TRUE_SIGMA_ALPHA, TRUE_SIGMA_BETA, seed=s
        )
        
        res = differential_evolution(
            func=calculate_nll_fast, 
            bounds=[(0.001, 0.2), (0.001, 0.2)],
            args=(choices, rewards),
            maxiter=30, popsize=10, workers=-1
        )
        recov_sigma_alpha, recov_sigma_beta = res.x
        
        # Note: execute_particle_smoother now returns the un-aggregated particle histories
        est_Q, alpha_particles_logit, beta_particles_logit = execute_particle_smoother(
            choices, rewards, sigma_alpha=recov_sigma_alpha, sigma_beta=recov_sigma_beta
        )
        
        # Transform the particles from logit space to natural bounds
        alpha_particles = np.clip(np.exp(alpha_particles_logit), 1e-4, 0.99)
        beta_particles = np.clip(np.exp(beta_particles_logit), 0.01, 100.0)
        
        est_alpha = np.mean(alpha_particles, axis=1)
        est_beta = np.mean(beta_particles, axis=1)
        
        mse_alpha = np.mean((true_alpha - est_alpha)**2)
        mse_beta = np.mean((true_beta - est_beta)**2)
        
        results.append({
            "seed": s,
            "rec_sigma_alpha": recov_sigma_alpha, 
            "err_sigma_alpha": abs(TRUE_SIGMA_ALPHA - recov_sigma_alpha),
            
            "rec_sigma_beta": recov_sigma_beta,   
            "err_sigma_beta": abs(TRUE_SIGMA_BETA - recov_sigma_beta),
            
            "mse_alpha_trajectory": mse_alpha,
            "mse_beta_trajectory": mse_beta
        })
        
        if s == seeds[-1]:
            sample_true_alpha, sample_alpha_particles = true_alpha, alpha_particles
            sample_true_beta, sample_beta_particles = true_beta, beta_particles
            
    df = pd.DataFrame(results)
    df.to_csv("em_full_recovery_study.csv", index=False)
    print("\nStudy complete! Results saved to 'em_full_recovery_study.csv'")
    
    print("\n--- Mean Errors ---")
    print(df[['err_sigma_alpha', 'err_sigma_beta', 'mse_alpha_trajectory', 'mse_beta_trajectory']].mean())

    plot_recovery_results(
        df, TRUE_SIGMA_ALPHA, TRUE_SIGMA_BETA, 
        sample_true_alpha, sample_alpha_particles, 
        sample_true_beta, sample_beta_particles
    )

if __name__ == "__main__":
    run_em_recovery_study(num_iterations=20)