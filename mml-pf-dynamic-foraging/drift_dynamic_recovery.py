import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from tqdm import tqdm

# ==========================================
# 1. Generative Environment & Agent
# ==========================================
def generate_environment_and_data(num_trials=1000):
    """Generates synthetic choices and rewards mimicking Samejima et al. (2003)."""
    np.random.seed(42)
    
    true_alpha = 0.05
    true_beta = 1.0

    blocks = [
        (200, [0.5, 0.5], 'white'),
        (100, [0.1, 0.9], 'darkgray'),
        (100, [0.5, 0.5], 'white'),
        (100, [0.9, 0.1], 'lightgray'),
        (100, [0.1, 0.9], 'darkgray'),
        (100, [0.5, 0.5], 'white'),
        (100, [0.1, 0.9], 'darkgray'),
        (100, [0.9, 0.1], 'lightgray'),
        (100, [0.5, 0.5], 'white')
    ]

    P_R = np.zeros((num_trials, 2))
    block_colors = []
    idx = 0
    for length, probs, color in blocks:
        P_R[idx:idx+length, :] = probs
        block_colors.append((idx, idx+length, color, probs))
        idx += length

    choices = np.zeros(num_trials, dtype=int)
    rewards = np.zeros(num_trials)
    true_Q = np.zeros((num_trials, 2))
    Q = np.zeros(2)

    for t in range(num_trials):
        true_Q[t] = Q.copy()
        exp_Q = np.exp(true_beta * Q)
        probs = exp_Q / np.sum(exp_Q)
        action = np.random.choice([0, 1], p=probs)
        reward = 5.0 if np.random.rand() < P_R[t, action] else 0.0
        
        Q[action] += true_alpha * (reward - Q[action])
        choices[t] = action
        rewards[t] = reward

    return choices, rewards, true_Q, block_colors, true_alpha, true_beta


# ==========================================
# 2. 4D Particle Filter Logic
# ==========================================
def execute_particle_filter(choices, rewards, sigma_alpha=0.05, sigma_beta=0.02, num_particles=1500):
    """Runs a 4D particle filter with a Genealogical Backward Smoother."""
    num_trials = len(choices)
    
    # --- 1. Pre-allocate History Tracking Arrays ---
    particles_hist = np.zeros((num_trials, num_particles, 4))
    parent_indices = np.zeros((num_trials, num_particles), dtype=int)
    
    # Initialize particles: Mean {0,0,0,0}, Variance {1, 1, 3, 1}
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.normal(0, 1, num_particles)
    particles[:, 1] = np.random.normal(0, 1, num_particles)
    particles[:, 2] = np.random.normal(0, np.sqrt(3), num_particles) 
    particles[:, 3] = np.random.normal(0, 1, num_particles)

    # ==========================================
    # FORWARD FILTER PASS
    # ==========================================
    for t in range(num_trials):
        # A. Prediction: Drift meta-parameters
        particles[:, 2] += np.random.normal(0, sigma_alpha, num_particles)
        particles[:, 3] += np.random.normal(0, sigma_beta, num_particles)

        alpha_vals = np.exp(particles[:, 2])
        beta_vals = np.exp(particles[:, 3])
        Q_vals = particles[:, :2]

        # B. Observation: Likelihood 
        max_Q = np.max(Q_vals, axis=1, keepdims=True)
        exp_Q = np.exp(beta_vals[:, None] * (Q_vals - max_Q))
        probs = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
        
        p_choice = probs[np.arange(num_particles), choices[t]]
        weights = p_choice / (np.sum(p_choice) + 1e-16)

        # C. Save State BEFORE Resampling
        particles_hist[t] = particles.copy()

        # D. Resample & Record Genealogy
        idx = np.random.choice(num_particles, size=num_particles, p=weights)
        parent_indices[t] = idx
        particles = particles[idx].copy()
        
        # E. Action: Deterministic Q-update
        alpha_vals_resampled = np.exp(particles[:, 2])
        particles[:, choices[t]] += alpha_vals_resampled * (rewards[t] - particles[:, choices[t]])
        
    # ==========================================
    # BACKWARD SMOOTHING PASS (Ancestor Tracing)
    # ==========================================
    smoothed_particles = np.zeros((num_trials, num_particles, 4))
    
    # Start with the indices of the particles that survived at the very end (T)
    current_idx = np.arange(num_particles)
    
    # Trace backwards from T-1 down to 0
    for t in range(num_trials - 1, -1, -1):
        # Find who the parents were at this specific time step
        current_idx = parent_indices[t][current_idx]
        
        # Extract the exact pre-resampled states of those specific parents
        smoothed_particles[t] = particles_hist[t][current_idx]
        
    # Recalculate the expected values using only the smoothed lineage
    est_Q = np.mean(smoothed_particles[:, :, :2], axis=1)
    est_alpha = np.mean(np.exp(smoothed_particles[:, :, 2]), axis=1)
    est_beta = np.mean(np.exp(smoothed_particles[:, :, 3]), axis=1)
    
    return est_Q, est_alpha, est_beta


# ==========================================
# 3. Grid Sweep Optimization
# ==========================================
def run_meta_parameter_sweep():
    """Sweeps over drift variances to find the optimal tracking configuration."""
    print("Starting Grid Sweep for Meta-Meta Parameters...")
    choices, rewards, _, _, true_alpha, true_beta = generate_environment_and_data()
    
    sigma_alpha_range = [0.01, 0.05, 0.1]
    sigma_beta_range = [0.001, 0.005, 0.02, 0.05]
    results = []
    
    for s_alpha, s_beta in tqdm(list(itertools.product(sigma_alpha_range, sigma_beta_range))):
        # Use 500 particles for faster grid sweeping
        _, est_alpha, est_beta = execute_particle_filter(
            choices, rewards, s_alpha, s_beta, num_particles=500
        )
        
        # Calculate MAE (ignoring first 100 trials of burn-in)
        burn_in = 100
        mae_a = np.mean(np.abs(est_alpha[burn_in:] - true_alpha))
        mae_b = np.mean(np.abs(est_beta[burn_in:] - true_beta))
        
        # Combined error metric (weighted to balance scale differences)
        combined_error = mae_a + (mae_b * 0.1) 
        
        results.append({
            'sigma_alpha': s_alpha,
            'sigma_beta': s_beta,
            'MAE_alpha': mae_a,
            'MAE_beta': mae_b,
            'Combined_Error': combined_error
        })
        
    df = pd.DataFrame(results).sort_values(by='Combined_Error')
    print("\n--- Top 3 Parameter Configurations ---")
    print(df.head(3).to_string(index=False))


# ==========================================
# 4. Main Execution & Visualization
# ==========================================
def visualize_best_tracking(sigma_alpha=0.05, sigma_beta=0.005):
    """Runs the filter with the optimized parameters and plots the results."""
    print(f"\nRunning final high-resolution tracking (s_alpha={sigma_alpha}, s_beta={sigma_beta})...")
    
    # 1. Fetch Data
    choices, rewards, true_Q, block_colors, true_alpha, true_beta = generate_environment_and_data()
    num_trials = len(choices)
    
    # 2. Run Filter
    est_Q, est_alpha, est_beta = execute_particle_filter(
        choices, rewards, sigma_alpha, sigma_beta, num_particles=1500
    )
    
    # 3. Plotting
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.5, 1.5, 1.5, 1.5]})

    def add_blocks(ax):
        for start, end, color, prob in block_colors:
            if color != 'white':
                ax.axvspan(start, end, color=color, alpha=0.5, lw=0)
                if prob == [0.1, 0.9]:
                    ax.text(start + 50, ax.get_ylim()[1], '[0.1, 0.9]', ha='center', va='bottom', fontsize=8)
                elif prob == [0.9, 0.1]:
                    ax.text(start + 50, ax.get_ylim()[1], '[0.9, 0.1]', ha='center', va='bottom', fontsize=8)

    # (a) Choices and Rewards
    axes[0].set_ylim(-0.5, 1.5)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Left', 'Right'])
    add_blocks(axes[0])
    
    unrewarded = rewards == 0
    rewarded = rewards > 0
    axes[0].scatter(np.where(unrewarded)[0], choices[unrewarded], color='black', s=10, marker='.')
    axes[0].scatter(np.where(rewarded)[0], choices[rewarded], edgecolors='black', facecolors='none', s=20, marker='o')

    # (b) Q_L
    axes[1].plot(true_Q[:, 0], ':', color='gray', label='True value')
    axes[1].plot(est_Q[:, 0], '-', color='black', label='Estimated')
    axes[1].set_ylabel('$Q_L$')
    axes[1].set_ylim(0, 5)
    add_blocks(axes[1])

    # (c) Q_R
    axes[2].plot(true_Q[:, 1], ':', color='gray')
    axes[2].plot(est_Q[:, 1], '-', color='black')
    axes[2].set_ylabel('$Q_R$')
    axes[2].set_ylim(0, 5)
    add_blocks(axes[2])

    # (d) Alpha
    axes[3].plot(np.full(num_trials, true_alpha), ':', color='gray', label='True value')
    axes[3].plot(est_alpha, '-', color='black', label='Estimated')
    axes[3].set_ylabel(r'$\alpha$')
    axes[3].set_yscale('log')
    axes[3].set_ylim(1e-2, 1e0)
    axes[3].legend(loc='upper right', frameon=False)

    # (e) Beta
    axes[4].plot(np.full(num_trials, true_beta), ':', color='gray')
    axes[4].plot(est_beta, '-', color='black')
    axes[4].set_ylabel(r'$\beta$')
    axes[4].set_yscale('log')
    axes[4].set_ylim(0.2, 2)
    axes[4].set_xlabel('Trials')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Choose whether you want to run the sweep or just visualize the best result
    
    # 1. Run the parameter optimization
    # run_meta_parameter_sweep()
    
    # 2. Visualize using the best found parameters
    visualize_best_tracking(sigma_alpha=0.05, sigma_beta=0.020)