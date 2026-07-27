import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import aind_dynamic_foraging_database as db

# ==========================================
# 1. Fast E-Step: NLL Calculator 
# ==========================================
def calculate_nll_fast(hyperparams, choices, rewards, num_particles=500):
    np.random.seed(42)  # Temporarily converts objective function into a deterministic function for DE optimization
    sigma_alpha, sigma_beta = hyperparams
    num_trials = len(choices)
    
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.uniform(0, 1, num_particles) 
    particles[:, 1] = np.random.uniform(0, 1, num_particles) 
    particles[:, 2] = np.random.normal(np.log(0.05), 0.5, num_particles) 
    particles[:, 3] = np.random.normal(np.log(4.0), 0.5, num_particles)  
    
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

# ==========================================
# 2. Final E-Step: Fixed-Lag Particle Smoother
# ==========================================
def execute_particle_smoother(choices, rewards, sigma_alpha, sigma_beta, num_particles=1500, lag=15):
    num_trials = len(choices)
    
    particles_hist = np.zeros((num_trials, num_particles, 4))
    parent_indices = np.zeros((num_trials, num_particles), dtype=int)
    
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.uniform(0, 1, num_particles) 
    particles[:, 1] = np.random.uniform(0, 1, num_particles) 
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
        # Determine how far to look ahead (stops at the end of the session)
        target_t = min(t + lag, num_trials - 1)
        
        # Start with all 1500 particles that survived at target_t
        current_idx = np.arange(num_particles)
        
        # Trace their specific lineages backwards down to trial t
        for step in range(target_t - 1, t - 1, -1):
            current_idx = parent_indices[step][current_idx]
            
        # Extract the ancestral states at trial t
        smoothed_particles[t] = particles_hist[t][current_idx]
        
    est_Q = np.mean(smoothed_particles[:, :, :2], axis=1)
    est_alpha = np.mean(np.clip(np.exp(smoothed_particles[:, :, 2]), 1e-4, 0.99), axis=1)
    est_beta = np.mean(np.clip(np.exp(smoothed_particles[:, :, 3]), 0.01, 100.0), axis=1)
    
    return est_Q, est_alpha, est_beta

# ==========================================
# 3. Batch Execution & Export
# ==========================================
if __name__ == "__main__":
    print("Querying AIND Database for cohort...")
    
    # 1. Define your cohort
    cohort_query = "task LIKE '%Uncoupled%' AND foraging_eff > 0.8 AND finished_trials > 300"
    all_sessions = db.select_sessions(where=cohort_query)
    
    # Restrict to 3 sessions for a fast initial test run. Remove this to run the full cohort.
    test_sessions = all_sessions.head(3)
    
    print(f"Found {len(all_sessions)} total sessions matching criteria. Processing {len(test_sessions)} sessions...\n")
    
    # 2. Fetch all trials for the cohort in one fast S3 read
    trials_df = db.fetch_trials(test_sessions, columns=["animal_response", "earned_reward"])
    
    # Filter out ignored trials globally
    valid_trials_df = trials_df[trials_df['animal_response'] != 2].copy()
    
    processed_session_dfs = []
    
    # 3. Iterate through each unique session in the dataframe
    for (subject_id, session_date, session_id), session_data in valid_trials_df.groupby(['subject_id', 'session_date', 'session_id']):
        print(f"Processing Subject: {subject_id} | Date: {session_date} | Trials: {len(session_data)}")
        
        choices = session_data['animal_response'].astype(int).values
        rewards = session_data['earned_reward'].astype(int).values
        
        # --- M-Step: Optimize Volatility ---
        m_step_result = differential_evolution(
            func=calculate_nll_fast, 
            bounds=[(0.001, 0.2), (0.001, 0.2)], 
            args=(choices, rewards),
            maxiter=30, popsize=15, tol=0.01, 
            workers=-1, disp=False  
        )
        opt_sigma_alpha, opt_sigma_beta = m_step_result.x
        
        # --- E-Step: Extract Smoothed Trajectories ---
        # Note: 'lag' defaults to 15, but you can pass it explicitly here if desired
        est_Q, est_alpha, est_beta = execute_particle_smoother(
            choices, rewards, sigma_alpha=opt_sigma_alpha, sigma_beta=opt_sigma_beta
        )
        
        # 4. Attach the latent variables directly to the session dataframe
        session_data = session_data.copy()
        session_data['est_Q_L'] = est_Q[:, 0]
        session_data['est_Q_R'] = est_Q[:, 1]
        session_data['est_alpha'] = est_alpha
        session_data['est_beta'] = est_beta
        session_data['opt_sigma_alpha'] = opt_sigma_alpha
        session_data['opt_sigma_beta'] = opt_sigma_beta
        
        processed_session_dfs.append(session_data)
        print(f"  -> Done. Volatilities: Sigma Alpha={opt_sigma_alpha:.4f}, Sigma Beta={opt_sigma_beta:.4f}\n")
        
    # 5. Concatenate and Export
    if processed_session_dfs:
        final_batch_df = pd.concat(processed_session_dfs, ignore_index=True)
        export_filename = "batch_em_trajectories.csv"
        final_batch_df.to_csv(export_filename, index=False)
        print(f"Batch processing complete! Dataframe saved to {export_filename}")
        print(final_batch_df.head())
    else:
        print("No valid data processed.")