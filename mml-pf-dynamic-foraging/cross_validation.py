import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import aind_dynamic_foraging_database as db

# ==========================================
# 1. Latent State Filtering (MML Particle Filter)
# ==========================================
def calculate_nll_window(sigma_alpha, sigma_beta, choices, rewards, num_particles=500):
    """
    Runs the particle filter over a complete session and returns the total NLL.
    """
    np.random.seed(42)  
    num_trials = len(choices)
    
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.uniform(0, 1, num_particles) 
    particles[:, 1] = np.random.uniform(0, 1, num_particles) 
    
    # Matched empirical priors for highly trained cohort
    particles[:, 2] = np.random.normal(np.log(0.3), 0.5, num_particles) 
    particles[:, 3] = np.random.normal(np.log(15.0), 0.5, num_particles)  
    
    session_nll = 0.0

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
        
        session_nll -= np.log(np.mean(p_choice) + 1e-16)
        
        weights = p_choice / (np.sum(p_choice) + 1e-16)
        if np.sum(weights) < 1e-20:
             weights = np.ones(num_particles) / num_particles
        weights /= np.sum(weights)

        idx = np.random.choice(num_particles, size=num_particles, p=weights)
        particles = particles[idx].copy()
        
        alpha_vals_resampled = np.clip(np.exp(particles[:, 2]), 1e-4, 0.99)
        particles[:, choices[t]] += alpha_vals_resampled * (rewards[t] - particles[:, choices[t]])
        
    return session_nll

# ==========================================
# 2. Global Parameter Optimization (Joint Likelihood)
# ==========================================
def objective_function(hyperparams, train_sessions):
    """
    Computes the Joint Negative Log-Likelihood across all historical training sessions.
    """
    sigma_alpha, sigma_beta = hyperparams
    joint_nll = 0.0
    
    for choices, rewards in train_sessions:
        joint_nll += calculate_nll_window(sigma_alpha, sigma_beta, choices, rewards)
        
    return joint_nll

# ==========================================
# 3. Chronological Walk-Forward Execution
# ==========================================
def execute_chronological_walk_forward(session_data_list):
    """
    Iteratively accumulates past sessions to train the global parameters, 
    and predicts the entirety of the next chronological session.
    """
    results = []
    cumulative_train_trials = 0
    train_sessions = []
    
    # Start testing at Session index 1, so we have at least Session 0 to train on
    for i in range(1, len(session_data_list)):
        
        # 1. Append the immediate past session to the training pool
        train_sessions.append(session_data_list[i-1])
        cumulative_train_trials += len(session_data_list[i-1][0])
        
        # 2. Define the upcoming session as the test target
        test_choices, test_rewards = session_data_list[i]
        
        print(f"  [Train: {len(train_sessions)} Sessions | {cumulative_train_trials} Trials] --> Predicting Session {i+1} ({len(test_choices)} trials)")
        
        # --- Phase 1: MML Parameter Optimization (Training) ---
        res = differential_evolution(
            func=objective_function,
            bounds=[(0.001, 0.2), (0.001, 0.2)],
            args=(train_sessions,),
            maxiter=30, popsize=10, workers=-1, disp=False
        )
        opt_sigma_alpha, opt_sigma_beta = res.x
        
        # --- Phase 2: Out-of-Sample Prediction (Testing) ---
        test_nll = calculate_nll_window(opt_sigma_alpha, opt_sigma_beta, test_choices, test_rewards)
        avg_nll_per_trial = test_nll / len(test_choices)
        
        results.append({
            "test_session_number": i + 1,
            "cumulative_train_trials": cumulative_train_trials,
            "opt_sigma_alpha": opt_sigma_alpha,
            "opt_sigma_beta": opt_sigma_beta,
            "out_of_sample_nll": test_nll,
            "test_trials": len(test_choices),
            "avg_nll_per_trial": avg_nll_per_trial
        })
        
    return pd.DataFrame(results)

# ==========================================
# 4. AIND Database Batch Execution
# ==========================================
if __name__ == "__main__":
    print("Querying AIND Database for highly trained cohort...")
    
    cohort_query = "task LIKE '%Uncoupled%' AND foraging_eff > 0.8 AND finished_trials > 300"
    all_sessions = db.select_sessions(where=cohort_query)
    
    # Sort chronologically to ensure the walk-forward respects the arrow of time
    all_sessions = all_sessions.sort_values(by=['subject_id', 'session_date'])
    
    # --- THE FIX: Filter for subjects with a long enough session history ---
    session_counts = all_sessions['subject_id'].value_counts()
    valid_subjects = session_counts[session_counts.between(4,10)].index.unique()
    
    if len(valid_subjects) == 0:
        print("Error: No subjects found with 4 or more valid sessions under these criteria.")
    else:
        # Grab the first two subjects that actually have enough data to walk forward
        subjects_to_test = valid_subjects[:2]
        test_sessions = all_sessions[all_sessions['subject_id'].isin(subjects_to_test)]
        
        trials_df = db.fetch_trials(test_sessions, columns=["animal_response", "earned_reward"])
        valid_trials_df = trials_df[trials_df['animal_response'] != 2].copy()
        
        all_cv_results = []
        
        for subject_id, subject_data in valid_trials_df.groupby('subject_id'):
            print(f"\n==========================================")
            print(f"Executing Chronological CV for Subject: {subject_id}")
            print(f"==========================================")
            
            session_data_list = []
            
            # Ensure chronological grouping
            for (session_date, session_id), session_data in subject_data.groupby(['session_date', 'session_id']):
                choices = session_data['animal_response'].astype(int).values
                rewards = session_data['earned_reward'].astype(int).values
                session_data_list.append((choices, rewards))
                
            print(f"Loaded {len(session_data_list)} consecutive sessions. Initializing MML Loop...\n")
            
            if len(session_data_list) > 1:
                subject_cv_df = execute_chronological_walk_forward(session_data_list)
                subject_cv_df.insert(0, 'subject_id', subject_id)
                all_cv_results.append(subject_cv_df)
            else:
                print("Not enough sessions to perform cross-session validation.")
            
        if all_cv_results:
            final_cv_df = pd.concat(all_cv_results, ignore_index=True)
            export_filename = "chronological_walk_forward_cv.csv"
            final_cv_df.to_csv(export_filename, index=False)
            print(f"\nCross-validation complete! Results saved to '{export_filename}'")