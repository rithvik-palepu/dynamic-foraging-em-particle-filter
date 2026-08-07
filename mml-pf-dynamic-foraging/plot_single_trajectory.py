import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import aind_dynamic_foraging_database as db
from scipy.optimize import differential_evolution


# Import the actual methods present in your file
from empirical_drifting_agent_mml_pf import calculate_nll_fast, execute_particle_smoother

def generate_trajectory_plot(subject_id=651554):
    print(f"Fetching data for Subject {subject_id}...")
    
    # Isolate the exact 25-session data block used in the Walk-Forward CV
    subject_sessions = db.select_sessions(where=f"subject_id == {subject_id}").head(25)
    trials_df = db.fetch_trials(subject_sessions, columns=["animal_response", "earned_reward"])
    valid_trials = trials_df[trials_df['animal_response'] != 2].copy()
    
    choices = valid_trials['animal_response'].astype(int).values
    rewards = valid_trials['earned_reward'].astype(int).values
    
    print("Optimizing volatility hyper-parameters (MMLPF)...")
    # Optimization step matching your empirical script
    optimization_result = differential_evolution(
        func=calculate_nll_fast, 
        bounds=[(0.001, 0.2), (0.001, 0.2)], 
        args=(choices, rewards),
        maxiter=30, popsize=15, tol=0.01, 
        workers=-1, disp=False  
    )
    opt_sigma_alpha, opt_sigma_beta = optimization_result.x
    
    print(f"Running Particle Smoother with Sigma Alpha={opt_sigma_alpha:.4f}, Sigma Beta={opt_sigma_beta:.4f}...")
    # Trajectory extraction
    est_Q, est_alpha, est_beta = execute_particle_smoother(
        choices, rewards, sigma_alpha=opt_sigma_alpha, sigma_beta=opt_sigma_beta
    )
    
    print("Formatting presentation graphic...")
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot Latent Action Values (Q)
    ax.plot(est_Q[:, 0], color='#5b5cf0', linewidth=2, label='Q Left (Allen Blue)') 
    ax.plot(est_Q[:, 1], color='gray', linewidth=2, label='Q Right')
    
    # Plot animal choices as dots at the top and bottom of the graph
    right_choices = (choices == 1)
    left_choices = (choices == 0)
    ax.scatter(np.where(right_choices)[0], np.ones(sum(right_choices)) * 1.05, color='gray', s=10, marker='s', label='Chose Right')
    ax.scatter(np.where(left_choices)[0], np.ones(sum(left_choices)) * -0.05, color='#5b5cf0', s=10, label='Chose Left')

    ax.set_title(f'MMLPF Latent Trajectories (Subject {subject_id})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Latent Action Value (Q)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, 500) # Zooms in on just the first 500 trials
    
    plt.tight_layout()
    save_path = f"latent_trajectory_mmlpf_{subject_id}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Trajectory saved to '{save_path}'")

if __name__ == "__main__":
    generate_trajectory_plot()