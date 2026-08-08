import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import aind_dynamic_foraging_database as db
from scipy.optimize import differential_evolution


from empirical_drifting_agent_mml_pf import calculate_nll_fast, execute_particle_smoother

def generate_trajectory_plot(subject_id=651554):
    print(f"Fetching data for Subject {subject_id}...")
    
    subject_sessions = db.select_sessions(where=f"subject_id == {subject_id}").head(25)
    trials_df = db.fetch_trials(subject_sessions, columns=["animal_response", "earned_reward"])
    valid_trials = trials_df[trials_df['animal_response'] != 2].copy()
    
    choices = valid_trials['animal_response'].astype(int).values
    rewards = valid_trials['earned_reward'].astype(int).values
    
    print("Optimizing volatility hyper-parameters (MMLPF)...")
    optimization_result = differential_evolution(
        func=calculate_nll_fast, 
        bounds=[(0.001, 0.2), (0.001, 0.2)], 
        args=(choices, rewards),
        maxiter=30, popsize=15, tol=0.01, 
        workers=-1, disp=False  
    )
    opt_sigma_alpha, opt_sigma_beta = optimization_result.x
    
    print(f"Running Particle Smoother with Sigma Alpha={opt_sigma_alpha:.4f}, Sigma Beta={opt_sigma_beta:.4f}...")
    est_Q, est_alpha, est_beta = execute_particle_smoother(
        choices, rewards, sigma_alpha=opt_sigma_alpha, sigma_beta=opt_sigma_beta
    )
    
    print("Formatting presentation graphic...")
    # Create a 3-panel stacked plot sharing the exact same trial timeline
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # -----------------------------------------
    # Panel 1: Latent Action Values (Q)
    # -----------------------------------------
    ax1.plot(est_Q[:, 0], color='#5b5cf0', linewidth=2, label='Q Left (Allen Blue)') 
    ax1.plot(est_Q[:, 1], color='gray', linewidth=2, label='Q Right')
    
    right_choices = (choices == 1)
    left_choices = (choices == 0)
    ax1.scatter(np.where(right_choices)[0], np.ones(sum(right_choices)) * 1.05, color='gray', s=10, marker='s', label='Chose Right')
    ax1.scatter(np.where(left_choices)[0], np.ones(sum(left_choices)) * -0.05, color='#5b5cf0', s=10, label='Chose Left')

    ax1.set_title(f'MMLPF Full Latent State Tracking (Subject {subject_id})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Latent Value (Q)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # -----------------------------------------
    # Panel 2: Alpha (Learning Rate)
    # -----------------------------------------
    ax2.plot(est_alpha, color='#2ca02c', linewidth=2, label='Alpha (Learning Rate)')
    ax2.set_ylabel('Alpha', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # -----------------------------------------
    # Panel 3: Beta (Inverse Temperature)
    # -----------------------------------------
    ax3.plot(est_beta, color='#d62728', linewidth=2, label='Beta (Inverse Temp)')
    ax3.set_xlabel('Trial Number', fontsize=12)
    ax3.set_ylabel('Beta', fontsize=12)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Apply the static 500-trial zoom to all subplots
    ax1.set_xlim(0, 500)
    
    plt.tight_layout()
    save_path = f"latent_trajectory_mmlpf_full_{subject_id}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Full trajectory saved to '{save_path}'")

if __name__ == "__main__":
    generate_trajectory_plot()