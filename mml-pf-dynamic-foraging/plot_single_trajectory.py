import matplotlib
import matplotlib.pyplot as plt
import aind_dynamic_foraging_database as db

# Force headless plotting for the HPC
matplotlib.use('Agg')

# Import your existing methods to avoid redundant rewrites!
# (Verify these function names match exactly what is inside your empirical script)
from empirical_drifting_agent_mml_pf import extract_latent_states, plot_trajectory

def generate_trajectory_plot(subject_id=651554):
    print(f"Fetching data for Subject {subject_id}...")
    
    # 1. Isolate the exact 25-session data block used in the Walk-Forward CV
    subject_sessions = db.select_sessions(where=f"subject_id == {subject_id}").head(25)
    trials_df = db.fetch_trials(subject_sessions, columns=["animal_response", "earned_reward"])
    valid_trials = trials_df[trials_df['animal_response'] != 2].copy()
    
    choices = valid_trials['animal_response'].astype(int).values
    rewards = valid_trials['earned_reward'].astype(int).values
    
    # 2. Process through your existing MMLPF empirical script
    print("Running MMLPF to extract continuous latent trajectories...")
    latent_states = extract_latent_states(choices, rewards) 
    
    # 3. Generate and save the plot using your existing rendering function
    print("Formatting presentation graphic...")
    fig = plot_trajectory(latent_states, choices, rewards)
    
    # Apply Allen Institute corporate periwinkle to the latent trace if your function allows axis modifications
    # plt.gca().get_lines()[0].set_color('#5b5cf0') 
    
    save_path = f"latent_trajectory_mmlpf_{subject_id}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Trajectory saved to '{save_path}'")

if __name__ == "__main__":
    generate_trajectory_plot()