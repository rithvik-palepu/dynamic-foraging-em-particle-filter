import numpy as np
import matplotlib.pyplot as plt
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask
from aind_dynamic_foraging_models.generative_model import ForagerQLearning

def run_baseline_recovery():
    # ==========================================
    # 1. Instantiate the Ground Truth Agent
    # ==========================================
    print("1. Initializing ground-truth agent and synthetic task...")
    
    # Create a 1000-trial simulation environment
    task = CoupledBlockTask(reward_baiting=True, num_trials=1000, seed=42)
    
    # Initialize the true agent with the drift/forgetting parameter enabled[cite: 11]
    true_agent = ForagerQLearning(
        number_of_learning_rate=2,
        number_of_forget_rate=1,   # 1 = enables the 'forget_rate_unchosen' parameter[cite: 11]
        choice_kernel="none",
        action_selection="softmax"
    )
    
    # Set the ground-truth parameters we want to recover
    true_params = dict(
        learn_rate_rew=0.4,
        learn_rate_unrew=0.05,
        forget_rate_unchosen=0.15,  # Our target drift parameter
        softmax_inverse_temperature=5.0,
        biasL=0.0
    )
    true_agent.set_params(**true_params)

    # ==========================================
    # 2. Generate Synthetic Data
    # ==========================================
    print("2. Simulating behavior (Open-Loop)...")
    
    # Allow the agent to interact freely with the task[cite: 14]
    true_agent.perform(task)
    
    # Extract the resulting behavioral history[cite: 14]
    synthetic_choices = true_agent.get_choice_history()
    synthetic_rewards = true_agent.get_reward_history()

    # ==========================================
    # 3. Run the Recovery Pipeline
    # ==========================================
    print("3. Initializing recovery agent and starting DE/MLE fitting...")
    
    # Initialize a structurally identical, but blank, recovery agent
    recovery_agent = ForagerQLearning(
        number_of_learning_rate=2,
        number_of_forget_rate=1,
        choice_kernel="none",
        action_selection="softmax"
    )
    
    # Run the optimizer. This utilizes scipy's differential_evolution[cite: 14]
    # We pass DE_kwargs to utilize parallel workers for a faster search[cite: 14]
    fitting_result, _ = recovery_agent.fit(
        fit_choice_history=synthetic_choices,
        fit_reward_history=synthetic_rewards,
        DE_kwargs={'workers': -1, 'popsize': 15} 
    )
    
    # Print numerical parameter recovery results
    print("\n--- Parameter Recovery Results ---")
    print(f"{'Parameter':<30} | {'True Value':<10} | {'Recovered Value':<10}")
    print("-" * 57)
    for key, true_val in true_params.items():
        recovered_val = fitting_result.params.get(key, np.nan)
        print(f"{key:<30} | {true_val:<10.3f} | {recovered_val:<10.3f}")

    # ==========================================
    # 4. Extract and Compare Latent Variables
    # ==========================================
    print("\n4. Extracting latent variables and generating plot...")
    
    # True latents are already cached from the .perform() run[cite: 14]
    true_latents = true_agent.get_latent_variables()
    true_q_values = np.array(true_latents['q_value']) # shape: (n_actions, n_trials + 1)[cite: 11]
    
    # Recovered latents are cached because .fit() automatically runs 
    # .perform_closed_loop() using the best parameters before returning[cite: 14]
    recovered_latents = recovery_agent.get_latent_variables()
    recovered_q_values = np.array(recovered_latents['q_value'])

    # Visualize the tracked states
    plot_recovery(synthetic_choices, true_q_values, recovered_q_values)


def plot_recovery(choices, true_q, recovered_q):
    """Plots the true vs recovered Q-values side-by-side."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    trials = np.arange(len(choices) + 1)
    
    # Left Port Q-Values
    axes[0].plot(trials, true_q[0, :], label="True Q(Left)", color="blue", alpha=0.6, lw=3)
    axes[0].plot(trials, recovered_q[0, :], label="Recovered Q(Left)", color="black", ls="--", lw=2)
    axes[0].set_title("Latent State Recovery: Left Port Expected Value")
    axes[0].set_ylabel("Q-Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right Port Q-Values
    axes[1].plot(trials, true_q[1, :], label="True Q(Right)", color="red", alpha=0.6, lw=3)
    axes[1].plot(trials, recovered_q[1, :], label="Recovered Q(Right)", color="black", ls="--", lw=2)
    axes[1].set_title("Latent State Recovery: Right Port Expected Value")
    axes[1].set_xlabel("Trial Number")
    axes[1].set_ylabel("Q-Value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_baseline_recovery()