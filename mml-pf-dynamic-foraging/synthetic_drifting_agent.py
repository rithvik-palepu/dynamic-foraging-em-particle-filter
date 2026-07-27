import numpy as np
import matplotlib.pyplot as plt

def generate_drifting_agent_data(num_trials=500, seed=42):
    """
    Simulates a 2-armed bandit task where the agent's learning rate (alpha) 
    and inverse temperature (beta) undergo a Gaussian random walk.
    """
    np.random.seed(seed)
    
    # Meta-parameters governing the random walk (Volatility)
    sigma_alpha = 0.05
    sigma_beta = 0.5
    
    # Initialize latent states
    alpha = 0.5
    beta = 5.0
    Q = np.array([0.5, 0.5])
    
    # Environment: Reward probabilities switch every 100 trials
    reward_probs = np.array([0.8, 0.2])
    
    # Tracking arrays
    choices, rewards = [], []
    true_alpha, true_beta, true_Q = [], [], []
    
    for t in range(num_trials):
        # Trigger block switch
        if t > 0 and t % 100 == 0:
            reward_probs = reward_probs[::-1]
            
        # 1. Random walk for latents (with biological bounds applied)
        alpha = np.clip(alpha + np.random.normal(0, sigma_alpha), 0.01, 0.99)
        beta = np.clip(beta + np.random.normal(0, sigma_beta), 0.1, 20.0)
        
        true_alpha.append(alpha)
        true_beta.append(beta)
        true_Q.append(Q.copy())
        
        # 2. Agent makes a decision
        exp_Q = np.exp(beta * (Q - np.max(Q)))
        probs = exp_Q / np.sum(exp_Q)
        choice = np.random.choice([0, 1], p=probs)
        
        # 3. Environment delivers reward
        reward = np.random.binomial(1, reward_probs[choice])
        
        # 4. Agent updates internal model (Q-learning)
        Q[choice] += alpha * (reward - Q[choice])
        
        choices.append(choice)
        rewards.append(reward)
        
    return {
        'choices': np.array(choices),
        'rewards': np.array(rewards),
        'true_alpha': np.array(true_alpha),
        'true_beta': np.array(true_beta),
        'true_Q': np.array(true_Q)
    }

if __name__ == "__main__":
    # Quick visual verification of the synthetic biology
    data = generate_drifting_agent_data(num_trials=500, seed=10)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(data['true_Q'][:, 0], color='blue', label='Q Left')
    axes[0].plot(data['true_Q'][:, 1], color='red', label='Q Right')
    axes[0].set_ylabel("True Q-Values")
    axes[0].legend(loc='upper right')
    axes[0].set_title("Synthetic Ground Truth: Drifting Latent States")
    
    axes[1].plot(data['true_beta'], color='black')
    axes[1].set_ylabel(r"True Inverse Temp ($\beta$)")
    
    axes[2].plot(data['true_alpha'], color='black')
    axes[2].set_ylabel(r"True Learning Rate ($\alpha$)")
    axes[2].set_xlabel("Trials")
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.tight_layout()
    plt.show()