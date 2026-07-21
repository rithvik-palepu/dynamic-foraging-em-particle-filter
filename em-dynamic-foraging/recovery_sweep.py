import numpy as np
import pandas as pd
from tqdm import tqdm # Install with: pip install tqdm

# Import your existing functions from em_dynamic_recovery_3.py
# (Ensure your functions are in the same directory or defined here)
from em_dynamic_recovery import (
    generate_stochastic_q_data, 
    run_av_particle_filter_fast
)
from scipy.optimize import differential_evolution

def run_recovery_study(num_iterations=20):
    results = []
    
    # Define a range of seeds to ensure varied synthetic datasets
    seeds = range(num_iterations)
    
    print(f"Starting Parameter Recovery Study across {num_iterations} seeds...")
    
    for s in tqdm(seeds):
        # 1. Generate data with specific seed
        # Note: You may need to update generate_stochastic_q_data to accept 'seed'
        choices, rewards, _, true_sigma, true_beta, true_alpha = generate_stochastic_q_data(seed=s)
        
        # 2. Run Optimization
        # We use a slightly faster iter/popsize for the sweep to save time
        res = differential_evolution(
            func=run_av_particle_filter_fast, 
            bounds=[(0.001, 0.5), (1.0, 15.0), (0.001, 0.999)],
            args=(choices, rewards),
            maxiter=30, popsize=10, workers=-1
        )
        
        rec_sigma, rec_beta, rec_alpha = res.x
        
        # 3. Store Results
        results.append({
            "seed": s,
            "true_sigma": true_sigma, "rec_sigma": rec_sigma, "err_sigma": abs(true_sigma - rec_sigma),
            "true_beta": true_beta,   "rec_beta": rec_beta,   "err_beta": abs(true_beta - rec_beta),
            "true_alpha": true_alpha, "rec_alpha": rec_alpha, "err_alpha": abs(true_alpha - rec_alpha)
        })
        
    # 4. Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("parameter_recovery_study.csv", index=False)
    print("\nStudy complete! Results saved to 'parameter_recovery_study.csv'")
    
    # Print summary for quick verification
    print("\n--- Mean Absolute Errors ---")
    print(df[['err_sigma', 'err_beta', 'err_alpha']].mean())

if __name__ == "__main__":
    run_recovery_study()