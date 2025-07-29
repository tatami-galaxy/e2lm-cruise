import numpy as np
import matplotlib.pyplot as plt
import math
import json

def var(a):
    return sum((a - np.mean(a))**2) / len(a)

def normalize(a, min_a, max_a):
    return (a - min_a) / (max_a - min_a + 1e-20)

def autocorrelation(signal, lag=1):
    """Compute the autocorrelation at given lag."""
    n = len(signal)
    s1 = signal[lag:]
    s2 = signal[:n-lag]
    return np.corrcoef(s1, s2)[0, 1]

def spearmancorrelation(a):
    x, y = np.array(range(len(a))), np.array(a)
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    n = len(x)
    d_squared_sum = np.sum((x_rank - y_rank)**2)
    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return max(rho, 0)

def signal_quality_score(a):
    a_new = a
    sc = spearmancorrelation(a_new)
    ac = np.mean([np.abs(autocorrelation(a_new, i)) for i in range(1,len(a_new)//4)])
    return sc, ac, (sc + ac)/2

def convert_iter_to_gt(a):
    global_batch_size = 1024
    batch_size_rampup = [256, 256, 4882812]
    seq_len = 4096

    num_phases = global_batch_size / batch_size_rampup[1]
    num_samples_per_phase = batch_size_rampup[2] / (num_phases - 1)

    batch_size_per_phase = np.arange(batch_size_rampup[0], global_batch_size+1, batch_size_rampup[1])
    num_samples_per_phase_vec = np.ones(int(num_phases)) * num_samples_per_phase
    num_iter_per_phase = num_samples_per_phase_vec / batch_size_per_phase
    num_iter_per_phase[-1] = np.max(a) - np.sum(num_iter_per_phase[:-1])

    a_gt = []
    num_iter_per_phase_cumsum = np.cumsum(num_iter_per_phase)

    for i in range(len(a)):
        iter_idx = np.searchsorted(num_iter_per_phase_cumsum, a[i])
        if iter_idx > 0:
            a_gt.append(np.sum([num_iter_per_phase[j] * batch_size_per_phase[j] * seq_len for j in range(iter_idx)] +\
                        [(a[i] - num_iter_per_phase_cumsum[iter_idx-1]) * batch_size_per_phase[iter_idx] * seq_len])/1e9)
        else:
            a_gt.append((a[i] * batch_size_per_phase[iter_idx] * seq_len)/1e9)
    return a_gt

# Load data
path = 'consolidated_results.json'
with open(path, 'r') as file:
    data = json.load(file)['data']

# Define benchmark lists
mmlu_hs_list = [
    "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics"
]

mmlu_adv_list = [
    "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_physics"
]

# Initialize results storage
results_signal_quality = {}
models = ['dense-500m-arch1']

# Create figure
num_pairs = len(mmlu_hs_list)
cols = 2
rows = math.ceil(num_pairs / cols)
plt.figure(figsize=(10*cols, 6*rows))
plt.subplots_adjust(hspace=0.6, wspace=0.3, top=0.95)

for model in models:
    results_signal_quality[model] = {}
    for hs_bench, adv_bench in zip(mmlu_hs_list, mmlu_adv_list):
        # Process high school benchmark
        hs_iters = data[model]['iteration']
        hs_scores = data[model][hs_bench]
        hs_pairs = [(hs_iters[idx], hs_scores[idx]) for idx in range(min(len(hs_iters), len(hs_scores)))]
        hs_iters_gt = convert_iter_to_gt([x[0] for x in hs_pairs])
        hs_merged = list(zip(hs_iters_gt, [x[0] for x in hs_pairs], [x[1] for x in hs_pairs]))
        hs_merged = list(filter(lambda x: x[0] <= 206, hs_merged))
        hs_rho, hs_ac, hs_score = signal_quality_score(np.array([x[2] for x in hs_merged]))
        
        # Process advanced benchmark
        adv_iters = data[model]['iteration']
        adv_scores = data[model][adv_bench]
        adv_pairs = [(adv_iters[idx], adv_scores[idx]) for idx in range(min(len(adv_iters),len(adv_scores)))]
        adv_iters_gt = convert_iter_to_gt([x[0] for x in adv_pairs])
        adv_merged = list(zip(adv_iters_gt, [x[0] for x in adv_pairs], [x[1] for x in adv_pairs]))
        adv_merged = list(filter(lambda x: x[0] <= 206, adv_merged))
        adv_rho, adv_ac, adv_score = signal_quality_score(np.array([x[2] for x in adv_merged]))
        
        # Store results
        results_signal_quality[model][hs_bench] = (hs_rho, hs_ac, hs_score)
        results_signal_quality[model][adv_bench] = (adv_rho, adv_ac, adv_score)
        
        # Create subplot
        ax = plt.subplot(rows, cols, mmlu_hs_list.index(hs_bench) + 1)
        
        # Plot both curves
        ax.plot([x[0] for x in hs_merged], [x[2] for x in hs_merged], 
                linewidth=1.5, label=f'HS: {hs_bench.split("_")[-1]}', color='blue')
        ax.plot([x[0] for x in adv_merged], [x[2] for x in adv_merged], 
                linewidth=1.5, label=f'College: {adv_bench.split("_")[-1]}', color='orange')
        
        # Add title and labels
        title = f"{hs_bench.split('_')[-1]} vs {adv_bench.split('_')[-1]}\n"
        title += f"HS Score: {hs_score:.3f} | College Score: {adv_score:.3f}"
        ax.set_title(title, pad=10)
        ax.set_xlabel("steps")
        ax.set_ylabel("Benchmark score")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Visual improvements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("combined_benchmark_plot.png", dpi=300, bbox_inches='tight')
plt.show()