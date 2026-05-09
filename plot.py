import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_all_plots(csv_dir="results", out_dir="plots"):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    os.makedirs(out_dir, exist_ok=True)
    datasets = ['digits', 'breast-w']

    # 1. Ablation Study
    df_ablation = pd.read_csv(f"{csv_dir}/ablation.csv")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_ablation, x="dataset", y="mean_test_acc", hue="variant", capsize=0.1)
    plt.ylim(df_ablation['mean_test_acc'].min() - 2, df_ablation['mean_test_acc'].max() + 1)
    plt.title("Ablation Study: Test Accuracy by Variant")
    plt.ylabel("Mean Test Accuracy (%)")
    plt.legend(title="Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/1_ablation.png")
    plt.close()

    # 2. Accuracy Comparison
    df_acc = pd.read_csv(f"{csv_dir}/accuracy_results.csv")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_acc, x="dataset", y="mean_acc", hue="method", capsize=0.1)
    plt.ylim(df_acc['mean_acc'].min() - 2, df_acc['mean_acc'].max() + 1)
    plt.title("Accuracy Comparison: MEPSI vs Baselines")
    plt.ylabel("Mean Test Accuracy (%)")
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/2_accuracy.png")
    plt.close()

    # 3. Generalization Bounds
    df_bound = pd.read_csv(f"{csv_dir}/generalization_bound.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, ds in enumerate(datasets):
        subset = df_bound[df_bound['dataset'] == ds]
        ax = axes[i]
        ax.plot(subset['k'], subset['empirical_error'], marker='o', label='Empirical Error')
        ax.plot(subset['k'], subset['test_error'], marker='s', label='Test Error')
        ax.plot(subset['k'], subset['bound'], marker='^', linestyle='--', label='Gen. Bound')
        ax.set_title(f"Generalization Bound vs $k$ ({ds})")
        ax.set_xlabel("Ensemble Size $k$")
        ax.set_ylabel("Error")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/3_gen_bound.png")
    plt.close()

    # 4. Greedy Optimization Convergence
    df_conv = pd.read_csv(f"{csv_dir}/greedy_convergence.csv")
    for ds in df_conv['dataset'].unique():
        subset = df_conv[df_conv['dataset'] == ds]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = 'tab:red'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective Value', color=color)
        ax1.plot(subset['iteration'], subset['objective'], color=color, marker='o', label='Objective')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Empirical Error', color=color)  
        ax2.plot(subset['iteration'], subset['empirical_error'], color=color, marker='s', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.suptitle(f"Greedy Optimization Convergence ({ds})")
        fig.tight_layout()
        plt.savefig(f"{out_dir}/4_greedy_conv_{ds}.png")
        plt.close()

    # 5. Lambda Sensitivity
    df_lam = pd.read_csv(f"{csv_dir}/lambda_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, ds in enumerate(datasets):
        subset = df_lam[df_lam['dataset'] == ds]
        ax1 = axes[i]
        x = subset['lambda'].replace(0, 1e-5) # Offset for log scale
        
        color = 'tab:blue'
        ax1.set_xlabel(r'$\lambda$ (Log Scale)')
        ax1.set_ylabel('Mean Test Acc (%)', color=color)
        ax1.plot(x, subset['mean_acc'], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xscale('log')
        
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Mean Structural Info', color=color)
        ax2.plot(x, subset['mean_structural_info'], color=color, marker='s', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.set_title(f"Sensitivity to $\lambda$ ({ds})")
    fig.tight_layout()
    plt.savefig(f"{out_dir}/5_lambda_sens.png")
    plt.close()

    # 6. Pruning Size Sensitivity
    df_size = pd.read_csv(f"{csv_dir}/pruning_size_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, ds in enumerate(datasets):
        subset = df_size[df_size['dataset'] == ds]
        ax = axes[i]
        ax.plot(subset['k'], subset['mepsi_mean_acc'], marker='o', label='MEPSI')
        ax.fill_between(subset['k'], subset['mepsi_mean_acc'] - subset['mepsi_std_acc'], 
                        subset['mepsi_mean_acc'] + subset['mepsi_std_acc'], alpha=0.2)
        
        ax.plot(subset['k'], subset['random_mean_acc'], marker='s', label='Random')
        ax.fill_between(subset['k'], subset['random_mean_acc'] - subset['random_std_acc'], 
                        subset['random_mean_acc'] + subset['random_std_acc'], alpha=0.2)
        
        ax.set_title(f"Accuracy vs Pruning Size $k$ ({ds})")
        ax.set_xlabel("Ensemble Size $k$")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/6_pruning_size.png")
    plt.close()

    # 7. Runtime vs Total Trees (T)
    df_time = pd.read_csv(f"{csv_dir}/runtime_vs_T.csv")
    plt.figure(figsize=(7, 5))
    plt.plot(df_time['T'], df_time['t_total_s'], marker='o', label='Total Time')
    plt.plot(df_time['T'], df_time['t_prune_s'], marker='s', linestyle='--', label='Pruning Time')
    plt.title("Runtime vs Total Number of Trees ($T$)")
    plt.xlabel("Total Trees $T$")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/7_runtime.png")
    plt.close()

    # 8. Structural Stats (Selected vs Unselected)
    df_struct = pd.read_csv(f"{csv_dir}/structural_stats.csv")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_struct, x="dataset", y="esm", hue="selected")
    plt.title("Structural Complexity (ESM): Selected vs Unselected Trees")
    plt.ylabel("Empirical Structural Measure (ESM)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/8_struct_stats.png")
    plt.close()

    # 9. TED Distribution
    df_ted = pd.read_csv(f"{csv_dir}/ted_distribution.csv")
    melted = df_ted.melt(id_vars='dataset', 
                         value_vars=['mean_ted_full', 'mean_ted_selected'],
                         var_name='Group', value_name='Mean TED')
    melted['Group'] = melted['Group'].replace({'mean_ted_full': 'Full Ensemble', 'mean_ted_selected': 'Selected Subset'})
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted, x="dataset", y="Mean TED", hue="Group")
    plt.title("Tree Edit Distance (TED) Distribution")
    plt.ylabel("Mean TED")
    plt.legend(title="Ensemble")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/9_ted.png")
    plt.close()

if __name__ == "__main__":
    generate_all_plots()