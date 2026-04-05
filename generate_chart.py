"""Generate benchmark chart from actual test results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma-4-31b-it"]
PROVIDERS = ["Groq", "Groq", "Google AI Studio"]
TASKS = ["Task 1\nPrecision", "Task 2\nInstability", "Task 3\nOptimization"]

# Real scores from actual runs
SCORES = {
    "llama-3.1-8b-instant":    [1.000, 0.667, 0.486],
    "llama-3.3-70b-versatile": [1.000, 0.625, 0.719],
    "gemma-4-31b-it":          [1.000, 0.625, 0.608],
}

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#0f0f1a')

x = range(len(TASKS))
width = 0.24
colors = ['#6366f1', '#f59e0b', '#10b981']

for i, model in enumerate(MODELS):
    scores = SCORES[model]
    avg = sum(scores) / len(scores)
    label = f"{model} ({PROVIDERS[i]}) — avg {avg:.3f}"
    bars = ax.bar([xi + i * width for xi in x], scores, width,
                  label=label, color=colors[i], edgecolor='white', linewidth=0.5,
                  zorder=3)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='white')

ax.set_ylabel('Average Score', fontsize=13, fontweight='bold', color='white')
ax.set_title('Libratio — Multi-Model Benchmark Comparison', fontsize=16,
             fontweight='bold', pad=15, color='white')
ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(TASKS, fontsize=11, ha='center', color='white')
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9, facecolor='#1a1a2e')
for text in ax.get_legend().get_texts():
    text.set_color('white')
ax.axhline(y=0.7, color='#ef4444', linestyle='--', alpha=0.4)
ax.grid(axis='y', alpha=0.15, color='white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#333')
ax.spines['left'].set_color('#333')
ax.tick_params(colors='white')

plt.tight_layout()
plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
print("Chart saved to benchmark_results.png")
