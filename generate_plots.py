import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the figs directory exists
os.makedirs('papers/sibgrapi-2026/figs', exist_ok=True)

# 1. Architecture Comparison (Kappa and Accuracy)
archs = ['B0', 'B1', 'B2', 'B3', 'B7', 'ResNet-50']
kappa = [0.8321, 0.8390, 0.7788, 0.8284, 0.8034, 0.8150]
acc = [60.53, 63.38, 54.18, 65.39, 59.36, 61.00]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Architecture')
ax1.set_ylabel('Quadratic Weighted Kappa', color=color)
ax1.bar(archs, kappa, color=color, alpha=0.6, label='Kappa')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.5, 0.9)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(archs, acc, color=color, marker='o', label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(50, 70)

plt.title('Performance Comparison across Architectures')
fig.tight_layout()
plt.savefig('papers/sibgrapi-2026/figs/arch_comparison.png')
plt.close()

# 2. Entropy Impact
labels = ['Baseline (B0)', 'B0 + Entropy Filtering']
kappa_entropy = [0.8321, 0.8730]
acc_entropy = [60.53, 66.59]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, kappa_entropy, width, label='Kappa', color='skyblue')
rects2 = ax.bar(x + width/2, [a/100 for a in acc_entropy], width, label='Accuracy', color='salmon')

ax.set_ylabel('Score')
ax.set_title('Impact of Entropy-based Data Filtering')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0.5, 1.0)

plt.savefig('papers/sibgrapi-2026/figs/entropy_impact.png')
plt.close()

# 3. Confusion Matrix
# Based on b0-entropy-ordinal-test-results.txt
cm = np.array([
    [396,  32,   3,   2,   1,   0],
    [ 39, 318,  35,   7,   1,   0],
    [  4,  82,  86,  20,   6,   2],
    [  6,  16,  51,  68,  30,  14],
    [ 10,   9,  15,  29,  90,  34],
    [  7,   4,   5,  18,  44, 106]
])

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (B0 + Entropy Filtering)')
plt.colorbar()
tick_marks = np.arange(6)
plt.xticks(tick_marks, [f'ISUP {i}' for i in range(6)])
plt.yticks(tick_marks, [f'ISUP {i}' for i in range(6)])

fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('papers/sibgrapi-2026/figs/confusion_matrix.png')
plt.close()

print("Plots generated successfully.")
