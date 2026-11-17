import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data = pd.read_csv("outputs/test_predictions.csv")
colors = {
    "Main sequence": "#4A90E2",
    "Giant": "#E5A044",
    "Supergiant": "#A66BE0",
    "White dwarf": "#55C6C0"
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for category, color in colors.items():
    subset = data[data["actual"] == category]
    axes[0].scatter(
        subset["ci"],
        subset["absmag"],
        c=color,
        label=category,
        s=8,
        alpha=0.6
    )

axes[0].set_title("Actual Star Classifications")
axes[0].set_xlabel("Colour Index (B-V)")
axes[0].set_ylabel("Absolute Magnitude")
axes[0].invert_yaxis()
axes[0].legend()
axes[0].grid(alpha=0.2)

for category, color in colors.items():
    subset = data[data["predicted"] == category]
    axes[1].scatter(
        subset["ci"],
        subset["absmag"],
        c=color,
        label=category,
        s=8,
        alpha=0.6
    )

axes[1].set_title("AI Predicted Classifications")
axes[1].set_xlabel("Colour Index (B-V)")
axes[1].set_ylabel("Absolute Magnitude")
axes[1].invert_yaxis()
axes[1].legend()
axes[1].grid(alpha=0.2)

fig.suptitle(
    "NEURON | Hertzsprung-Russell Diagram",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
Path("outputs").mkdir(exist_ok=True)
plt.savefig(
    "outputs/hr_diagram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
print("H-R diagram saved successfully!")
