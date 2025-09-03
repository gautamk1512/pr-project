import matplotlib.pyplot as plt

# --- Diagram 1: Single Perceptron ---
fig, ax = plt.subplots(figsize=(6, 6))

# Draw input nodes
ax.scatter(0, 3, s=200, c="skyblue")
ax.text(-0.2, 3, "x1", fontsize=12, ha="right")
ax.scatter(0, 2, s=200, c="skyblue")
ax.text(-0.2, 2, "x2", fontsize=12, ha="right")
ax.scatter(0, 1, s=200, c="skyblue")
ax.text(-0.2, 1, "xn", fontsize=12, ha="right")

# Draw perceptron (neuron)
circle = plt.Circle((2, 2), 0.5, color="lightgreen", ec="black")
ax.add_patch(circle)
ax.text(2, 2, "Σ\nf()", fontsize=12, ha="center", va="center")

# Draw output
ax.scatter(4, 2, s=200, c="salmon")
ax.text(4.2, 2, "Output", fontsize=12, ha="left")

# Arrows from inputs to perceptron
ax.arrow(0.1, 3, 1.7, -1, head_width=0.1, length_includes_head=True)
ax.arrow(0.1, 2, 1.7, 0, head_width=0.1, length_includes_head=True)
ax.arrow(0.1, 1, 1.7, 1, head_width=0.1, length_includes_head=True)

# Arrow from perceptron to output
ax.arrow(2.5, 2, 1.3, 0, head_width=0.1, length_includes_head=True)

ax.axis("off")
plt.title("Single Perceptron Model")
plt.savefig("perceptron_diagram.png")
plt.close()

# --- Diagram 2: MLP Architecture ---
fig, ax = plt.subplots(figsize=(7, 6))

# Input layer nodes
for i in range(3):
    ax.scatter(0, i, s=200, c="skyblue")
    ax.text(-0.2, i, f"x{i+1}", fontsize=12, ha="right")

# Hidden layer nodes
for i in range(4):
    ax.scatter(2, i-0.5, s=200, c="lightgreen")
    ax.text(2, i-0.5, f"h{i+1}", fontsize=10, ha="center", va="center")

# Output layer nodes
for i in range(2):
    ax.scatter(4, i, s=200, c="salmon")
    ax.text(4.3, i, f"y{i+1}", fontsize=12, ha="left")

# Connections Input -> Hidden
for i in range(3):
    for j in range(4):
        ax.plot([0, 2], [i, j-0.5], "k-", lw=0.5)

# Connections Hidden -> Output
for i in range(4):
    for j in range(2):
        ax.plot([2, 4], [i-0.5, j], "k-", lw=0.5)

ax.axis("off")
plt.title("Multi-Layer Perceptron (MLP) Architecture")
plt.savefig("mlp_diagram.png")
plt.close()

"perceptron_diagram.png", "mlp_diagram.png"
