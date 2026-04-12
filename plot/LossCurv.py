import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("training.csv")
df = pd.read_csv(csv_path)

tag = "train/loss"
scalar_df = df[df["Tag"] == tag]
plt.figure(figsize=(8, 4))
plt.plot(scalar_df["Step"], scalar_df["Value"], label=tag)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss with Step")
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png", dpi=300)
plt.show()