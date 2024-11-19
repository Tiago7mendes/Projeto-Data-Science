import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

print(df.info())  # So de curiosidade

caracteristicas = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
variedades = df["variety"].unique()

# Configurar subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8)) # Tamanho e quantidade

for i, caracteristica in enumerate(caracteristicas):
    ax = axs[i // 2, i % 2]
    
    for variedade in variedades:
        subset = df[df["variety"] == variedade]
        ax.hist(subset[caracteristica], bins=10, alpha=0.7, label=variedade)
    
    ax.set_title(f"Distribuição de {caracteristica}")
    ax.set_xlabel(caracteristica)
    ax.set_ylabel("Frequência")
    ax.legend()

plt.tight_layout()
plt.show()
