from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

GREEN = "#1F5E3B"
OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/processed/prices_monthly.csv", parse_dates=["date"]).set_index("date")
palm = df["palm_oil"]

# 1. Riwayat harga
fig, ax = plt.subplots(figsize=(11, 4.5))
palm.plot(ax=ax, color=GREEN, lw=1.4)
ax.set_title("Palm oil price, monthly (nominal US$ per metric ton)")
ax.set_ylabel("US$ / mt")
ax.set_xlabel("")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "palm_oil_price_history.png", dpi=130)
plt.close(fig)

# Perubahan bulanan (%) sejak 2000
cols = ["palm_oil", "soybean_oil", "coconut_oil", "crude_oil_avg"]
ret = (df[cols].pct_change() * 100).dropna().loc["2000":]
pr = ret["palm_oil"]

# 2. Pola musiman
season = pr.groupby(pr.index.month).agg(["mean", "std", "count"])
season["ci95"] = 1.96 * season["std"] / season["count"] ** 0.5
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(season.index, season["mean"], yerr=season["ci95"], color=GREEN, capsize=3, alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months)
ax.set_title("Average monthly change in palm oil price by calendar month, 2000 onward (error bars: 95% CI)")
ax.set_ylabel("% change")
fig.tight_layout()
fig.savefig(OUT / "palm_oil_seasonality.png", dpi=130)
plt.close(fig)

# 3. Korelasi perubahan bulanan
corr = ret.corr()
fig, ax = plt.subplots(figsize=(6, 4.8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1, ax=ax)
ax.set_title("Correlation of monthly price changes, 2000 onward")
fig.tight_layout()
fig.savefig(OUT / "price_change_correlation.png", dpi=130)
plt.close(fig)

# Angka kunci
print("Harga minyak sawit terakhir:", palm.index[-1].date(), "=", round(float(palm.iloc[-1]), 1), "$/mt")
print("Perubahan dibanding 12 bulan sebelumnya:", round((palm.iloc[-1] / palm.iloc[-13] - 1) * 100, 1), "%")
print("Perubahan bulanan sejak 2000: rata-rata", round(pr.mean(), 2), "% | simpangan baku", round(pr.std(), 2), "%")
print("Autokorelasi perubahan bulanan (lag 1):", round(pr.autocorr(1), 3))
print("Autokorelasi perubahan bulanan (lag 12):", round(pr.autocorr(12), 3))

print("\nPola musiman (perubahan bulanan, %):")
print(season.round(2).to_string())

print("\nKorelasi perubahan bulanan:")
print(corr.round(2).to_string())

print("\n5 penurunan bulanan terbesar (%):")
print(pr.nsmallest(5).round(1).to_string())
print("\n5 kenaikan bulanan terbesar (%):")
print(pr.nlargest(5).round(1).to_string())

print("\nGrafik tersimpan di reports/figures/")