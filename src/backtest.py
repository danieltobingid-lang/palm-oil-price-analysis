import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

HORIZONS = [1, 3, 6, 12]
TEST_START = "2010-01-01"  # origin ramalan pertama
ARIMA_WINDOW = 240         # bulan terakhir yang dipakai ARIMA dan ETS
RIDGE_ALPHA = 10.0         # ditetapkan di awal, tidak disetel pada data uji
MODELS = ["naive", "drift", "arima", "ets", "ridge"]

df = pd.read_csv("data/processed/prices_monthly.csv", parse_dates=["date"]).set_index("date").asfreq("MS")
ly = np.log(df[["palm_oil", "soybean_oil", "coconut_oil", "crude_oil_avg"]])
y = ly["palm_oil"]

feat = pd.DataFrame({
    "p1": y.diff(1), "p3": y.diff(3), "p6": y.diff(6), "p12": y.diff(12),
    "s1": ly["soybean_oil"].diff(1), "s3": ly["soybean_oil"].diff(3),
    "c1": ly["coconut_oil"].diff(1), "o1": ly["crude_oil_avg"].diff(1),
})
targets = {h: y.shift(-h) - y for h in HORIZONS}

origins = np.where(y.index >= pd.Timestamp(TEST_START))[0]
origins = origins[origins < len(y) - 1]
max_h = max(HORIZONS)

rows = []
for n, i in enumerate(origins, start=1):
    hist = y.iloc[: i + 1]
    last = hist.iloc[-1]
    drift_per_month = (hist.iloc[-1] - hist.iloc[0]) / (len(hist) - 1)
    window = hist.iloc[-ARIMA_WINDOW:]

    try:
        arima_fc = ARIMA(window, order=(1, 1, 1)).fit().forecast(max_h).values
    except Exception:
        arima_fc = np.full(max_h, np.nan)
    try:
        ets_fc = ExponentialSmoothing(
            window, trend="add", damped_trend=True, initialization_method="estimated"
        ).fit().forecast(max_h).values
    except Exception:
        ets_fc = np.full(max_h, np.nan)

    for h in HORIZONS:
        if i + h >= len(y):
            continue
        actual = y.iloc[i + h]
        preds = {
            "naive": last,
            "drift": last + drift_per_month * h,
            "arima": arima_fc[h - 1],
            "ets": ets_fc[h - 1],
        }
        # ridge hanya dilatih dari origin yang targetnya sudah diketahui pada waktu i
        end = i - h + 1
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(
            feat.iloc[12:end], targets[h].iloc[12:end]
        )
        preds["ridge"] = last + ridge.predict(feat.iloc[[i]])[0]

        for name, p in preds.items():
            rows.append((y.index[i], h, name, p - actual, np.sign(p - last), np.sign(actual - last)))

    if n % 40 == 0:
        print(f"  ... {n}/{len(origins)} origin selesai")

res = pd.DataFrame(rows, columns=["origin", "h", "model", "err", "pred_dir", "true_dir"]).dropna(subset=["err"])
res["abs_err"] = res["err"].abs()

summary = res.groupby(["h", "model"]).agg(
    n=("err", "size"),
    mae_pct=("abs_err", lambda s: 100 * s.mean()),
    rmse_pct=("err", lambda s: 100 * np.sqrt((s ** 2).mean())),
).reset_index()
naive = summary[summary["model"] == "naive"].set_index("h")
summary["mae_vs_naive"] = summary.apply(lambda r: r["mae_pct"] / naive.loc[r["h"], "mae_pct"], axis=1)
summary["rmse_vs_naive"] = summary.apply(lambda r: r["rmse_pct"] / naive.loc[r["h"], "rmse_pct"], axis=1)

Path("reports/figures").mkdir(parents=True, exist_ok=True)
summary.to_csv("reports/backtest_summary.csv", index=False)

print(f"\nBacktest walk-forward: {len(origins)} origin, {y.index[origins[0]].date()} sampai {y.index[origins[-1]].date()}")
print("Galat dihitung pada skala logaritmik, mendekati persen.")

print("\nMAE (%), makin kecil makin baik:")
print(summary.pivot(index="model", columns="h", values="mae_pct").loc[MODELS].round(1).to_string())
print("\nMAE relatif terhadap naive (di bawah 1,00 = lebih baik dari naive):")
print(summary.pivot(index="model", columns="h", values="mae_vs_naive").loc[MODELS].round(3).to_string())
print("\nRMSE relatif terhadap naive:")
print(summary.pivot(index="model", columns="h", values="rmse_vs_naive").loc[MODELS].round(3).to_string())

piv = res.pivot_table(index=["origin", "h"], columns="model", values="abs_err")
win = {}
for h in HORIZONS:
    sub = piv.xs(h, level="h")
    win[h] = {m: round(100 * (sub[m] < sub["naive"]).mean(), 1) for m in MODELS[1:]}
print("\nPersen origin di mana model lebih akurat dari naive:")
print(pd.DataFrame(win).to_string())

d = res[res["model"] != "naive"].copy()
d["hit"] = d["pred_dir"] == d["true_dir"]
print("\nAkurasi arah (% tebakan naik/turun yang benar):")
print((d.groupby(["model", "h"])["hit"].mean().mul(100).round(1)).unstack("h").to_string())
up = res[res["model"] == "naive"].groupby("h")["true_dir"].apply(lambda s: round(100 * (s > 0).mean(), 1))
print("Pembanding: persentase kasus harga benar-benar naik pada tiap horizon:")
print(up.to_string())

res["period"] = np.where(res["origin"] < pd.Timestamp("2020-01-01"), "2010-2019", "2020-2026")
by_period = res.groupby(["period", "h", "model"])["abs_err"].mean().mul(100).unstack("model")
print("\nMAE relatif terhadap naive per periode:")
print(by_period[MODELS[1:]].div(by_period["naive"], axis=0).round(2).to_string())

rel = summary.pivot(index="h", columns="model", values="mae_vs_naive")[MODELS[1:]]
ax = rel.plot(kind="bar", figsize=(9, 4.5), width=0.8)
ax.axhline(1.0, color="black", lw=1)
ax.set_ylabel("MAE relative to naive (below 1 = better)")
ax.set_xlabel("Forecast horizon (months)")
ax.set_title("Walk-forward backtest of palm oil price forecasts (2010 onward)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/figures/backtest_relative_mae.png", dpi=130)
print("\nTersimpan: reports/backtest_summary.csv dan reports/figures/backtest_relative_mae.png")