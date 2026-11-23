from pathlib import Path

import pandas as pd

RAW = Path("data/raw/CMO-Historical-Data-Monthly.xlsx")
OUT = Path("data/processed/prices_monthly.csv")

# Posisi kolom (mulai dari 0) di sheet "Monthly Prices" dan nama barunya.
# Satuan: crude_oil_avg dalam $/bbl, lainnya dalam $/mt (dolar nominal).
COLUMNS = {
    1: "crude_oil_avg",
    18: "coconut_oil",
    22: "palm_oil",
    23: "palm_kernel_oil",
    25: "soybean_oil",
}
EXPECTED = {
    1: "Crude oil, average",
    18: "Coconut oil",
    22: "Palm oil",
    23: "Palm kernel oil",
    25: "Soybean oil",
}
HEADER_ROW = 4
FIRST_DATA_ROW = 6

raw = pd.read_excel(RAW, sheet_name="Monthly Prices", header=None)

# Pastikan kolom yang diambil benar-benar yang dimaksud
for c, name in EXPECTED.items():
    found = str(raw.iat[HEADER_ROW, c]).strip()
    assert found == name, f"Kolom {c}: diharapkan '{name}', ditemukan '{found}'"

data = raw.iloc[FIRST_DATA_ROW:, [0] + list(COLUMNS)].copy()
data.columns = ["period"] + list(COLUMNS.values())
data["period"] = data["period"].astype(str).str.strip()
data = data[data["period"].str.match(r"^\d{4}M\d{2}$")]
data["date"] = pd.to_datetime(data["period"].str.replace("M", "-", regex=False) + "-01")

for col in COLUMNS.values():
    data[col] = pd.to_numeric(data[col], errors="coerce")  # tanda '…' menjadi NaN

data = data[["date"] + list(COLUMNS.values())].sort_values("date").reset_index(drop=True)

print("Baris:", len(data), "| rentang:", data["date"].min().date(), "sampai", data["date"].max().date())
gaps = data["date"].diff().dt.days.dropna()
print("Selisih tanggal terbesar antar baris (hari, harusnya 31):", int(gaps.max()))

print("\nData kosong per kolom:")
print(data.isna().sum().to_string())

print("\nRentang data yang ada untuk tiap kolom:")
for col in COLUMNS.values():
    s = data.loc[data[col].notna(), "date"]
    print(f"  {col}: {s.min().date()} sampai {s.max().date()}")

print("\n6 baris terakhir:")
print(data.tail(6).to_string(index=False))

print("\nStatistik harga minyak sawit ($/mt):")
print(data["palm_oil"].describe().round(1).to_string())

OUT.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUT, index=False)
print("\nTersimpan di", OUT)