from pathlib import Path

import pandas as pd

path = Path("data/raw/CMO-Historical-Data-Monthly.xlsx")
xls = pd.ExcelFile(path)
print("Sheet:", xls.sheet_names)

sheet = next((s for s in xls.sheet_names if "monthly prices" in s.lower()), None)
if sheet is None:
    raise SystemExit("Sheet 'Monthly Prices' tidak ditemukan, kirim daftar sheet di atas.")

raw = pd.read_excel(path, sheet_name=sheet, header=None)
print("\nSheet yang dipakai:", sheet)
print("Ukuran sheet (baris, kolom):", raw.shape)

print("\n8 baris pertama, 6 kolom pertama:")
print(raw.iloc[:8, :6].to_string())

print("\nKolom yang mengandung palm, soybean, coconut, atau crude oil (di 10 baris pertama):")
for r in range(min(10, len(raw))):
    for c in range(raw.shape[1]):
        v = str(raw.iat[r, c]).lower()
        if any(k in v for k in ["palm", "soybean", "coconut", "crude oil"]):
            print(f"  baris {r}, kolom {c}: {raw.iat[r, c]}")

print("\n5 baris terakhir, 3 kolom pertama:")
print(raw.iloc[-5:, :3].to_string())