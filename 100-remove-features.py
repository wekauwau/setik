# from google.colab import drive
# drive.mount('/content/drive')
# ===========================
from pandas import read_csv

df = read_csv("/content/drive/My Drive/Colab/setik/dataset/vehicles.csv")
# ===========================
for i in df.columns:
    print(i)
# ===========================
df_filtered = df.filter(
    [
        "price",
        "year",
        "manufacturer",
        "model",
        "condition",
        "cylinders",
        "fuel",
        "odometer",
        "transmission",
        "drive",
        "type",
        "paint_color",
        "posting_date",
    ]
)
# ===========================
df.shape
# ===========================
df_filtered.shape
# ===========================
df_filtered.to_csv(
    "/content/drive/My Drive/Colab/setik/dataset/01-remove-features.csv", index=False
)
