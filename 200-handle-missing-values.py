# from google.colab import drive
# drive.mount('/content/drive')
# ===========================
from pandas import read_csv

df = read_csv("/content/drive/My Drive/Colab/setik/dataset/01-remove-features.csv")
# ===========================
df.head()
# ===========================
print("dataset missing values\n")

for col in df.columns.values.tolist():
    n_miss = df[col].isna().sum()
    perc = n_miss / df.shape[0] * 100
    perc = round(perc, 2)
    print(f"{col} : {n_miss} ({perc}%)")
# ===========================
from sklearn.model_selection import train_test_split

y = df.price
X = df.drop(["price"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)
# ===========================
from numpy import number

num_train = X_train.select_dtypes(include=[number])
cat_train = X_train.select_dtypes(exclude=[number])
# ===========================
for col_types in (num_train, cat_train):
    for col in col_types.columns.values.tolist():
        n_miss = col_types[col].isna().sum()
        perc = n_miss / df.shape[0]
        perc = round(perc, 2)
        print(f"{col} : {n_miss} ({perc}%)")
    print()
# ===========================
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# use IterativeImputer
numerical_transformer = SimpleImputer(strategy="constant")
categorical_transformer = Pipeline(
    steps=[
        # ? use IterativeImputer
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # not all cols use OneHotEncoder
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numerical_transformer, num_train.columns),
        # change accordingly
        ("cat", categorical_transformer, cat_train.columns),
    ]
)

model = RandomForestRegressor(n_estimators=5, random_state=0)

my_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_test)
score = mean_absolute_error(y_test, preds)
print("MAE:", score)
