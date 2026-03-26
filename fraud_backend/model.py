import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "fraudTest.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
CATEGORY_MAP_PATH = os.path.join(BASE_DIR, "category_map.pkl")
CAT_AVG_PATH = os.path.join(BASE_DIR, "cat_avg.pkl")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def train_model():
    df = pd.read_csv(DATA_PATH)

    # Time
    df["hour"] = pd.to_datetime(df["unix_time"], unit="s").dt.hour
    df["day"] = pd.to_datetime(df["unix_time"], unit="s").dt.dayofweek

    # Age
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (pd.to_datetime("today") - df["dob"]).dt.days // 365

    # Distance
    df["distance"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    # Gender
    df["gender"] = df["gender"].map({"M": 0, "F": 1})

    # Category encoding
    df["category"] = df["category"].astype("category")
    category_mapping = dict(enumerate(df["category"].cat.categories))
    reverse_mapping = {v: k for k, v in category_mapping.items()}
    joblib.dump(reverse_mapping, CATEGORY_MAP_PATH)

    # 🚀 SAVE REAL CATEGORY AVERAGES (CRITICAL FIX)
    cat_avg_map = df.groupby("category")["amt"].mean().to_dict()
    joblib.dump(cat_avg_map, CAT_AVG_PATH)

    df["category"] = df["category"].cat.codes

    # 🚀 FEATURES
    df["log_amt"] = np.log1p(df["amt"])
    df["amt_to_category_ratio"] = df["amt"] / (df.groupby("category")["amt"].transform("mean") + 1)
    df["amt_per_population"] = df["amt"] / (df["city_pop"] + 1)

    df = df[[
        "amt",
        "log_amt",
        "amt_to_category_ratio",
        "amt_per_population",
        "city_pop",
        "hour",
        "day",
        "age",
        "distance",
        "gender",
        "category",
        "is_fraud"
    ]].dropna()

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    smote = SMOTE(sampling_strategy=0.2, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=2,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train_res, y_train_res)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("\n📊 MODEL PERFORMANCE")
    print("AUC:", round(auc, 4))
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, MODEL_PATH)

    return {"status": "Model trained 🚀", "auc": round(auc, 4)}


def load_model():
    return joblib.load(MODEL_PATH)


def load_category_map():
    return joblib.load(CATEGORY_MAP_PATH)


def load_cat_avg():
    return joblib.load(CAT_AVG_PATH)


def predict(data):
    model = load_model()
    category_map = load_category_map()
    cat_avg_map = load_cat_avg()

    category_name = data["category"]
    category_encoded = category_map.get(category_name, 0)

    avg = cat_avg_map.get(category_name, 5000)

    df = pd.DataFrame([{
        "amt": data["amount"],
        "log_amt": np.log1p(data["amount"]),
        "amt_to_category_ratio": data["amount"] / (avg + 1),
        "amt_per_population": data["amount"] / (data["city_pop"] + 1),
        "city_pop": data["city_pop"],
        "hour": data["hour"],
        "day": data["day"],
        "age": data["age"],
        "distance": data["distance"],
        "gender": 1 if data["gender"] == "F" else 0,
        "category": category_encoded
    }])

    return float(model.predict_proba(df)[0][1])