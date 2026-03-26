from fastapi import FastAPI
from pydantic import BaseModel
from model import train_model, predict
from agent import fraud_agent
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FraudGuard — Agentic Fraud Detection 🚀")

# ✅ CORS (frontend can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 INPUT SCHEMA
class Transaction(BaseModel):
    amount: float
    city_pop: int
    hour: int
    day: int
    age: int
    distance: float
    gender: str
    category: str

# 🏠 HEALTH CHECK
@app.get("/")
def home():
    return {"message": "FraudGuard backend running 🔥"}

# 🧠 TRAIN MODEL
@app.post("/train")
def train():
    return train_model()

# 🔍 PREDICT (FIXED — NO STREAMING BUGS)
@app.post("/predict")
def analyze(tx: Transaction):
    tx_data = tx.dict()

    # ML prediction
    prob = predict(tx_data)

    # Agent decision (ML + rules + actions)
    result = fraud_agent(prob, tx_data)

    return {
        "fraud_probability":  result["fraud_probability"],
        "risk_score":         result["risk_score"],
        "decision":           result["decision"],
        "risk_level":         result["risk"],
        "recommended_action": result["action"],
        "reasons":            result["reasons"],
        "contributions":      result["contributions"],
    }