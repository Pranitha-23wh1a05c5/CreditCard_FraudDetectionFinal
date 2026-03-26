import os
import requests
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

# ─────────────────────────────────────────────
# MEMORY (simple session memory)
# ─────────────────────────────────────────────
AGENT_MEMORY = []

# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────
def _call_llm(prompt: str) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=8,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception:
        return ""

# ─────────────────────────────────────────────
# ACTION TOOLS (Agent abilities)
# ─────────────────────────────────────────────
def block_transaction(tx):
    return f"🚫 Transaction BLOCKED for ${tx['amount']}"

def flag_transaction(tx):
    return f"⚠️ Transaction FLAGGED for review"

def approve_transaction(tx):
    return f"✅ Transaction APPROVED"

def log_case(tx, decision):
    entry = {
        "time": str(datetime.now()),
        "amount": tx["amount"],
        "decision": decision
    }
    AGENT_MEMORY.append(entry)
    return "🧾 Case logged"

# ─────────────────────────────────────────────
# ORIGINAL LOGIC (UNCHANGED)
# ─────────────────────────────────────────────
MINOR_AGE   = 18
BLOCK_SCORE = 70
FLAG_SCORE  = 35

def fraud_agent_logic(prob, tx):
    contributions = []

    amt   = tx["amount"]
    age   = tx["age"]
    dist  = tx["distance"]
    hour  = tx["hour"]
    cat   = tx["category"]
    pop   = tx["city_pop"]

    model_score = min(prob * 40, 40)
    risk_score  = model_score
    contributions.append(("Model signal", round(model_score, 2)))

    if age < MINOR_AGE:
        return {
            "decision": "BLOCK",
            "risk": "HIGH",
            "action": "Block transaction immediately — minor using card",
            "risk_score": 100,
            "fraud_probability": round(prob, 4),
            "reasons": ["Minor using card"],
            "contributions": contributions + [("Hard Rule: Minor usage", 100)]
        }

    if amt > 100000:
        risk_score += 40; contributions.append(("Amount", 40))
    elif amt > 50000:
        risk_score += 30; contributions.append(("Amount", 30))
    elif amt > 20000:
        risk_score += 25; contributions.append(("Amount", 25))
    elif amt > 5000:
        risk_score += 15; contributions.append(("Amount", 15))

    if age < 21 or age > 65:
        risk_score += 10; contributions.append(("Age", 10))

    if dist > 2000:
        risk_score += 30; contributions.append(("Distance", 30))
    elif dist > 500:
        risk_score += 12; contributions.append(("Distance", 12))

    if 0 <= hour <= 5:
        risk_score += 12; contributions.append(("Odd hours", 12))

    if cat in {"travel", "misc_net"}:
        risk_score += 15; contributions.append(("Category", 15))

    if risk_score >= BLOCK_SCORE:
        decision, risk, action = "BLOCK", "HIGH", "Block transaction"
    elif risk_score >= FLAG_SCORE:
        decision, risk, action = "FLAG", "MEDIUM", "Review transaction"
    else:
        decision, risk, action = "APPROVE", "LOW", "Approve transaction"

    return {
        "decision": decision,
        "risk": risk,
        "action": action,
        "risk_score": round(risk_score, 2),
        "fraud_probability": round(prob, 4),
        "reasons": [],
        "contributions": contributions
    }

# ─────────────────────────────────────────────
# 🧠 AGENT LAYER (NEW 🔥)
# ─────────────────────────────────────────────
def fraud_agent(prob, tx):

    # STEP 1: THINK (use your existing logic)
    result = fraud_agent_logic(prob, tx)

    decision = result["decision"]

    # STEP 2: ACT (this is what makes it agentic)
    if decision == "BLOCK":
        action_result = block_transaction(tx)
    elif decision == "FLAG":
        action_result = flag_transaction(tx)
    else:
        action_result = approve_transaction(tx)

    # STEP 3: LOG (memory)
    log_result = log_case(tx, decision)

    # STEP 4: OPTIONAL LLM reasoning
    prompt = f"""
Explain briefly why this transaction was {decision}.
Amount: {tx['amount']}, Category: {tx['category']}, Age: {tx['age']}
"""
    explanation = _call_llm(prompt)

    # STEP 5: UPDATE RESPONSE
    result["agent_action"] = action_result
    result["agent_log"] = log_result
    if explanation:
        result["reasons"] = [explanation]

    return result