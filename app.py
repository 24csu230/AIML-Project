import json
import os
from datetime import datetime
from typing import Dict, List, Literal, Optional
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
LOG_FILE_PATH = os.path.join(BASE_DIR, "prediction_logs.jsonl")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Allowed categorical sets (per data contract)
ALLOWED_COUNTRIES = [
    "China", "India", "UK", "Germany", "France",
    "Australia", "Russia", "Brazil", "Japan", "USA"
]
ALLOWED_ATTACK_TYPES = [
    "Phishing", "Ransomware", "Man-in-the-Middle",
    "DDoS", "SQL Injection", "Malware"
]
ALLOWED_INDUSTRIES = [
    "Education", "Retail", "IT", "Telecommunications",
    "Government", "Banking", "Healthcare"
]
ALLOWED_SOURCES = ["Hacker Group", "Nation-state", "Insider", "Unknown"]
ALLOWED_VULNERABILITIES = [
    "Unpatched Software", "Weak Passwords", "Social Engineering", "Zero-day"
]
ALLOWED_DEFENSES = [
    "VPN", "Firewall", "AI-based Detection", "Antivirus", "Encryption"
]

# Global model instance (loaded ONCE at startup outside request handlers)
ml_model = None


def load_model():
    """Load serialized model once at startup."""
    global ml_model
    if os.path.exists(MODEL_PATH):
        try:
            ml_model = joblib.load(MODEL_PATH)
            print(f"[+] Loaded trained model from {MODEL_PATH}")
        except Exception as e:
            print(f"[-] Error loading model from {MODEL_PATH}: {e}")
            ml_model = None
    else:
        print(f"[!] Warning: Model file {MODEL_PATH} not found. Please run train.py first.")
        ml_model = None
    return ml_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model once
    load_model()
    yield
    # Shutdown logic (if any)


app = FastAPI(
    title="Cybersecurity Threat Incident Severity Predictor",
    description="CSL236 MLOps API for classifying cyber incident severity tiers and providing automated SOC mitigation rules.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if directory exists
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class IncidentPredictionRequest(BaseModel):
    country: str = Field(
        ...,
        description="Country where the incident occurred",
        example="USA",
    )
    year: int = Field(
        ...,
        ge=2015,
        le=2030,
        description="Year of the security breach (2015 - 2030)",
        example=2024,
    )
    attack_type: str = Field(
        ...,
        description="Type of cyber attack",
        example="Ransomware",
    )
    target_industry: str = Field(
        ...,
        description="Industry sector targeted",
        example="Banking",
    )
    number_of_affected_users: int = Field(
        ...,
        ge=1,
        le=50000000,
        description="Number of affected/compromised user accounts (1 - 50,000,000)",
        example=450000,
    )
    attack_source: str = Field(
        ...,
        description="Threat actor attribution",
        example="Nation-state",
    )
    security_vulnerability_type: str = Field(
        ...,
        description="Primary root-cause vulnerability exploited",
        example="Zero-day",
    )
    defense_mechanism_used: str = Field(
        ...,
        description="Primary security defense perimeter deployed",
        example="Firewall",
    )
    incident_resolution_time: float = Field(
        ...,
        ge=0.1,
        le=720.0,
        description="Time taken to contain and resolve the incident in hours (0.1 - 720.0)",
        example=48.5,
    )

    @field_validator("country")
    def validate_country(cls, v):
        for allowed in ALLOWED_COUNTRIES:
            if v.strip().lower() == allowed.lower():
                return allowed
        raise ValueError(f"Invalid country '{v}'. Must be one of: {ALLOWED_COUNTRIES}")

    @field_validator("attack_type")
    def validate_attack_type(cls, v):
        for allowed in ALLOWED_ATTACK_TYPES:
            if v.strip().lower() == allowed.lower():
                return allowed
        raise ValueError(f"Invalid attack_type '{v}'. Must be one of: {ALLOWED_ATTACK_TYPES}")

    @field_validator("target_industry")
    def validate_target_industry(cls, v):
        for allowed in ALLOWED_INDUSTRIES:
            if v.strip().lower() == allowed.lower():
                return allowed
        raise ValueError(f"Invalid target_industry '{v}'. Must be one of: {ALLOWED_INDUSTRIES}")

    @field_validator("attack_source")
    def validate_attack_source(cls, v):
        for allowed in ALLOWED_SOURCES:
            if v.strip().lower() == allowed.lower():
                return allowed
        raise ValueError(f"Invalid attack_source '{v}'. Must be one of: {ALLOWED_SOURCES}")

    @field_validator("security_vulnerability_type")
    def validate_security_vulnerability_type(cls, v):
        for allowed in ALLOWED_VULNERABILITIES:
            if v.strip().lower() == allowed.lower():
                return allowed
        raise ValueError(f"Invalid security_vulnerability_type '{v}'. Must be one of: {ALLOWED_VULNERABILITIES}")

    @field_validator("defense_mechanism_used")
    def validate_defense_mechanism_used(cls, v):
        for allowed in ALLOWED_DEFENSES:
            if v.strip().lower() == allowed.lower():
                return allowed
        raise ValueError(f"Invalid defense_mechanism_used '{v}'. Must be one of: {ALLOWED_DEFENSES}")


class IncidentPredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted Incident Severity Tier: Low, Medium, or High")
    severity_tier: str = Field(..., description="Alias for prediction tier")
    risk_score: float = Field(..., description="Normalized incident risk score from 0.0 to 1.0")
    confidence: float = Field(..., description="Model confidence score for the predicted tier")
    class_probabilities: Dict[str, float] = Field(..., description="Probability breakdown across Low, Medium, High")
    estimated_financial_loss_range: str = Field(..., description="Estimated economic breach impact range in USD Millions")
    recommendations: List[str] = Field(..., description="Actionable rule-based defense mitigation tips")
    timestamp: str = Field(..., description="ISO 8601 prediction timestamp")


def generate_security_recommendations(req: IncidentPredictionRequest, predicted_tier: str) -> List[str]:
    """Generate 2-3 tailored rule-based defense recommendations for the SOC team."""
    tips = []

    # 1. Attack type specific rule
    if req.attack_type == "Ransomware":
        tips.append("Isolate all active directory domain controllers and disconnect offline air-gapped immutable backup storage immediately.")
    elif req.attack_type == "DDoS":
        tips.append("Activate Border Gateway Protocol (BGP) routing to an upstream scrubbing center and enforce rate-limiting on ingress API gateways.")
    elif req.attack_type == "SQL Injection":
        tips.append("Deploy parameterized queries/ORMs across all backend endpoints and activate Web Application Firewall (WAF) deep inspection rules.")
    elif req.attack_type == "Phishing":
        tips.append("Revoke active user session tokens, purge matching email hash indicators from all mailboxes, and trigger emergency phishing awareness alerts.")
    elif req.attack_type == "Man-in-the-Middle":
        tips.append("Enforce strict TLS 1.3 with HTTP Strict Transport Security (HSTS) and certificate pinning across internal microservice meshes.")
    else:  # Malware
        tips.append("Trigger automated Endpoint Detection and Response (EDR) host isolation and collect volatile memory dumps for reverse-engineering.")

    # 2. Vulnerability / Root cause specific rule
    if req.security_vulnerability_type == "Zero-day":
        tips.append("Zero-day exploit detected: Engage tier-3 threat intelligence, apply behavioral anomaly containment, and engage vendor security advisory for emergency out-of-band mitigation.")
    elif req.security_vulnerability_type == "Unpatched Software":
        tips.append("Implement automated Common Vulnerabilities and Exposures (CVE) patch management with prioritized 24-hour SLA for critical internet-facing assets.")
    elif req.security_vulnerability_type == "Weak Passwords":
        tips.append("Mandate hardware-backed FIDO2 / WebAuthn Multi-Factor Authentication (MFA) and enforce credential-stuffing prevention with breached password checks.")
    else:  # Social Engineering
        tips.append("Enforce multi-party cryptographic authorization for sensitive outbound financial transactions and administrative privilege escalations.")

    # 3. Severity & Defense enhancement rule
    if predicted_tier == "High" or req.attack_source == "Nation-state":
        tips.append("High-severity nation-state attack profile: Initiate corporate crisis communications, notify cybersecurity regulatory oversight bodies within 72 hours, and contract third-party forensic specialists.")
    elif req.defense_mechanism_used == "Firewall" or req.defense_mechanism_used == "Antivirus":
        tips.append(f"Legacy {req.defense_mechanism_used} proved insufficient: Accelerate migration to an AI-powered XDR (Extended Detection and Response) and Zero-Trust architecture.")
    else:
        tips.append("Conduct comprehensive post-incident Root Cause Analysis (RCA) and update threat hunting SIEM rule correlation pipelines.")

    return tips[:3]


def log_prediction(request_data: dict, response_data: dict):
    """Log prediction input and output to disk for data drift monitoring (Stage 8 requirement)."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": request_data,
        "output": response_data,
    }
    try:
        with open(LOG_FILE_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[-] Logging error: {e}")


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    """Health check endpoint required by Stage 4 & CI tests."""
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "model_type": type(ml_model.named_steps["classifier"]).__name__ if ml_model else None,
        "version": "1.0.0",
        "service": "Cybersecurity Incident Severity Predictor",
    }


@app.get("/", tags=["UI"])
async def serve_ui():
    """Serve the web application user interface at root /."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(
        content={
            "message": "Cybersecurity Threat API is operational. Visit /docs for Swagger UI documentation.",
            "health": "/health",
            "predict": "/predict",
        }
    )


@app.post("/predict", response_model=IncidentPredictionResponse, status_code=status.HTTP_200_OK, tags=["Prediction"])
async def predict_severity(req: IncidentPredictionRequest):
    """
    Predict cybersecurity incident severity tier and return tailored SOC defense tips.
    Enforces strict Pydantic validation: any out-of-range or wrongly typed input returns HTTP 422.
    """
    global ml_model
    if ml_model is None:
        ml_model = load_model()
        if ml_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained ML model is not loaded. Run train.py to generate model/model.joblib.",
            )

    # Format input row matching training features
    input_dict = {
        "country": req.country,
        "year": req.year,
        "attack_type": req.attack_type,
        "target_industry": req.target_industry,
        "number_of_affected_users": req.number_of_affected_users,
        "attack_source": req.attack_source,
        "security_vulnerability_type": req.security_vulnerability_type,
        "defense_mechanism_used": req.defense_mechanism_used,
        "incident_resolution_time": req.incident_resolution_time,
    }
    input_df = pd.DataFrame([input_dict])

    try:
        # Prediction
        prediction_arr = ml_model.predict(input_df)
        predicted_tier = str(prediction_arr[0])

        # Probabilities
        class_probs = {}
        if hasattr(ml_model, "predict_proba"):
            probs = ml_model.predict_proba(input_df)[0]
            classes = ml_model.classes_
            class_probs = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
            confidence = round(float(max(probs)), 4)
            # Normalized risk score: weight Low=0.2, Medium=0.5, High=0.9
            risk_score = round(
                class_probs.get("Low", 0.0) * 0.2
                + class_probs.get("Medium", 0.0) * 0.55
                + class_probs.get("High", 0.0) * 0.95,
                4,
            )
        else:
            confidence = 1.0
            risk_score = 0.9 if predicted_tier == "High" else (0.55 if predicted_tier == "Medium" else 0.2)
            class_probs = {predicted_tier: 1.0}

        # Estimate financial impact range based on tier
        if predicted_tier == "High":
            loss_range = "$66.7M – $100.0M+ (Critical Enterprise Impact)"
        elif predicted_tier == "Medium":
            loss_range = "$33.4M – $66.6M (Moderate Impact)"
        else:
            loss_range = "$0.5M – $33.3M (Controlled / Low Impact)"

        # Contextual mitigation tips
        tips = generate_security_recommendations(req, predicted_tier)

        response_payload = {
            "prediction": predicted_tier,
            "severity_tier": predicted_tier,
            "risk_score": risk_score,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "estimated_financial_loss_range": loss_range,
            "recommendations": tips,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Stage 8 requirement: Log every prediction input to file for drift checks
        log_prediction(input_dict, response_payload)

        return response_payload

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.get("/logs", tags=["Monitoring"])
async def get_recent_logs(limit: int = 50):
    """Retrieve recent prediction logs for monitoring and drift tracking."""
    if not os.path.exists(LOG_FILE_PATH):
        return {"logs": [], "total_entries": 0}

    logs = []
    with open(LOG_FILE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return {"logs": logs[-limit:], "total_entries": len(logs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
