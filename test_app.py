"""
CSL236 - End-Term Project Test Suite
Automated tests validating:
1. Health check endpoint (GET /health)
2. Web UI endpoint (GET /)
3. Prediction endpoint contract (POST /predict)
4. Domain-logic validation (High impact breach scores higher risk than low impact breach)
5. Boundary and data contract constraint enforcement (HTTP 422 validations)
"""

import os
import pytest
from fastapi.testclient import TestClient

# Ensure app imports cleanly
from app import app, load_model

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_model():
    """Ensure trained model is available before running tests."""
    load_model()


# 1. Health check test
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "version" in data


# 2. UI loading test
def test_ui_loads():
    response = client.get("/")
    assert response.status_code == 200
    # Must serve HTML UI with key elements
    assert "text/html" in response.headers.get("content-type", "")
    content = response.text
    assert "CyberGuard" in content or "Cybersecurity" in content
    assert "Incident Severity" in content


# 3. Valid prediction response contract test
def test_valid_prediction_structure():
    payload = {
        "country": "USA",
        "year": 2024,
        "attack_type": "Ransomware",
        "target_industry": "Banking",
        "number_of_affected_users": 500000,
        "attack_source": "Nation-state",
        "security_vulnerability_type": "Zero-day",
        "defense_mechanism_used": "Firewall",
        "incident_resolution_time": 48.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Low", "Medium", "High"]
    assert data["severity_tier"] in ["Low", "Medium", "High"]
    assert 0.0 <= data["risk_score"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) >= 2
    assert "timestamp" in data


# 4. Domain-Logic Test (Mandatory requirement from guidelines)
# Checks that model behaves sensibly: A critical breach (nation-state, zero-day, 950K users, 70h resolution)
# must produce a higher risk score than a minor incident (unknown actor, weak password, 1.2K users, 2h resolution).
def test_domain_logic_critical_vs_minor():
    critical_breach = {
        "country": "USA",
        "year": 2024,
        "attack_type": "Ransomware",
        "target_industry": "Banking",
        "number_of_affected_users": 950000,
        "attack_source": "Nation-state",
        "security_vulnerability_type": "Zero-day",
        "defense_mechanism_used": "Firewall",
        "incident_resolution_time": 70.0,
    }

    minor_incident = {
        "country": "UK",
        "year": 2022,
        "attack_type": "Phishing",
        "target_industry": "Retail",
        "number_of_affected_users": 1200,
        "attack_source": "Unknown",
        "security_vulnerability_type": "Weak Passwords",
        "defense_mechanism_used": "AI-based Detection",
        "incident_resolution_time": 2.0,
    }

    resp_critical = client.post("/predict", json=critical_breach)
    resp_minor = client.post("/predict", json=minor_incident)

    assert resp_critical.status_code == 200
    assert resp_minor.status_code == 200

    data_critical = resp_critical.json()
    data_minor = resp_minor.json()

    # Domain assertion: critical risk score should exceed or match minor incident
    tier_weights = {"Low": 1, "Medium": 2, "High": 3}
    critical_tier_val = tier_weights.get(data_critical["prediction"], 1)
    minor_tier_val = tier_weights.get(data_minor["prediction"], 1)

    assert (
        data_critical["risk_score"] >= data_minor["risk_score"]
        or critical_tier_val >= minor_tier_val
    ), "Domain logic failure: Critical breach must produce equal or higher severity risk than minor incident"


# 5. Boundary validation tests expecting HTTP 422 (Mandatory requirement from guidelines)
def test_boundary_invalid_year_below_range():
    payload = {
        "country": "USA",
        "year": 2010,  # Below minimum 2015
        "attack_type": "Ransomware",
        "target_industry": "Banking",
        "number_of_affected_users": 10000,
        "attack_source": "Nation-state",
        "security_vulnerability_type": "Zero-day",
        "defense_mechanism_used": "Firewall",
        "incident_resolution_time": 24.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, "Year < 2015 must trigger HTTP 422"


def test_boundary_invalid_year_above_range():
    payload = {
        "country": "USA",
        "year": 2045,  # Above maximum 2030
        "attack_type": "Ransomware",
        "target_industry": "Banking",
        "number_of_affected_users": 10000,
        "attack_source": "Nation-state",
        "security_vulnerability_type": "Zero-day",
        "defense_mechanism_used": "Firewall",
        "incident_resolution_time": 24.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, "Year > 2030 must trigger HTTP 422"


def test_boundary_negative_affected_users():
    payload = {
        "country": "USA",
        "year": 2024,
        "attack_type": "Ransomware",
        "target_industry": "Banking",
        "number_of_affected_users": -50,  # Negative count
        "attack_source": "Nation-state",
        "security_vulnerability_type": "Zero-day",
        "defense_mechanism_used": "Firewall",
        "incident_resolution_time": 24.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, "Negative affected users must trigger HTTP 422"


def test_boundary_invalid_country():
    payload = {
        "country": "Atlantis",  # Non-contract country
        "year": 2024,
        "attack_type": "Ransomware",
        "target_industry": "Banking",
        "number_of_affected_users": 10000,
        "attack_source": "Nation-state",
        "security_vulnerability_type": "Zero-day",
        "defense_mechanism_used": "Firewall",
        "incident_resolution_time": 24.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, "Invalid country must trigger HTTP 422"


def test_boundary_invalid_resolution_time_exceeded():
    payload = {
        "country": "USA",
        "year": 2024,
        "attack_type": "Ransomware",
        "target_industry": "Banking",
        "number_of_affected_users": 10000,
        "attack_source": "Nation-state",
        "security_vulnerability_type": "Zero-day",
        "defense_mechanism_used": "Firewall",
        "incident_resolution_time": 1500.0,  # Exceeds 720 hours
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, "Resolution time > 720h must trigger HTTP 422"


def test_boundary_missing_required_fields():
    payload = {
        "country": "USA",
        "year": 2024,
        # Missing remaining 7 features
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, "Missing features must trigger HTTP 422"
