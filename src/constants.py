"""
Constants for weight stream processor.
All hard-coded values that should not be configurable for safety.
"""

# Physiological limits (hard-coded for safety)
MIN_WEIGHT = 30.0  # kg
MAX_WEIGHT = 400.0  # kg
MIN_VALID_BMI = 10.0
MAX_VALID_BMI = 90.0
DEFAULT_HEIGHT_M = 1.67

# Optimized physiological change limits
MAX_CHANGE_1H_PERCENT = 0.02
MAX_CHANGE_6H_PERCENT = 0.025
MAX_CHANGE_24H_PERCENT = 0.035
MAX_CHANGE_1H_ABSOLUTE = 4.22  # kg
MAX_CHANGE_6H_ABSOLUTE = 6.23  # kg
MAX_CHANGE_24H_ABSOLUTE = 6.44  # kg
MAX_SUSTAINED_DAILY_KG = 2.57  # kg/day

# Session detection
SESSION_TIMEOUT_MINUTES = 5.0
SESSION_VARIANCE_THRESHOLD = 5.81  # kg

# Tolerances
LIMIT_TOLERANCE = 0.2493
SUSTAINED_TOLERANCE = 0.50

# Source-specific noise multipliers (data-driven from 709K+ measurements)
SOURCE_NOISE_MULTIPLIERS = {
    "patient-upload": 1.0,              # Most reliable baseline
    "care-team-upload": 1.2,            # Slightly less reliable
    "internal-questionnaire": 1.6,      # Questionnaires are noisier
    "initial-questionnaire": 1.6,
    "questionnaire": 1.6,
    "https://connectivehealth.io": 2.2, # Moderate reliability
    "patient-device": 2.5,              # Higher noise
    "https://api.iglucose.com": 2.6,    # Highest noise
}

# Source reliability classifications
SOURCE_RELIABILITY = {
    "patient-upload": "excellent",
    "care-team-upload": "excellent", 
    "internal-questionnaire": "good",
    "initial-questionnaire": "good",
    "patient-device": "good",
    "https://connectivehealth.io": "moderate",
    "https://api.iglucose.com": "poor",
}

# Questionnaire sources (for special handling)
QUESTIONNAIRE_SOURCES = {
    'internal-questionnaire',
    'initial-questionnaire',
    'care-team-upload',
    'questionnaire'
}

# BMI categories
BMI_UNDERWEIGHT = 18.5
BMI_NORMAL = 25.0
BMI_OVERWEIGHT = 30.0
BMI_OBESE = 35.0

# BMI validation limits
BMI_IMPOSSIBLE_LOW = 10.0
BMI_IMPOSSIBLE_HIGH = 100.0
BMI_SUSPICIOUS_LOW = 13.0
BMI_SUSPICIOUS_HIGH = 60.0