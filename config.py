"""
Configuration file for F1 Chatbot
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = RAW_DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data files
SETUPS_FILE = RAW_DATA_DIR / "F125-Setups.xlsx"
TECHNICAL_KB_FILE = PROCESSED_DATA_DIR / "technical_kb.json"
TRACK_GUIDES_FILE = PROCESSED_DATA_DIR / "track_guides.json"
FEEDBACK_RULES_FILE = PROCESSED_DATA_DIR / "feedback_rules.json"

# Model paths
INTENT_CLASSIFIER_DIR = MODELS_DIR / "intent_classifier"
SENTENCE_ENCODER_MODEL = "all-MiniLM-L6-v2"

# Intent classes
INTENTS = [
    "setup_request",           # "Give me setup for Monaco"
    "track_guide",            # "How to drive Silverstone"
    "track_tour",             # "Tell me about all tracks"
    "explain_component",      # "What is front wing"
    "feedback_understeer",    # "Car is understeering"
    "feedback_oversteer",     # "Car is oversteering"
    "feedback_tire_overheat", # "Tires overheating"
    "feedback_tire_wear",     # "Too much tire wear"
    "feedback_balance",       # "Car feels unstable"
    "feedback_bottoming",     # "Car bottoming out"
    "feedback_brake_lock",    # "Brakes locking"
    "general_question",       # Other queries
    "greeting",              # "Hi", "Hello"
    "thanks"                 # "Thank you"
]

# Track name mappings (for fuzzy matching)
TRACK_NAMES = [
    "Australia", "China", "Japan", "Bahrain", "Saudi Arabia",
    "Miami", "Imola", "Monaco", "Spain", "Canada",
    "Austria", "Silverstone", "Belgium Spa", "Hungary", "Netherlands",
    "Monza", "Baku", "Singapore", "COTA", "Mexican",
    "Brazil", "Las Vegas", "Qatar", "Abu dhabi"
]

# Aliases for track names
TRACK_ALIASES = {
    "melbourne": "Australia",
    "shanghai": "China",
    "suzuka": "Japan",
    "sakhir": "Bahrain",
    "jeddah": "Saudi Arabia",
    "barcelona": "Spain",
    "montreal": "Canada",
    "red bull ring": "Austria",
    "spielberg": "Austria",
    "spa": "Belgium Spa",
    "spa-francorchamps": "Belgium Spa",
    "hungaroring": "Hungary",
    "zandvoort": "Netherlands",
    "yas marina": "Abu dhabi",
    "abu dhabi": "Abu dhabi",
    "cota": "COTA",
    "austin": "COTA",
    "mexico city": "Mexican",
    "interlagos": "Brazil",
    "sao paulo": "Brazil",
    "vegas": "Las Vegas",
    "lusail": "Qatar"
}

# Setup parameter abbreviations
PARAM_ABBREV = {
    "FW": "front_wing",
    "RW": "rear_wing",
    "Diff on": "differential_on_throttle",
    "Diff off": "differential_off_throttle",
    "FC": "front_camber",
    "RC": "rear_camber",
    "FTO": "front_toe",
    "RTI": "rear_toe_in",
    "FS": "front_suspension",
    "RS": "rear_suspension",
    "FARB": "front_anti_roll_bar",
    "RARB": "rear_anti_roll_bar",
    "FRH": "front_ride_height",
    "RRH": "rear_ride_height",
    "Bias": "brake_bias",
    "Pressure": "brake_pressure",
    "FRT": "front_right_tyre",
    "FLT": "front_left_tyre",
    "RRT": "rear_right_tyre",
    "RLT": "rear_left_tyre"
}

# Model training parameters
INTENT_CLASSIFIER_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "warmup_steps": 100
}

# Response generation settings
USE_LLM_GENERATION = False  # Set to True to use Phi-3-mini, False for template-based
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
LLM_MAX_LENGTH = 512

# Hardware settings
DEVICE = "cuda"  # Will auto-detect in code
USE_4BIT_QUANTIZATION = True
