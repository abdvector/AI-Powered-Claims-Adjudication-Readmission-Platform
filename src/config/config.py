from dotenv import load_dotenv
import os
load_dotenv()

def get_config_val(key: str, default=None):
    return os.getenv(key, default)

GEMINI_API_KEY = get_config_val("GEMINI_API_KEY")
SUPABASE_URL = get_config_val("SUPABASE_URL")
SUPABASE_KEY = get_config_val("SUPABASE_KEY")

HIGH_VALUE_MULTIPLIER = 2.0
STANDARD_MULTIPLIER = 1.0
STOP_WORD_MULTIPLIER = 0.0
IMPORTANT_WORD_LOW_CONF_THRESHOLD = 70.0

HUMMING_DISTANCE = 5
JACCARED_SIMILARITY = 0.85
AUTH_COOKIE_EXPIRY_DAYS = 1
