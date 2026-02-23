import os

def get_config():
    return {
        "api_key": os.getenv("API_KEY", ""),
        "group_id": os.getenv("GROUP_ID", ""),
    }
