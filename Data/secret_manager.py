import os
from pathlib import Path
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file"""
    # Look for .env at the repository root
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)

def read_secret_from_env(key):
    """Read a secret value from environment variable"""
    load_env()
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Environment variable '{key}' is not set. Please check your .env file.")
    return value

def get_neo4j_credentials():
    """Get Neo4j connection credentials from environment variables"""
    return {
        "local_neo4j_url": read_secret_from_env("NEO4J_URI"),
        "local_neo4j_username": read_secret_from_env("NEO4J_USERNAME"),
        "local_neo4j_password": read_secret_from_env("NEO4J_PASSWORD"),
    }
