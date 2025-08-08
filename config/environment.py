from pathlib import Path

# Home Directory
HOME_DIR = Path(Path(__file__).resolve()).parent.parent

# Subdirectories
DB_DIR = Path(HOME_DIR, "db")
MODEL_DIR = Path(HOME_DIR, "models")
ASSETS_DIR = Path(HOME_DIR, "assets")
LIB_DIR = Path(HOME_DIR, "lib")
CONFIG_DIR = Path(HOME_DIR, "config")
PAGES_DIR = Path(HOME_DIR, "pages")
LOG_DIR = Path(HOME_DIR, "logs")
SCRIPTS_DIR = Path(HOME_DIR, "scripts")
CSS_DIR = Path(HOME_DIR, "css")
DATA_DIR = Path(HOME_DIR, "data")

# Database and Data Paths
SQLITE_DB_PATH = Path(DB_DIR, "sqlite", "chat_with_data.db")
VECTOR_DB_DIR = Path(DB_DIR, "vector")
CHROMADB_DIR = Path(VECTOR_DB_DIR, "chromadb")

# Output Directories
OUTPUT_DIR = Path(HOME_DIR, "output")
IMAGE_OUTPUT_DIR = Path(OUTPUT_DIR, "image")
TEXT_OUTPUT_DIR = Path(OUTPUT_DIR, "text")
VIDEO_OUTPUT_DIR = Path(OUTPUT_DIR, "video")
AUDIO_OUTPUT_DIR = Path(OUTPUT_DIR, "audio")
DOCUMENT_OUTPUT_DIR = Path(OUTPUT_DIR, "document")
TEMP_DIR = Path(OUTPUT_DIR, "temp")