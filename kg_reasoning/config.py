"""Constants and default configuration."""

from pathlib import Path

# Ollama
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-r1:latest"
OLLAMA_TIMEOUT = 120  # seconds per problem

# Neo4j
NEO4J_URI = "bolt://localhost:7688"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "autoresearch"

# GSM8K (parquet via HuggingFace API)
GSM8K_CACHE_DIR = Path.home() / ".cache" / "autoresearch" / "gsm8k"
GSM8K_TRAIN_URL = "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/train/0.parquet"
GSM8K_TEST_URL = "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/test/0.parquet"

# Experiment
DEV_EVAL_SIZE = 200  # problems for evaluation
KG_BUILD_BATCH = 500  # initial KG building problems
KG_REFRESH_INTERVAL = 5  # run more KG-building every N experiments
KG_REFRESH_SIZE = 100  # problems per refresh

# Docker
DOCKER_COMPOSE_FILE = Path(__file__).resolve().parent.parent / "docker-compose.kg.yml"

# Results
RESULTS_DIR = Path(__file__).resolve().parent
RESULTS_FILE = RESULTS_DIR / "results.tsv"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.pkl"
