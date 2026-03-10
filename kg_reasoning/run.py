#!/usr/bin/env python3
"""Entry point for the KG-augmented reasoning experiment.

Run from repo root:
    uv run --project kg_reasoning python kg_reasoning/run.py
"""

import logging
import os
import random
import signal
import subprocess
import sys
import time

# Add kg_reasoning/ to sys.path so plain imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from gsm8k import load_train
from knowledge_graph import KnowledgeGraph
from ollama_client import check_available
from experiment_loop import run_experiment_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.RESULTS_DIR / "experiment.log"),
    ],
)
log = logging.getLogger(__name__)

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.info("Received signal %d, shutting down gracefully...", signum)
    _shutdown = True


def _check_ollama():
    log.info("Checking ollama...")
    if not check_available():
        log.error(
            "ollama not available or model %s not found. "
            "Make sure ollama is running: ollama serve",
            config.OLLAMA_MODEL,
        )
        sys.exit(1)
    log.info("ollama OK — model %s available", config.OLLAMA_MODEL)


def _start_neo4j():
    log.info("Starting Neo4j container...")
    subprocess.run(
        ["docker", "compose", "-f", str(config.DOCKER_COMPOSE_FILE), "up", "-d"],
        check=True,
    )

    log.info("Waiting for Neo4j to be ready...")
    for attempt in range(30):
        try:
            kg = KnowledgeGraph()
            kg.get_stats()
            kg.close()
            log.info("Neo4j ready!")
            return
        except Exception:
            if attempt % 5 == 0:
                log.info("  still waiting... (attempt %d/30)", attempt + 1)
            time.sleep(2)

    log.error("Neo4j did not become ready in 60s")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("=" * 60)
    log.info("KG-Augmented Reasoning Experiment")
    log.info("=" * 60)

    _check_ollama()
    _start_neo4j()

    log.info("Loading GSM8K dataset...")
    train_all = load_train()
    log.info("Loaded %d train problems", len(train_all))

    random.seed(42)
    random.shuffle(train_all)
    dev_problems = train_all[:config.DEV_EVAL_SIZE]
    train_problems = train_all[config.DEV_EVAL_SIZE:]

    log.info("Dev eval: %d problems", len(dev_problems))
    log.info("KG building pool: %d problems", len(train_problems))

    with KnowledgeGraph() as kg:
        kg.init_schema()
        run_experiment_loop(dev_problems, train_problems, kg)


if __name__ == "__main__":
    main()
