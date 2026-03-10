"""Download, parse, and score GSM8K math reasoning problems."""

import logging
import re
from dataclasses import dataclass

import httpx
import pyarrow.parquet as pq

import config

log = logging.getLogger(__name__)


@dataclass
class Problem:
    id: str
    question: str
    solution: str  # step-by-step
    answer: int  # final numeric answer


def _download_split(url: str, dest_name: str) -> list[dict]:
    """Download a parquet split if not cached, return list of dicts."""
    config.GSM8K_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = config.GSM8K_CACHE_DIR / dest_name

    if cache_path.exists():
        log.info("Loading cached %s", cache_path)
    else:
        log.info("Downloading %s ...", url)
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
        log.info("Saved to %s", cache_path)

    table = pq.read_table(cache_path)
    return table.to_pylist()


def _parse_answer(answer_str: str) -> int:
    """Extract the final numeric answer after ####."""
    match = re.search(r"####\s*(-?[\d,]+)", answer_str)
    if match:
        return int(match.group(1).replace(",", ""))
    raise ValueError(f"Cannot parse answer from: {answer_str!r}")


def load_train() -> list[Problem]:
    """Load GSM8K train split (7,473 problems)."""
    raw = _download_split(config.GSM8K_TRAIN_URL, "train.parquet")
    problems = []
    for i, rec in enumerate(raw):
        try:
            problems.append(Problem(
                id=f"train_{i}",
                question=rec["question"],
                solution=rec["answer"],
                answer=_parse_answer(rec["answer"]),
            ))
        except (ValueError, KeyError) as e:
            log.warning("Skipping train problem %d: %s", i, e)
    return problems


def load_test() -> list[Problem]:
    """Load GSM8K test split (1,319 problems)."""
    raw = _download_split(config.GSM8K_TEST_URL, "test.parquet")
    problems = []
    for i, rec in enumerate(raw):
        try:
            problems.append(Problem(
                id=f"test_{i}",
                question=rec["question"],
                solution=rec["answer"],
                answer=_parse_answer(rec["answer"]),
            ))
        except (ValueError, KeyError) as e:
            log.warning("Skipping test problem %d: %s", i, e)
    return problems


def extract_model_answer(response: str) -> int | None:
    """Extract numeric answer from model response.

    Tries: \\boxed{N}, #### N, then last number.
    """
    match = re.search(r"\\boxed\{(-?[\d,]+)\}", response)
    if match:
        return int(match.group(1).replace(",", ""))

    match = re.search(r"####\s*(-?[\d,]+)", response)
    if match:
        return int(match.group(1).replace(",", ""))

    numbers = re.findall(r"-?\d[\d,]*", response)
    if numbers:
        return int(numbers[-1].replace(",", ""))

    return None


def score(predicted: int | None, expected: int) -> bool:
    """Exact match scoring."""
    return predicted == expected
