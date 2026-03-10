"""Autonomous overnight experiment loop."""

import logging
import random
import time
from datetime import datetime

import config
from gsm8k import Problem
from knowledge_graph import KnowledgeGraph
from reasoning_pipeline import solve_problem
from strategies import StrategyConfig, baseline_strategy, default_kg_strategy, mutate

log = logging.getLogger(__name__)


def _write_result_header():
    if not config.RESULTS_FILE.exists():
        config.RESULTS_FILE.write_text(
            "timestamp\tstrategy_id\taccuracy\tcorrect\ttotal\t"
            "kg_nodes\tkg_edges\ttime_sec\tdescription\n"
        )


def _append_result(strategy: StrategyConfig, accuracy: float, correct: int,
                   total: int, kg_stats: dict, elapsed: float):
    line = (
        f"{datetime.now().isoformat()}\t{strategy.id}\t{accuracy:.4f}\t"
        f"{correct}\t{total}\t{kg_stats.get('nodes', 0)}\t"
        f"{kg_stats.get('edges', 0)}\t{elapsed:.1f}\t{strategy.describe()}\n"
    )
    with open(config.RESULTS_FILE, "a") as f:
        f.write(line)


def evaluate(
    problems: list[Problem],
    strategy: StrategyConfig,
    kg: KnowledgeGraph | None = None,
) -> tuple[float, int, int]:
    correct = 0
    total = 0

    for i, problem in enumerate(problems):
        try:
            result = solve_problem(problem, strategy, kg)
            if result["correct"]:
                correct += 1
            total += 1

            if (i + 1) % 25 == 0:
                log.info(
                    "  [%d/%d] running accuracy: %.1f%%",
                    i + 1, len(problems), 100 * correct / total,
                )
        except Exception as e:
            log.error("Problem %s failed: %s", problem.id, e)
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def build_kg(
    problems: list[Problem],
    kg: KnowledgeGraph,
    strategy: StrategyConfig | None = None,
):
    if strategy is None:
        strategy = default_kg_strategy()

    log.info("Building KG with %d problems...", len(problems))
    for i, problem in enumerate(problems):
        try:
            solve_problem(problem, strategy, kg)
            if (i + 1) % 50 == 0:
                stats = kg.get_stats()
                log.info(
                    "  KG build [%d/%d] — %d nodes, %d edges",
                    i + 1, len(problems), stats["nodes"], stats["edges"],
                )
        except Exception as e:
            log.error("KG build problem %s failed: %s", problem.id, e)


def run_experiment_loop(
    dev_problems: list[Problem],
    train_problems: list[Problem],
    kg: KnowledgeGraph,
    max_hours: float = 8.0,
):
    _write_result_header()
    start_time = time.time()
    deadline = start_time + max_hours * 3600

    # Phase 1: Baseline (no KG)
    log.info("=" * 60)
    log.info("PHASE 1: Baseline evaluation (no KG)")
    log.info("=" * 60)

    bl_strategy = baseline_strategy()
    t0 = time.time()
    bl_accuracy, bl_correct, bl_total = evaluate(dev_problems, bl_strategy)
    bl_elapsed = time.time() - t0

    _append_result(bl_strategy, bl_accuracy, bl_correct, bl_total,
                   {"nodes": 0, "edges": 0}, bl_elapsed)
    log.info("Baseline: %.1f%% (%d/%d) in %.1fs",
             100 * bl_accuracy, bl_correct, bl_total, bl_elapsed)

    # Phase 2: KG Building
    log.info("=" * 60)
    log.info("PHASE 2: Building knowledge graph")
    log.info("=" * 60)

    kg_build_problems = train_problems[:config.KG_BUILD_BATCH]
    remaining_train = train_problems[config.KG_BUILD_BATCH:]
    random.shuffle(remaining_train)
    build_kg(kg_build_problems, kg)

    kg_stats = kg.get_stats()
    log.info("KG built: %d nodes, %d edges", kg_stats["nodes"], kg_stats["edges"])

    # Phase 3: Experiment loop
    log.info("=" * 60)
    log.info("PHASE 3: Experiment loop")
    log.info("=" * 60)

    best_strategy = default_kg_strategy()
    t0 = time.time()
    best_accuracy, best_correct, best_total = evaluate(dev_problems, best_strategy, kg)
    elapsed = time.time() - t0

    kg_stats = kg.get_stats()
    _append_result(best_strategy, best_accuracy, best_correct, best_total,
                   kg_stats, elapsed)
    log.info("Initial KG strategy: %.1f%% (%d/%d)",
             100 * best_accuracy, best_correct, best_total)

    experiment_num = 0
    kg_refresh_ptr = 0

    while time.time() < deadline:
        experiment_num += 1
        hours_left = (deadline - time.time()) / 3600
        log.info("-" * 60)
        log.info("Experiment %d (%.1fh remaining) | best=%.1f%% baseline=%.1f%%",
                 experiment_num, hours_left, 100 * best_accuracy, 100 * bl_accuracy)

        # Periodic KG refresh
        if experiment_num % config.KG_REFRESH_INTERVAL == 0 and remaining_train:
            refresh_end = min(kg_refresh_ptr + config.KG_REFRESH_SIZE, len(remaining_train))
            refresh_batch = remaining_train[kg_refresh_ptr:refresh_end]
            kg_refresh_ptr = refresh_end
            if refresh_batch:
                log.info("Refreshing KG with %d more problems...", len(refresh_batch))
                build_kg(refresh_batch, kg, best_strategy)

        # Mutate and evaluate
        candidate = mutate(best_strategy)
        log.info("Testing: %s", candidate.describe())

        t0 = time.time()
        accuracy, correct, total = evaluate(dev_problems, candidate, kg)
        elapsed = time.time() - t0

        kg_stats = kg.get_stats()
        _append_result(candidate, accuracy, correct, total, kg_stats, elapsed)

        if accuracy > best_accuracy:
            log.info(
                "NEW BEST: %.1f%% -> %.1f%% (%s)",
                100 * best_accuracy, 100 * accuracy, candidate.describe(),
            )
            best_strategy = candidate
            best_accuracy = accuracy
        else:
            log.info(
                "No improvement: %.1f%% vs best %.1f%%",
                100 * accuracy, 100 * best_accuracy,
            )

    # Summary
    elapsed_total = (time.time() - start_time) / 3600
    log.info("=" * 60)
    log.info("EXPERIMENT COMPLETE after %.1f hours, %d experiments", elapsed_total, experiment_num)
    log.info("Baseline: %.1f%%", 100 * bl_accuracy)
    log.info("Best KG:  %.1f%% (%s)", 100 * best_accuracy, best_strategy.describe())
    log.info("KG size:  %d nodes, %d edges", kg_stats["nodes"], kg_stats["edges"])
    log.info("Results:  %s", config.RESULTS_FILE)
    log.info("=" * 60)
