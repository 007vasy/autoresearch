"""Three-phase reasoning pipeline: KG-retrieve -> solve -> extract+update."""

import json
import logging
import re

import ollama_client
from gsm8k import Problem, extract_model_answer, score
from knowledge_graph import KnowledgeGraph
from strategies import StrategyConfig, PROMPT_TEMPLATES

log = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
Analyze the following math problem and its solution attempt. Extract structured information.

Problem: {question}
Expected answer: {expected}
Model's answer: {predicted}
Correct: {correct}
Model's solution:
{response}

Return a JSON object with these fields:
- "concepts": list of 1-3 math concept names (e.g. "arithmetic", "rate problems", "fractions")
- "strategy": a short name for the approach used (e.g. "work-backwards", "unit-rate")
- "strategy_description": one-sentence description of the approach
- "mistakes": list of 0-2 mistake pattern names if incorrect (e.g. "off-by-one", "unit-confusion")

Return ONLY valid JSON, no other text.
"""


def _build_kg_context(kg_result: dict) -> str:
    parts = []

    if kg_result["strategies"]:
        parts.append("Effective strategies for similar problems:")
        for s in kg_result["strategies"]:
            rate = f" (success rate: {s['success_rate']:.0%})" if s.get("success_rate") else ""
            parts.append(f"  - {s['name']}: {s['description']}{rate}")
        parts.append("")

    if kg_result["mistakes"]:
        parts.append("Common mistakes to avoid:")
        for m in kg_result["mistakes"]:
            parts.append(f"  - {m['name']}: {m['description']}")
        parts.append("")

    if kg_result["similar_problems"]:
        parts.append("Similar problems you've seen:")
        for p in kg_result["similar_problems"]:
            q = p["question"][:150] + "..." if len(p["question"]) > 150 else p["question"]
            parts.append(f"  - {q} (answer: {p['answer']})")
        parts.append("")

    return "\n".join(parts) + "\n" if parts else ""


def _build_few_shot(kg_result: dict) -> str:
    parts = []
    for p in kg_result.get("similar_problems", []):
        parts.append(f"Problem: {p['question']}")
        parts.append(f"Answer: {p['answer']}")
        parts.append("")
    return "\n".join(parts) if parts else ""


def _quick_extract_concepts(question: str) -> list[str]:
    concepts = []
    patterns = {
        "arithmetic": r"\b(add|subtract|multiply|divide|sum|difference|product|quotient)\b",
        "fractions": r"\b(fraction|half|third|quarter|ratio)\b",
        "percentages": r"\b(percent|%|percentage|discount|markup|tax)\b",
        "rate_problems": r"\b(per hour|per minute|per day|speed|rate|mph|km/h)\b",
        "geometry": r"\b(area|perimeter|volume|circle|rectangle|triangle|square)\b",
        "money": r"\b(\$|dollar|cent|cost|price|pay|earn|spend|profit)\b",
        "time": r"\b(hour|minute|second|day|week|month|year|ago|later)\b",
        "comparison": r"\b(more than|less than|twice|triple|times as|greater|fewer)\b",
        "division_remainder": r"\b(remain|left over|evenly|divide equally)\b",
        "sequences": r"\b(each day|every week|pattern|sequence|consecutive)\b",
    }
    for concept, pattern in patterns.items():
        if re.search(pattern, question, re.IGNORECASE):
            concepts.append(concept)
    return concepts or ["general_math"]


def solve_problem(
    problem: Problem,
    strategy: StrategyConfig,
    kg: KnowledgeGraph | None = None,
) -> dict:
    """Run the 3-phase pipeline on a single problem."""
    # Phase 1: KG Retrieve
    kg_context = ""
    few_shot = ""
    concepts = _quick_extract_concepts(problem.question)

    if kg and (strategy.num_similar_problems > 0 or strategy.num_strategies > 0 or strategy.num_mistakes > 0):
        kg_result = kg.find_similar_context(
            concepts=concepts,
            num_problems=strategy.num_similar_problems,
            num_strategies=strategy.num_strategies,
            num_mistakes=strategy.num_mistakes,
        )
        kg_context = _build_kg_context(kg_result)
        few_shot = _build_few_shot(kg_result)

    # Phase 2: Solve
    template = PROMPT_TEMPLATES.get(strategy.prompt_template, PROMPT_TEMPLATES["standard"])
    prompt = template.format(
        question=problem.question,
        kg_context=kg_context,
        few_shot_examples=few_shot,
    )

    response = ollama_client.generate(
        prompt=prompt,
        temperature=strategy.temperature,
    )

    predicted = extract_model_answer(response)
    correct = score(predicted, problem.answer)

    result = {
        "problem_id": problem.id,
        "predicted": predicted,
        "expected": problem.answer,
        "correct": correct,
        "response": response,
        "concepts": concepts,
        "strategy_name": None,
        "mistakes": [],
    }

    # Phase 3: Extract + Update KG
    if kg:
        should_extract = (correct and strategy.extract_on_correct) or \
                         (not correct and strategy.extract_on_incorrect)

        if should_extract and strategy.use_llm_extraction:
            result = _llm_extract_and_update(problem, result, strategy, kg)
        elif should_extract:
            _regex_extract_and_update(problem, result, kg)
        else:
            kg.add_problem(problem.id, problem.question, problem.answer)
            for c in concepts:
                kg.add_concept(c)
                kg.link_problem_concept(problem.id, c)

    return result


def _llm_extract_and_update(
    problem: Problem, result: dict, strategy: StrategyConfig, kg: KnowledgeGraph,
) -> dict:
    try:
        extraction_prompt = EXTRACTION_PROMPT.format(
            question=problem.question,
            expected=problem.answer,
            predicted=result["predicted"],
            correct=result["correct"],
            response=result["response"][:2000],
        )
        raw = ollama_client.generate(extraction_prompt, temperature=0.0)

        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            log.warning("No JSON found in extraction response for %s", problem.id)
            _regex_extract_and_update(problem, result, kg)
            return result

        concepts = data.get("concepts", result["concepts"])[:5]
        strat_name = data.get("strategy", "unknown")
        strat_desc = data.get("strategy_description", "")
        mistakes = data.get("mistakes", [])[:3]

        kg.add_problem(problem.id, problem.question, problem.answer)

        for c in concepts:
            c = c.lower().replace(" ", "_")[:50]
            kg.add_concept(c)
            kg.link_problem_concept(problem.id, c)

        strat_id = strat_name.lower().replace(" ", "_")[:50]
        kg.add_strategy(strat_id, strat_name, strat_desc)
        kg.link_problem_strategy(problem.id, strat_id, result["correct"])
        kg.update_strategy_stats(strat_id, result["correct"])
        for c in concepts:
            c = c.lower().replace(" ", "_")[:50]
            kg.link_strategy_concept(strat_id, c)

        if not result["correct"]:
            for m in mistakes:
                m_name = m.lower().replace(" ", "_")[:50]
                kg.add_mistake(m_name)
                kg.link_problem_mistake(problem.id, m_name)

        result["concepts"] = concepts
        result["strategy_name"] = strat_name
        result["mistakes"] = mistakes

    except Exception as e:
        log.warning("LLM extraction failed for %s: %s, falling back to regex", problem.id, e)
        _regex_extract_and_update(problem, result, kg)

    return result


def _regex_extract_and_update(problem: Problem, result: dict, kg: KnowledgeGraph):
    concepts = result["concepts"]

    kg.add_problem(problem.id, problem.question, problem.answer)
    for c in concepts:
        kg.add_concept(c)
        kg.link_problem_concept(problem.id, c)

    strat_id = f"{'_'.join(concepts[:2])}_strategy"
    strat_name = f"{' + '.join(concepts[:2])} approach"
    kg.add_strategy(strat_id, strat_name, f"Standard approach for {', '.join(concepts)}")
    kg.link_problem_strategy(problem.id, strat_id, result["correct"])
    kg.update_strategy_stats(strat_id, result["correct"])
