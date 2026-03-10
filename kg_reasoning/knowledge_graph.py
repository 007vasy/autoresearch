"""Neo4j knowledge graph: schema, CRUD, retrieval queries."""

import logging
from contextlib import contextmanager

from neo4j import GraphDatabase

import config

log = logging.getLogger(__name__)

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Problem) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Strategy) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:MistakePattern) REQUIRE m.name IS UNIQUE",
    "CREATE INDEX IF NOT EXISTS FOR (p:Problem) ON (p.topic)",
]


class KnowledgeGraph:
    def __init__(
        self,
        uri: str = config.NEO4J_URI,
        user: str = config.NEO4J_USER,
        password: str = config.NEO4J_PASSWORD,
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def _session(self):
        with self._driver.session() as session:
            yield session

    def init_schema(self):
        """Create constraints and indexes."""
        with self._session() as session:
            for stmt in SCHEMA_STATEMENTS:
                session.run(stmt)
        log.info("KG schema initialized")

    # ── CRUD ──────────────────────────────────────────────

    def add_problem(self, problem_id: str, question: str, answer: int,
                    difficulty: str = "unknown", topic: str = "math"):
        with self._session() as session:
            session.run(
                """MERGE (p:Problem {id: $id})
                   SET p.question = $question, p.answer = $answer,
                       p.difficulty = $difficulty, p.topic = $topic""",
                id=problem_id, question=question, answer=answer,
                difficulty=difficulty, topic=topic,
            )

    def add_concept(self, name: str, description: str = ""):
        with self._session() as session:
            session.run(
                """MERGE (c:Concept {name: $name})
                   SET c.description = $description""",
                name=name, description=description,
            )

    def add_strategy(self, strategy_id: str, name: str, description: str,
                     success_rate: float = 0.0, times_used: int = 0):
        with self._session() as session:
            session.run(
                """MERGE (s:Strategy {id: $id})
                   SET s.name = $name, s.description = $description,
                       s.success_rate = $success_rate, s.times_used = $times_used""",
                id=strategy_id, name=name, description=description,
                success_rate=success_rate, times_used=times_used,
            )

    def add_mistake(self, name: str, description: str = "", frequency: int = 1):
        with self._session() as session:
            session.run(
                """MERGE (m:MistakePattern {name: $name})
                   ON CREATE SET m.description = $description, m.frequency = $frequency
                   ON MATCH SET m.frequency = m.frequency + 1,
                       m.description = CASE WHEN $description <> '' THEN $description ELSE m.description END""",
                name=name, description=description, frequency=frequency,
            )

    # ── Relationships ─────────────────────────────────────

    def link_problem_concept(self, problem_id: str, concept_name: str):
        with self._session() as session:
            session.run(
                """MATCH (p:Problem {id: $pid})
                   MATCH (c:Concept {name: $cname})
                   MERGE (p)-[:REQUIRES_CONCEPT]->(c)""",
                pid=problem_id, cname=concept_name,
            )

    def link_problem_strategy(self, problem_id: str, strategy_id: str, correct: bool):
        with self._session() as session:
            session.run(
                """MATCH (p:Problem {id: $pid})
                   MATCH (s:Strategy {id: $sid})
                   MERGE (p)-[r:SOLVED_BY]->(s)
                   SET r.correct = $correct""",
                pid=problem_id, sid=strategy_id, correct=correct,
            )

    def link_problem_mistake(self, problem_id: str, mistake_name: str):
        with self._session() as session:
            session.run(
                """MATCH (p:Problem {id: $pid})
                   MATCH (m:MistakePattern {name: $mname})
                   MERGE (m)-[:OCCURS_IN]->(p)""",
                pid=problem_id, mname=mistake_name,
            )

    def link_strategy_concept(self, strategy_id: str, concept_name: str):
        with self._session() as session:
            session.run(
                """MATCH (s:Strategy {id: $sid})
                   MATCH (c:Concept {name: $cname})
                   MERGE (s)-[:USES_CONCEPT]->(c)""",
                sid=strategy_id, cname=concept_name,
            )

    def link_similar_problems(self, problem_id_1: str, problem_id_2: str):
        with self._session() as session:
            session.run(
                """MATCH (p1:Problem {id: $id1})
                   MATCH (p2:Problem {id: $id2})
                   MERGE (p1)-[:SIMILAR_TO]-(p2)""",
                id1=problem_id_1, id2=problem_id_2,
            )

    def update_strategy_stats(self, strategy_id: str, correct: bool):
        """Increment times_used and update rolling success_rate."""
        with self._session() as session:
            session.run(
                """MATCH (s:Strategy {id: $sid})
                   SET s.times_used = s.times_used + 1,
                       s.success_rate = CASE
                           WHEN s.times_used = 0 THEN toFloat($correct_int)
                           ELSE (s.success_rate * s.times_used + toFloat($correct_int)) / (s.times_used + 1)
                       END""",
                sid=strategy_id, correct_int=1 if correct else 0,
            )

    # ── Retrieval for prompt augmentation ─────────────────

    def find_similar_context(
        self, concepts: list[str], num_problems: int = 3,
        num_strategies: int = 3, num_mistakes: int = 3,
    ) -> dict:
        """Find similar problems, top strategies, and common mistakes."""
        with self._session() as session:
            problems_result = session.run(
                """UNWIND $concepts AS concept_name
                   MATCH (c:Concept {name: concept_name})<-[:REQUIRES_CONCEPT]-(p:Problem)
                   WITH p, count(DISTINCT c) AS overlap
                   ORDER BY overlap DESC
                   LIMIT $limit
                   RETURN p.id AS id, p.question AS question, p.answer AS answer""",
                concepts=concepts, limit=num_problems,
            )
            similar_problems = [dict(r) for r in problems_result]

            strategies_result = session.run(
                """UNWIND $concepts AS concept_name
                   MATCH (c:Concept {name: concept_name})<-[:USES_CONCEPT]-(s:Strategy)
                   WITH DISTINCT s
                   WHERE s.times_used > 0
                   ORDER BY s.success_rate DESC
                   LIMIT $limit
                   RETURN s.name AS name, s.description AS description,
                          s.success_rate AS success_rate""",
                concepts=concepts, limit=num_strategies,
            )
            strategies = [dict(r) for r in strategies_result]

            mistakes_result = session.run(
                """UNWIND $concepts AS concept_name
                   MATCH (c:Concept {name: concept_name})<-[:REQUIRES_CONCEPT]-(p:Problem)<-[:OCCURS_IN]-(m:MistakePattern)
                   WITH DISTINCT m
                   ORDER BY m.frequency DESC
                   LIMIT $limit
                   RETURN m.name AS name, m.description AS description""",
                concepts=concepts, limit=num_mistakes,
            )
            mistakes = [dict(r) for r in mistakes_result]

        return {
            "similar_problems": similar_problems,
            "strategies": strategies,
            "mistakes": mistakes,
        }

    def get_stats(self) -> dict:
        """Return node/edge counts."""
        with self._session() as session:
            nodes = session.run(
                "MATCH (n) RETURN count(n) AS count"
            ).single()["count"]
            edges = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS count"
            ).single()["count"]
        return {"nodes": nodes, "edges": edges}
