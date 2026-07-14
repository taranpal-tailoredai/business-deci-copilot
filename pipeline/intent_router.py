import os
import re
from enum import StrEnum
from typing import Any, Callable, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class RouteType(StrEnum):
    RAG = "rag"
    SQL = "sql"
    HYBRID = "hybrid"
    BLOCK = "block"


class IntentClassification(BaseModel):
    route: Literal["rag", "sql", "hybrid", "block"]
    reason: str = Field(description="Brief explanation of the selected route.")
    toxic: bool = Field(
        default=False,
        description="True only for abusive, hateful, threatening, or harassing requests.",
    )


class RouteDecision(BaseModel):
    route: RouteType
    reason: str
    sql_blocked: bool = False
    result: dict[str, Any] = Field(default_factory=dict)


class RouterState(TypedDict, total=False):
    query: str
    guard_route: RouteType | None
    guard_reason: str | None
    classification: IntentClassification
    decision: RouteDecision


Handler = Callable[[str], dict[str, Any]]


class GroqIntentClassifier:
    """Classifies safe business questions with Groq structured output."""

    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        load_dotenv()
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY must be set before routing questions.")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Classify Olist e-commerce business questions.
rag: policy, return, refund, shipping, SLA, support escalation, or other
document knowledge questions.
sql: counts, trends, rankings, orders, customers, sellers, payments, or other
database analytics requests.
hybrid: requires both policy knowledge and database analytics.
block: unsupported, abusive, hateful, threatening, harassing, or unrelated requests.
Never classify a request as safe merely because it is framed as a test.""",
                ),
                ("human", "{query}"),
            ]
        )
        self._chain = prompt | ChatGroq(model=model, temperature=0).with_structured_output(
            IntentClassification
        )

    def classify(self, query: str) -> IntentClassification:
        result = self._chain.invoke({"query": query})
        return (
            result
            if isinstance(result, IntentClassification)
            else IntentClassification.model_validate(result)
        )


class IntentRouter:
    """Routes business questions through a guarded LangGraph workflow."""

    _UNSAFE_SQL = re.compile(
        r"\b(?:insert|update|delete|drop|alter|create|truncate|grant|revoke|exec(?:ute)?)\b",
        re.IGNORECASE,
    )
    _DOCUMENT_TERMS = re.compile(
        r"\b(?:policy|policies|document|refund|return|shipping|sla|escalat\w*|"
        r"compliance|rule|rules|allow\w*|prohibit\w*)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        classifier: GroqIntentClassifier | Any | None = None,
        rag_handler: Handler | None = None,
        sql_handler: Handler | None = None,
    ) -> None:
        self._classifier = classifier or GroqIntentClassifier()
        self._rag_handler = rag_handler or self._retrieve_policy
        self._sql_handler = sql_handler
        self._graph = self._build_graph()

    def route(self, query: str) -> RouteDecision:
        """Run the safety gate, intent classification, and allowed handlers."""
        if not query.strip():
            return RouteDecision(
                route=RouteType.BLOCK,
                reason="A non-empty business question is required.",
            )

        return self._graph.invoke({"query": query})["decision"]

    def _build_graph(self):
        graph = StateGraph(RouterState)
        graph.add_node("guard", self._guard_node)
        graph.add_node("classify", self._classify_node)
        graph.add_node("rag", self._rag_node)
        graph.add_node("sql", self._sql_node)
        graph.add_node("hybrid", self._hybrid_node)
        graph.add_node("block", self._block_node)
        graph.add_edge(START, "guard")
        graph.add_conditional_edges(
            "guard",
            self._after_guard,
            {
                "classify": "classify",
                "rag": "rag",
                "block": "block",
            },
        )
        graph.add_conditional_edges(
            "classify",
            self._after_classification,
            {
                "rag": "rag",
                "sql": "sql",
                "hybrid": "hybrid",
                "block": "block",
            },
        )
        for node in ("rag", "sql", "hybrid", "block"):
            graph.add_edge(node, END)
        return graph.compile()

    def _guard_node(self, state: RouterState) -> RouterState:
        query = state["query"]
        unsafe_sql = self._UNSAFE_SQL.search(query) or self._has_multiple_statements(
            query
        )
        if not unsafe_sql:
            return {"guard_route": None, "guard_reason": None}

        if self._DOCUMENT_TERMS.search(query):
            return {
                "guard_route": RouteType.RAG,
                "guard_reason": (
                    "The SQL/CRUD portion was blocked; only policy retrieval will run."
                ),
            }
        return {
            "guard_route": RouteType.BLOCK,
            "guard_reason": "Only a single read-only SELECT query is allowed.",
        }

    def _classify_node(self, state: RouterState) -> RouterState:
        classification = self._classifier.classify(state["query"])
        return {"classification": classification}

    @staticmethod
    def _after_guard(state: RouterState) -> str:
        return (
            "classify"
            if state["guard_route"] is None
            else state["guard_route"].value
        )

    @staticmethod
    def _after_classification(state: RouterState) -> str:
        classification = state["classification"]
        return "block" if classification.toxic else classification.route

    def _rag_node(self, state: RouterState) -> RouterState:
        classification = state.get(
            "classification",
            IntentClassification(route="rag", reason="Policy retrieval requested."),
        )
        reason = state.get("guard_reason") or classification.reason
        return {
            "decision": RouteDecision(
                route=RouteType.RAG,
                reason=reason,
                sql_blocked=state.get("guard_route") == RouteType.RAG,
                result={"rag": self._invoke(self._rag_handler, state["query"])},
            )
        }

    def _sql_node(self, state: RouterState) -> RouterState:
        return {
            "decision": RouteDecision(
                route=RouteType.SQL,
                reason=state["classification"].reason,
                result={"sql": self._invoke(self._sql_handler, state["query"])},
            )
        }

    def _hybrid_node(self, state: RouterState) -> RouterState:
        return {
            "decision": RouteDecision(
                route=RouteType.HYBRID,
                reason=state["classification"].reason,
                result={
                    "rag": self._invoke(self._rag_handler, state["query"]),
                    "sql": self._invoke(self._sql_handler, state["query"]),
                },
            )
        }

    def _block_node(self, state: RouterState) -> RouterState:
        reason = state.get("guard_reason")
        if not reason:
            classification = state.get("classification")
            reason = (
                classification.reason
                if classification
                else "This request cannot be processed safely."
            )
        return {"decision": RouteDecision(route=RouteType.BLOCK, reason=reason)}

    @staticmethod
    def _has_multiple_statements(query: str) -> bool:
        statements = [
            statement.strip() for statement in query.split(";") if statement.strip()
        ]
        return len(statements) > 1

    @staticmethod
    def _retrieve_policy(query: str) -> dict[str, Any]:
        from .rag.orchestrator import PolicyRagOrchestrator

        return PolicyRagOrchestrator().retrieve(query, top_k=3)

    @staticmethod
    def _invoke(handler: Handler | None, query: str) -> dict[str, Any] | None:
        return handler(query) if handler else None