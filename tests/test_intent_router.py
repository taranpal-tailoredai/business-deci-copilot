import pytest

from pipeline.intent_router import (
    IntentClassification,
    IntentRouter,
    RouteType,
)


class FakeClassifier:
    def __init__(self, route: str, toxic: bool = False) -> None:
        self._result = IntentClassification(
            route=route,
            reason=f"{route} route",
            toxic=toxic,
        )
        self.calls: list[str] = []

    def classify(self, query: str) -> IntentClassification:
        self.calls.append(query)
        return self._result


def handler(name: str):
    return lambda query: {"handler": name, "query": query}


@pytest.mark.parametrize(
    ("route", "expected", "expected_handlers"),
    [
        ("rag", RouteType.RAG, {"rag"}),
        ("sql", RouteType.SQL, {"sql"}),
        ("hybrid", RouteType.HYBRID, {"rag", "sql"}),
    ],
)
def test_routes_safe_questions_to_expected_handlers(
    route, expected, expected_handlers
):
    classifier = FakeClassifier(route)
    router = IntentRouter(classifier, handler("rag"), handler("sql"))

    decision = router.route("Show the relevant business information.")

    assert decision.route is expected
    assert set(decision.result) == expected_handlers


def test_blocks_toxic_question():
    router = IntentRouter(FakeClassifier("block", toxic=True))

    assert router.route("abusive request").route is RouteType.BLOCK


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM orders",
        "Update every seller record",
        "SELECT * FROM orders; DROP TABLE orders",
    ],
)
def test_blocks_crud_and_multi_statement_sql(query):
    classifier = FakeClassifier("sql")
    router = IntentRouter(classifier, handler("rag"), handler("sql"))

    decision = router.route(query)

    assert decision.route is RouteType.BLOCK
    assert classifier.calls == []
    assert decision.result == {}


def test_mixed_policy_and_crud_runs_only_rag():
    classifier = FakeClassifier("hybrid")
    router = IntentRouter(classifier, handler("rag"), handler("sql"))

    decision = router.route(
        "According to the refund policy, delete the related orders."
    )

    assert decision.route is RouteType.RAG
    assert decision.sql_blocked is True
    assert decision.result == {
        "rag": {
            "handler": "rag",
            "query": "According to the refund policy, delete the related orders.",
        }
    }
    assert classifier.calls == []
