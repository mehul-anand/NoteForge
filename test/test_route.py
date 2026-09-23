from src.nodes.synthesis import ReviewSynthesizer, Intent, IntentOut
from src.state.state import State


class FakeClassifier:
    """with_structured_output returns self; invoke returns a preset IntentOut."""

    def __init__(self, intent: Intent, mixed: bool):
        self._intent = intent
        self._mixed = mixed

    def invoke(self, prompt):
        return IntentOut(intent=self._intent, mixed=self._mixed)


class RaisingClassifier:
    def invoke(self, prompt):
        raise RuntimeError("classifier failure")


class FakeLLM:
    def __init__(self, classifier):
        self._classifier = classifier

    def with_structured_output(self, schema):
        return self._classifier


def _route(intent: Intent, mixed: bool) -> str:
    syn = ReviewSynthesizer(FakeLLM(FakeClassifier(intent, mixed)))
    state = syn.route_intent(State(question="test"))
    return state.task_type


def test_mixed_question_routes_to_qa():
    # Commonality + factual/ordinal parts: must land on the hardened agent.
    assert _route(Intent.COMPARE, True) == "qa"
    assert _route(Intent.GAPS, True) == "qa"


def test_pure_intents_keep_their_task():
    assert _route(Intent.COMPARE, False) == "compare"
    assert _route(Intent.REVIEW, False) == "review"
    assert _route(Intent.GAPS, False) == "gaps"
    assert _route(Intent.QA, False) == "qa"


def test_classifier_failure_degrades_to_qa():
    syn = ReviewSynthesizer(FakeLLM(RaisingClassifier()))
    state = syn.route_intent(State(question="test"))
    assert state.task_type == "qa"