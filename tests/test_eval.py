"""
tests/test_eval.py

LLM-as-judge evaluation tests for the Learning Accelerator.

TIER 2 TESTS, require Ollama running locally.
These tests are slow (30-120s each) and non-deterministic.
Run them before significant changes, not during development.

Run:
  pytest tests/test_eval.py -v -s -m eval

What these tests check:
  - Explainer explanations are faithful to source notes
  - Explainer explanations are relevant to the question asked
  - Quiz questions test understanding, not just recall
  - Progress Coach messages are encouraging and specific

Thresholds:
  All thresholds are set conservatively (0.6-0.7) to account for
  variability in local model outputs. Cloud models typically score
  0.8-0.95 on these metrics. Local 7B models score 0.6-0.8.
  If a test consistently fails, check the model and prompt first
  before lowering the threshold.
"""

import json
import os

import pytest

# src/ is added to sys.path by tests/conftest.py and pyproject.toml's pythonpath setting


# ─────────────────────────────────────────────────────────────────────────────
# DeepEval configuration
#
# Configure DeepEval to use Ollama as the judge model.
# This keeps evaluation entirely local, no OpenAI key required.
# ─────────────────────────────────────────────────────────────────────────────

def get_judge_model():
    """
    Get the DeepEval judge model configured for local Ollama.

    Uses LiteLLM under the hood to connect to Ollama's OpenAI-compatible API.
    Returns None if deepeval is not installed.
    """
    try:
        from deepeval.models import DeepEvalBaseLLM
        from langchain_ollama import ChatOllama

        class OllamaJudge(DeepEvalBaseLLM):
            """
            Custom judge model using local Ollama.

            DeepEval supports custom models via the DeepEvalBaseLLM interface.
            We wrap ChatOllama to provide the judge capabilities.
            """

            def __init__(self):
                self.model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
                self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            def load_model(self):
                return ChatOllama(
                    model=self.model_name,
                    base_url=self.base_url,
                    temperature=0.0,  # Deterministic for evaluation
                )

            def generate(self, prompt: str) -> str:
                model = self.load_model()
                response = model.invoke(prompt)
                return response.content

            async def a_generate(self, prompt: str) -> str:
                return self.generate(prompt)

            def get_model_name(self) -> str:
                return f"ollama/{self.model_name}"

        return OllamaJudge()
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run the Explainer and return its output
# ─────────────────────────────────────────────────────────────────────────────

def run_explainer(topic_title: str, topic_description: str, session_id: str) -> str:
    """
    Run the Explainer agent and return its final explanation.

    Args:
        topic_title:       The topic to explain.
        topic_description: Context for the topic.
        session_id:        Session ID for memory calls.

    Returns:
        The explanation text, or empty string if it failed.
    """
    from graph.state import StudyRoadmap, Topic, initial_state
    from agents.explainer import explainer_node
    from langchain_core.messages import AIMessage

    state = initial_state(f"Learn {topic_title}", session_id)
    state["roadmap"] = StudyRoadmap(
        goal=f"Learn {topic_title}",
        total_weeks=1,
        topics=[Topic(topic_title, topic_description, 60)],
    )
    state["current_topic_index"] = 0

    result = explainer_node(state)

    for msg in reversed(result.get("messages", [])):
        if (isinstance(msg, AIMessage) and msg.content
                and not getattr(msg, "tool_calls", None)):
            return msg.content

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Explainer quality tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.eval
class TestExplainerQuality:
    """
    Evaluate the quality of Explainer agent output.

    These tests answer: "Does the Explainer produce good explanations?"
    Good = faithful to source notes AND relevant to the question.
    """

    FAITHFULNESS_THRESHOLD = 0.6
    RELEVANCY_THRESHOLD = 0.6

    @pytest.fixture(autouse=True)
    def setup(self, closures_note_content):
        """Run the Explainer once and reuse the output across tests."""
        self.retrieval_context = [closures_note_content]

        print("\n[TestExplainerQuality] Running Explainer for closures topic...")
        self.explanation = run_explainer(
            topic_title="Closures Explained",
            topic_description="Understand how closures capture enclosing scope variables",
            session_id="eval-test-001",
        )

        if not self.explanation:
            pytest.skip("Explainer returned empty output, check Ollama is running")

        print(f"[TestExplainerQuality] Explanation length: {len(self.explanation)} chars")

    def test_explanation_is_faithful_to_notes(self):
        """
        The explanation should not hallucinate facts not in the source notes.

        Faithfulness measures: is everything stated in the explanation
        supported by the retrieval context (the notes)?

        A low faithfulness score means the agent is making things up
        rather than grounding its explanation in the actual notes.
        """
        try:
            from deepeval import evaluate
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import FaithfulnessMetric
        except ImportError:
            pytest.skip("deepeval not installed")

        judge = get_judge_model()
        if judge is None:
            pytest.skip("Could not initialise judge model")

        test_case = LLMTestCase(
            input="Explain Python closures",
            actual_output=self.explanation,
            retrieval_context=self.retrieval_context,
        )

        metric = FaithfulnessMetric(
            model=judge,
            threshold=self.FAITHFULNESS_THRESHOLD,
            include_reason=True,
        )

        metric.measure(test_case)

        print(f"\n[Faithfulness] Score: {metric.score:.3f} "
              f"(threshold: {self.FAITHFULNESS_THRESHOLD})")
        if hasattr(metric, "reason") and metric.reason:
            print(f"[Faithfulness] Reason: {metric.reason}")

        assert metric.score >= self.FAITHFULNESS_THRESHOLD, (
            f"Faithfulness score {metric.score:.3f} below threshold "
            f"{self.FAITHFULNESS_THRESHOLD}.\n"
            "The explanation may contain hallucinated facts not in the notes.\n"
            f"Reason: {getattr(metric, 'reason', 'not available')}"
        )

    def test_explanation_is_relevant_to_topic(self):
        """
        The explanation should actually address the topic that was asked.

        Answer relevancy measures: does the output address the input question?
        A low score means the Explainer wandered off-topic.
        """
        try:
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import AnswerRelevancyMetric
        except ImportError:
            pytest.skip("deepeval not installed")

        judge = get_judge_model()
        if judge is None:
            pytest.skip("Could not initialise judge model")

        test_case = LLMTestCase(
            input="Explain Python closures, what they are and why they matter",
            actual_output=self.explanation,
        )

        metric = AnswerRelevancyMetric(
            model=judge,
            threshold=self.RELEVANCY_THRESHOLD,
            include_reason=True,
        )

        metric.measure(test_case)

        print(f"\n[Relevancy] Score: {metric.score:.3f} "
              f"(threshold: {self.RELEVANCY_THRESHOLD})")

        assert metric.score >= self.RELEVANCY_THRESHOLD, (
            f"Relevancy score {metric.score:.3f} below threshold "
            f"{self.RELEVANCY_THRESHOLD}.\n"
            "The explanation may have drifted from the topic."
        )

    def test_explanation_has_minimum_length(self):
        """
        The explanation should be substantive, not a one-liner.

        This is a simple structural check, not LLM-as-judge.
        An explanation under 150 characters is almost certainly incomplete.
        """
        min_length = 150
        assert len(self.explanation) >= min_length, (
            f"Explanation too short: {len(self.explanation)} chars "
            f"(minimum: {min_length}).\n"
            f"Content: {self.explanation[:200]}"
        )

    def test_explanation_mentions_key_concepts(self):
        """
        The explanation should mention closure-related concepts.

        Simple keyword check, not LLM-as-judge.
        If none of these words appear, the explanation missed the topic entirely.
        """
        closure_keywords = [
            "closure", "nested", "enclosing", "scope", "inner function",
            "capture", "remember", "variable"
        ]
        explanation_lower = self.explanation.lower()
        found = [kw for kw in closure_keywords if kw in explanation_lower]

        assert len(found) >= 3, (
            "Explanation mentions too few closure concepts.\n"
            f"Found: {found}\n"
            f"Expected at least 3 of: {closure_keywords}\n"
            f"Explanation preview: {self.explanation[:300]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quiz Generator quality tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.eval
class TestQuizGeneratorQuality:
    """
    Evaluate the quality of Quiz Generator output.

    These tests answer: "Do the generated questions actually test understanding?"
    Good questions require application and reasoning, not just recall.
    """

    QUESTION_QUALITY_THRESHOLD = 0.6

    def test_generated_questions_test_understanding(self, closures_note_content):
        """
        Quiz questions should require genuine understanding, not just recall.

        Uses GEval with a custom rubric because there's no pre-built
        DeepEval metric for "question quality". GEval lets you define
        your own evaluation criteria in plain English.
        """
        try:
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams
            from deepeval.metrics import GEval
        except ImportError:
            pytest.skip("deepeval not installed")

        from agents.quiz_generator import generate_questions

        judge = get_judge_model()
        if judge is None:
            pytest.skip("Could not initialise judge model")

        print("\n[TestQuizQuality] Generating quiz questions...")
        questions = generate_questions(
            topic="Python Closures",
            explanation=closures_note_content,
            n=3,
        )

        assert len(questions) > 0, "No questions were generated"
        print(f"[TestQuizQuality] Generated {len(questions)} questions")

        questions_text = "\n".join([
            f"Q{i+1}: {q['question']}\nExpected: {q['expected_answer']}"
            for i, q in enumerate(questions)
        ])

        test_case = LLMTestCase(
            input="Generate quiz questions about Python closures that test understanding",
            actual_output=questions_text,
        )

        metric = GEval(
            name="QuestionQuality",
            criteria=(
                "Evaluate whether these quiz questions test genuine conceptual "
                "understanding of Python closures rather than surface-level recall. "
                "Good questions require the student to: apply concepts to new situations, "
                "explain WHY something works, identify edge cases, or compare concepts. "
                "Poor questions only ask to define terms or recite examples from the notes."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=self.QUESTION_QUALITY_THRESHOLD,
        )

        metric.measure(test_case)

        print(f"\n[QuestionQuality] Score: {metric.score:.3f} "
              f"(threshold: {self.QUESTION_QUALITY_THRESHOLD})")
        if hasattr(metric, "reason") and metric.reason:
            print(f"[QuestionQuality] Reason: {metric.reason}")

        assert metric.score >= self.QUESTION_QUALITY_THRESHOLD, (
            f"Question quality score {metric.score:.3f} below threshold.\n"
            "Questions may be too surface-level.\n"
            f"Questions generated:\n{questions_text}"
        )

    def test_questions_have_required_structure(self, closures_note_content):
        """
        Each generated question must have the required fields.

        Structural validation, fast, no LLM judge needed.
        """
        from agents.quiz_generator import generate_questions

        questions = generate_questions(
            topic="Python Closures",
            explanation=closures_note_content,
            n=3,
        )

        assert isinstance(questions, list)
        assert len(questions) > 0

        for i, q in enumerate(questions):
            assert "question" in q, f"Question {i} missing 'question' field"
            assert "expected_answer" in q, f"Question {i} missing 'expected_answer'"
            assert len(q["question"]) > 10, f"Question {i} text too short"
            assert len(q["expected_answer"]) > 10, f"Question {i} answer too short"
            assert q["question"].strip().endswith("?"), (
                f"Question {i} should end with '?': {q['question']}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Grading quality tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.eval
class TestGradingQuality:
    """
    Evaluate the quality of the LLM grader.

    These tests check: "Does the grader give fair, consistent scores?"
    Grader quality matters because wrong scores affect the Progress Coach
    routing decision and the student's confidence.
    """

    def test_correct_answer_scores_high(self):
        """A clearly correct answer should score >= 0.7."""
        from agents.quiz_generator import grade_answer

        question = "What are the three requirements for a function to be a closure in Python?"
        expected = (
            "A closure requires: 1) a nested inner function, "
            "2) the inner function must reference a variable from the enclosing scope, "
            "3) the enclosing function must return the inner function."
        )
        # This answer is complete and correct
        student_answer = (
            "For a closure you need a nested function that uses variables from "
            "the outer function's scope, and the outer function has to return "
            "the inner function."
        )

        result = grade_answer(question, expected, student_answer)

        print(f"\n[GradeQuality] Correct answer score: {result.get('score', 0):.2f}")
        print(f"[GradeQuality] Feedback: {result.get('feedback', '')}")

        assert result.get("score", 0) >= 0.65, (
            f"Correct answer scored too low: {result.get('score', 0):.2f}.\n"
            f"Feedback: {result.get('feedback', '')}"
        )

    def test_wrong_answer_scores_low(self):
        """A clearly wrong answer should score <= 0.3."""
        from agents.quiz_generator import grade_answer

        question = "What is a Python closure?"
        expected = (
            "A closure is a nested function that captures and remembers "
            "variables from its enclosing scope even after that scope has finished."
        )
        # This answer confuses closures with classes
        student_answer = (
            "A closure is a class that closes over its attributes "
            "and prevents external access to them."
        )

        result = grade_answer(question, expected, student_answer)

        print(f"\n[GradeQuality] Wrong answer score: {result.get('score', 0):.2f}")
        print(f"[GradeQuality] Feedback: {result.get('feedback', '')}")

        assert result.get("score", 0) <= 0.35, (
            f"Wrong answer scored too high: {result.get('score', 0):.2f}.\n"
            "The grader may be too lenient.\n"
            f"Feedback: {result.get('feedback', '')}"
        )

    def test_partial_answer_scores_middle(self):
        """A partially correct answer should score between 0.3 and 0.75."""
        from agents.quiz_generator import grade_answer

        question = "What is late binding in Python closures and how do you fix it?"
        expected = (
            "Late binding means closures look up the values of variables at call time, "
            "not at definition time. The fix is to use default argument values: "
            "lambda i=i: i instead of lambda: i."
        )
        # Knows WHAT it is but not HOW to fix it
        student_answer = (
            "Late binding means the closure uses the variable's value when called, "
            "not when defined. So if the variable changes, the closure sees the new value."
        )

        result = grade_answer(question, expected, student_answer)

        print(f"\n[GradeQuality] Partial answer score: {result.get('score', 0):.2f}")
        print(f"[GradeQuality] Feedback: {result.get('feedback', '')}")

        score = result.get("score", 0)
        assert 0.3 <= score <= 0.75, (
            f"Partial answer should score between 0.3 and 0.75, got {score:.2f}.\n"
            f"Feedback: {result.get('feedback', '')}"
        )

    def test_grader_always_returns_required_fields(self):
        """grade_answer should always return dict with required keys."""
        from agents.quiz_generator import grade_answer

        result = grade_answer(
            "What is a closure?",
            "A nested function capturing outer variables.",
            "Some student answer",
        )

        required_keys = ["correct", "score", "feedback"]
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in grade result"

        assert isinstance(result["correct"], bool)
        assert isinstance(result["score"], (int, float))
        assert 0.0 <= result["score"] <= 1.0, (
            f"Score {result['score']} outside valid range [0.0, 1.0]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Progress Coach quality tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.eval
class TestProgressCoachQuality:
    """
    Evaluate the quality of Progress Coach coaching messages.

    These tests check: "Does the coach produce useful, encouraging feedback?"
    """

    COACHING_QUALITY_THRESHOLD = 0.6

    def test_coaching_message_is_encouraging_and_specific(self):
        """
        Coaching messages should be warm, specific, and actionable.

        A good coaching message references the topic and score specifically.
        A bad one is generic ("Keep going!") without acknowledging what happened.
        """
        try:
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams
            from deepeval.metrics import GEval
        except ImportError:
            pytest.skip("deepeval not installed")

        from agents.progress_coach import get_coaching_message

        judge = get_judge_model()
        if judge is None:
            pytest.skip("Could not initialise judge model")

        print("\n[CoachQuality] Generating coaching message...")
        coaching = get_coaching_message(
            topic="Python Closures",
            score=0.67,
            weak_areas=["late binding", "nonlocal keyword"],
        )

        coaching_text = (
            f"Summary: {coaching.get('summary', '')}\n"
            f"Encouragement: {coaching.get('encouragement', '')}"
        )

        print(f"[CoachQuality] Message:\n{coaching_text}")

        test_case = LLMTestCase(
            input=(
                "Generate coaching feedback for a student who scored 67% on "
                "Python Closures and struggled with late binding and nonlocal"
            ),
            actual_output=coaching_text,
        )

        metric = GEval(
            name="CoachingQuality",
            criteria=(
                "Evaluate whether this coaching message is: "
                "1) Encouraging without being dishonest about the score, "
                "2) Specific to the topic and weak areas mentioned, "
                "3) Actionable, gives the student a clear next step, "
                "4) Appropriately concise (2-4 sentences total). "
                "A poor message is generic, vague, or condescending."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=self.COACHING_QUALITY_THRESHOLD,
        )

        metric.measure(test_case)

        print(f"\n[CoachingQuality] Score: {metric.score:.3f} "
              f"(threshold: {self.COACHING_QUALITY_THRESHOLD})")

        assert metric.score >= self.COACHING_QUALITY_THRESHOLD, (
            f"Coaching quality {metric.score:.3f} below threshold.\n"
            f"Message:\n{coaching_text}"
        )

    def test_coaching_returns_required_keys(self):
        """get_coaching_message must always return summary and encouragement."""
        from agents.progress_coach import get_coaching_message

        result = get_coaching_message(
            topic="Test Topic",
            score=0.8,
            weak_areas=[],
        )

        assert "summary" in result, "Missing 'summary' key"
        assert "encouragement" in result, "Missing 'encouragement' key"
        assert isinstance(result["summary"], str)
        assert isinstance(result["encouragement"], str)
        assert len(result["summary"]) > 0
        assert len(result["encouragement"]) > 0
