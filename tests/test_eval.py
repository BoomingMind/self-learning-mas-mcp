import os
from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI


class OllamaJudge(DeepEvalBaseLLM):
    """
    Custom judge model using local Ollama.

    DeepEval supports custom models via the DeepEvalBaseLLM interface.
    We wrap ChatOllama to provide synchronous and async generation.

    The judge runs at temperature=0.0 for consistency. The same answer
    evaluated twice should produce the same score.
    """

    def __init__(self):
        self.model_name = os.getenv("OPEN_OPENROUTER_MODEL", "thinkingmachines/inkling:free")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")

    def load_model(self):
        return ChatOpenAI(
        model=self.model_name,
        base_url=self.base_url,
        api_key=self.api_key,
        temperature=0.0,
    )

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"ollama/{self.model_name}"


def get_judge_model():
    """Return an OllamaJudge, or None if deepeval is not installed."""
    try:
        return OllamaJudge()
    except ImportError:
        return None

def run_explainer(topic_title: str, topic_description: str, session_id: str) -> str:
    """Run the Explainer agent and return its final explanation text."""
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

    # Extract the final response: last AIMessage with no tool_calls
    for msg in reversed(result.get("messages", [])):
        if (isinstance(msg, AIMessage) and msg.content
                and not getattr(msg, "tool_calls", None)):
            return msg.content
    return ""


@pytest.mark.eval
class TestExplainerQuality:

    FAITHFULNESS_THRESHOLD = 0.6
    RELEVANCY_THRESHOLD    = 0.6

    @pytest.fixture(autouse=True)
    def setup(self, closures_note_content):
        """Run the Explainer once, reuse the output across all tests in this class."""
        self.retrieval_context = [closures_note_content]
        self.explanation = run_explainer(
            topic_title="Closures Explained",
            topic_description="Understand how closures capture enclosing scope variables",
            session_id="eval-test-001",
        )
        if not self.explanation:
            pytest.skip("Explainer returned empty output. Check Ollama is running.")

    def test_explanation_is_faithful_to_notes(self):
        """
        The explanation should not hallucinate facts not in the source notes.

        FaithfulnessMetric asks the judge: is every claim in the output
        supported by the retrieval context (the notes)?
        A low score means the agent is making things up.
        """
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import FaithfulnessMetric

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

        print(f"\n[Faithfulness] Score: {metric.score:.3f}")
        if hasattr(metric, "reason"):
            print(f"[Faithfulness] Reason: {metric.reason}")

        assert metric.score >= self.FAITHFULNESS_THRESHOLD, (
            f"Faithfulness {metric.score:.3f} below {self.FAITHFULNESS_THRESHOLD}.\n"
            f"The explanation may contain hallucinated facts.\n"
            f"Reason: {getattr(metric, 'reason', 'not available')}"
        )

    def test_explanation_is_relevant_to_topic(self):
        """The explanation should address what was actually asked."""
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import AnswerRelevancyMetric

        judge = get_judge_model()
        if judge is None:
            pytest.skip("Could not initialise judge model")

        test_case = LLMTestCase(
            input="Explain Python closures",
            actual_output=self.explanation,
        )
        metric = AnswerRelevancyMetric(
            model=judge,
            threshold=self.RELEVANCY_THRESHOLD,
        )
        metric.measure(test_case)

        print(f"\n[Relevancy] Score: {metric.score:.3f}")

        assert metric.score >= self.RELEVANCY_THRESHOLD, (
            f"Relevancy {metric.score:.3f} below {self.RELEVANCY_THRESHOLD}.\n"
            f"The explanation may have wandered off-topic."
        )

@pytest.mark.eval
class TestGradingQuality:

    def test_correct_answer_scores_high(self):
        """A clearly correct answer should score >= 0.65."""
        from agents.quiz_generator import grade_answer

        result = grade_answer(
            question="What are the three requirements for a Python closure?",
            expected=(
                "A closure requires: 1) a nested inner function, "
                "2) the inner function references a variable from the enclosing scope, "
                "3) the enclosing function returns the inner function."
            ),
            student_answer=(
                "You need a nested function that uses variables from the outer "
                "function's scope, and the outer function has to return the inner function."
            ),
        )
        print(f"\n[GradeQuality] Correct answer: {result.get('score', 0):.2f}")
        assert result.get("score", 0) >= 0.65, (
            f"Correct answer scored too low: {result['score']:.2f}\n"
            f"Feedback: {result.get('feedback', '')}"
        )

    def test_wrong_answer_scores_low(self):
        """A clearly wrong answer should score <= 0.35."""
        from agents.quiz_generator import grade_answer

        result = grade_answer(
            question="What is a Python closure?",
            expected=(
                "A closure is a nested function that captures and remembers "
                "variables from its enclosing scope after the enclosing function returns."
            ),
            student_answer=(
                "A closure is a class that closes over its attributes "
                "and prevents external access to them."
            ),
        )
        print(f"\n[GradeQuality] Wrong answer: {result.get('score', 0):.2f}")
        assert result.get("score", 0) <= 0.35, (
            f"Wrong answer scored too high: {result['score']:.2f}\n"
            f"The grader may be too lenient."
        )

    def test_partial_answer_scores_middle(self):
        """A partially correct answer should score between 0.3 and 0.75."""
        from agents.quiz_generator import grade_answer

        result = grade_answer(
            question="What is late binding in closures and how do you fix it?",
            expected=(
                "Late binding means closures look up variable values at call time, "
                "not at definition time. Fix: use default argument values "
                "(lambda i=i: i instead of lambda: i)."
            ),
            student_answer=(
                "Late binding means the closure uses the variable's current value "
                "when called, not when defined."  # Knows what, not how to fix
            ),
        )
        score = result.get("score", 0)
        print(f"\n[GradeQuality] Partial answer: {score:.2f}")
        assert 0.3 <= score <= 0.75, (
            f"Partial answer should score 0.3 to 0.75, got {score:.2f}"
        )

@pytest.mark.eval
class TestProgressCoachQuality:

    COACHING_QUALITY_THRESHOLD = 0.6

    def test_coaching_message_is_encouraging_and_specific(self):
        """
        Coaching messages should be warm, specific, and actionable.

        GEval lets you write evaluation criteria in plain English.
        The judge scores the output 0.0 to 1.0 against those criteria.
        """
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
        from deepeval.metrics import GEval
        from agents.progress_coach import get_coaching_message

        judge = get_judge_model()
        if judge is None:
            pytest.skip("Could not initialise judge model")

        coaching = get_coaching_message(
            topic="Python Closures",
            score=0.67,
            weak_areas=["late binding", "nonlocal keyword"],
        )
        coaching_text = (
            f"Summary: {coaching.get('summary', '')}\n"
            f"Encouragement: {coaching.get('encouragement', '')}"
        )

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
                "3) Actionable. Gives the student a clear next step. "
                "4) Concise. 2 to 4 sentences total. "
                "A poor message is generic, vague, or condescending."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=self.COACHING_QUALITY_THRESHOLD,
        )
        metric.measure(test_case)

        print(f"\n[CoachingQuality] Score: {metric.score:.3f}")

        assert metric.score >= self.COACHING_QUALITY_THRESHOLD, (
            f"Coaching quality {metric.score:.3f} below threshold.\n"
            f"Message:\n{coaching_text}"
        )