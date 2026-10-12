import pytest
from fastapi.testclient import TestClient

from chatbot import DEFAULT_KB, SupportBot, Retriever, load_chunks


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_chunks(DEFAULT_KB))


@pytest.fixture(scope="module")
def bot(retriever):
    return SupportBot(retriever, llm=None)


class FakeLLM:
    """Records what it receives instead of calling the real API."""

    def __init__(self):
        self.system = None
        self.messages = None

    def generate(self, system, messages):
        self.system, self.messages = system, messages
        return "fake answer"


def test_knowledge_base_is_split_into_sections():
    chunks = load_chunks(DEFAULT_KB)
    assert len(chunks) >= 15
    assert all(c.title and c.text for c in chunks)


@pytest.mark.parametrize(
    "question, expected_title",
    [
        ("how can I pay my bill", "How do I pay my bill?"),
        ("my mobile data stopped working", "My mobile data is not working"),
        ("how do I set up an esim on my phone", "How do I activate an eSIM?"),
        ("price of roaming in the USA", "How much does roaming outside the EU cost?"),
        ("someone stole my phone", "I lost my SIM card or my phone was stolen"),
    ],
)
def test_retriever_finds_the_right_section(retriever, question, expected_title):
    # The bot sends the top 3 sections to the LLM, so the right one must be among them.
    titles = [chunk.title for chunk, _ in retriever.search(question, k=3)]
    assert expected_title in titles


@pytest.mark.parametrize(
    "question, expected_title",
    [
        ("how can I pay my bill", "How do I pay my bill?"),
        ("my mobile data stopped working", "My mobile data is not working"),
        ("someone stole my phone", "I lost my SIM card or my phone was stolen"),
    ],
)
def test_best_match_for_clear_questions(retriever, question, expected_title):
    assert retriever.search(question, k=1)[0][0].title == expected_title


def test_offline_answer_comes_from_knowledge_base(bot):
    reply = bot.answer("When is my bill issued?")
    assert not reply.escalated
    assert "1st of every month" in reply.answer
    assert reply.sources == ["billing.md > When is my bill issued?"]


def test_unrelated_question_is_escalated(bot):
    reply = bot.answer("What is the best pizza recipe?")
    assert reply.escalated


def test_request_for_human_is_escalated(bot):
    assert bot.answer("I want to talk to a real person").escalated


def test_empty_message(bot):
    reply = bot.answer("   ")
    assert not reply.escalated and reply.answer


def test_llm_receives_context_and_history(retriever):
    llm = FakeLLM()
    bot = SupportBot(retriever, llm=llm)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    reply = bot.answer("How do I activate roaming?", history)
    assert reply.answer == "fake answer"
    assert "Roaming is active by default" in llm.system
    assert llm.messages[:2] == history
    assert llm.messages[-1] == {"role": "user", "content": "How do I activate roaming?"}


def test_rest_api(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import app

    client = TestClient(app.app)
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/chat", json={"message": "How do I change my plan?"})
    assert response.status_code == 200
    body = response.json()
    assert "Change plan" in body["answer"]
    assert body["escalated"] is False
    assert client.post("/chat", json={"message": ""}).status_code == 422
