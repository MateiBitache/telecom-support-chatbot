"""The support bot: retrieval-augmented generation with escalation to a human agent."""

from dataclasses import dataclass, field

from .retriever import Retriever

SYSTEM_PROMPT = """You are the virtual support assistant of NetWave, a mobile operator.
Answer the customer's question using ONLY the information in the knowledge base excerpts below.
If the excerpts do not contain the answer, say you are not sure and offer to connect the customer to a human agent.
Be friendly, concise (at most 4 sentences) and give concrete steps when the customer has a problem.
Answer in the same language the customer writes in.

Knowledge base excerpts:
{context}"""

ESCALATION_MESSAGE = (
    "I'm not sure I can help with that. I'll connect you to a human agent, "
    "who will get back to you as soon as possible."
)

# Phrases that mean the customer explicitly wants a person.
HUMAN_REQUEST_PHRASES = (
    "human", "real person", "agent", "operator", "speak to someone",
    "talk to someone", "complaint", "cancel my contract",
)


@dataclass
class BotReply:
    answer: str
    sources: list[str] = field(default_factory=list)
    escalated: bool = False
    confidence: float = 0.0


class SupportBot:
    def __init__(self, retriever: Retriever, llm=None, min_score: float = 0.2, top_k: int = 3):
        self.retriever = retriever
        self.llm = llm  # None means offline mode: answer with the best matching section
        self.min_score = min_score
        self.top_k = top_k

    @staticmethod
    def wants_human(message: str) -> bool:
        text = message.lower()
        return any(phrase in text for phrase in HUMAN_REQUEST_PHRASES)

    def answer(self, message: str, history: list[dict] | None = None) -> BotReply:
        message = message.strip()
        if not message:
            return BotReply("Please type your question and I'll do my best to help.")

        if self.wants_human(message):
            return BotReply(ESCALATION_MESSAGE, escalated=True)

        results = self.retriever.search(message, k=self.top_k)
        best_chunk, best_score = results[0]
        if best_score < self.min_score:
            return BotReply(ESCALATION_MESSAGE, escalated=True, confidence=best_score)

        relevant = [(c, s) for c, s in results if s >= self.min_score]
        sources = [f"{c.source} > {c.title}" for c, _ in relevant]

        if self.llm is None:
            return BotReply(best_chunk.text, sources=sources[:1], confidence=best_score)

        context = "\n\n".join(f"[{c.source}] {c.full_text}" for c, _ in relevant)
        messages = list(history or []) + [{"role": "user", "content": message}]
        text = self.llm.generate(SYSTEM_PROMPT.format(context=context), messages)
        return BotReply(text, sources=sources, confidence=best_score)
