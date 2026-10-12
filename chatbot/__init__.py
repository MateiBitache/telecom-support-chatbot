from pathlib import Path

from .bot import BotReply, SupportBot
from .llm import llm_from_env
from .retriever import Retriever, load_chunks

DEFAULT_KB = Path(__file__).resolve().parent.parent / "knowledge_base"


def build_bot(kb_dir: str | Path = DEFAULT_KB, use_llm: bool = True) -> SupportBot:
    """Load the knowledge base and create a bot (online if an API key is set)."""
    retriever = Retriever(load_chunks(kb_dir))
    return SupportBot(retriever, llm=llm_from_env() if use_llm else None)


__all__ = ["BotReply", "SupportBot", "Retriever", "load_chunks", "build_bot"]
