"""Thin wrapper around the Claude API used to generate natural answers."""

import os


class ClaudeLLM:
    """Generates an answer with Claude. Needs the ANTHROPIC_API_KEY environment variable."""

    def __init__(self, model: str | None = None, max_tokens: int = 400):
        import anthropic  # imported here so offline mode works without the package configured

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("CHATBOT_MODEL", "claude-haiku-5-5")
        self.max_tokens = max_tokens

    def generate(self, system: str, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()


def llm_from_env():
    """Return a ClaudeLLM if an API key is set, otherwise None (offline mode)."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return ClaudeLLM()
    return None
