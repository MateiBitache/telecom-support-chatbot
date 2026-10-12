"""Chat with the bot in the terminal. Run with: python cli.py"""

from chatbot import build_bot


def main():
    bot = build_bot()
    mode = "Claude" if bot.llm else "offline (no API key, answers come straight from the knowledge base)"
    print(f"NetWave Support Assistant - mode: {mode}")
    print("Type your question, or 'quit' to exit.\n")

    history: list[dict] = []
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in {"quit", "exit"}:
            break
        reply = bot.answer(message, history)
        print(f"\nBot: {reply.answer}")
        if reply.sources:
            print(f"     (source: {', '.join(reply.sources)})")
        if reply.escalated:
            print("     [escalated to a human agent]")
        print()
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": reply.answer}]
        history = history[-10:]  # keep only the last 5 exchanges


if __name__ == "__main__":
    main()
