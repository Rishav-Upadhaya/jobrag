
from langchain_core.messages import AIMessage, BaseMessage


def extract_user_query(messages: list[BaseMessage]) -> str:
    if not messages:
        raise ValueError("No messages found.")

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            continue
        if not message.content:
            continue
        return (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )

    raise ValueError("No human message with content found.")


def build_history(
    messages: list[BaseMessage],
    limit: int = 3,
) -> str:
    history_messages = messages[-(limit + 1):-1]

    if not history_messages:
        return "No previous conversation."

    formatted: list[str] = []

    for message in history_messages:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        formatted.append(f"{role}: {message.content}")

    return "\n".join(formatted)
