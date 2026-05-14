"""Message extraction and formatting utilities."""

from langchain_core.messages import AIMessage, BaseMessage


def extract_user_query(messages: list[BaseMessage]) -> str:
    """
    Extract latest user query from messages.
    
    Args:
        messages: List of LangChain messages
        
    Returns:
        Content of the latest HumanMessage
        
    Raises:
        ValueError: If no messages or last message has no content
    """
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
    """
    Build lightweight conversational history for LLM context.
    
    Extracts the last N-1 messages (excludes current) and formats
    them as a readable conversation history.
    
    Args:
        messages: List of all messages
        limit: Number of past messages to include
        
    Returns:
        Formatted history string or "No previous conversation."
    """
    history_messages = messages[-(limit + 1):-1]

    if not history_messages:
        return "No previous conversation."

    formatted: list[str] = []

    for message in history_messages:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        formatted.append(f"{role}: {message.content}")

    return "\n".join(formatted)
