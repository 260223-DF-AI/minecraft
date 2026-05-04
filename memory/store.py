"""
ResearchFlow — Cross-Thread Memory (Store Interface)

Manages user preferences and query history across threads
using the LangGraph Store interface with namespaces and scopes.
"""
from langgraph.store.memory import InMemoryStore



store  = InMemoryStore() # unga bunga global store

DEFAULT_PREFERENCES = {
    "verbosity": "high", # If you don't specify, I'm going to make it as detailed as possible (low, med, high)
    "trusted_sources": [], # no trusted sources initially
}


# return an InMemoryStore from langgraph, do thisonce and make a new namespace for each user...
# recommended to use a factory function instead of a global store for thread safety
# deprecated due to not wanting to mess with config
def init_store() -> InMemoryStore:
    """
    Initialize the Store for cross-thread memory.
    """
    return InMemoryStore() # apparently you pass in the config at RUNTIME, so a lot of the code here is going to assume it exists


def get_user_preferences(user_id: str) -> dict:
    """
    Retrieve stored preferences for a user from the Store.

    TODO:
    - Use the Store interface with namespace = ("users", user_id).
    - Return a dict of preferences (verbosity, trusted sources, etc.).
    - Return sensible defaults if no preferences exist.
    """
    user_namespace = ("users", user_id)
    user_preferences = store.get(user_namespace, "preferences") # correct syntax to access the user's preferences

    if not user_preferences:
        return DEFAULT_PREFERENCES.copy()
    else:
        return user_preferences

def save_user_preferences( user_id: str, preferences: dict) -> None:
    """
    Persist user preferences to the Store.

    TODO:
    - Write to the Store under the user's namespace.
    """

    namespace = ("users", user_id)
    merged_namespace = {**DEFAULT_PREFERENCES, **preferences} # merge the default preferences with the user's preferences
    store.put(namespace, "preferences", merged_namespace)
    


def get_query_history(user_id: str, limit: int = 5) -> list[str]:
    """
    Retrieve recent query history for dynamic few-shot prompting.

    TODO:
    - Read from the Store under a "history" scope.
    - Return the most recent `limit` queries.
    """
    namespace = ("users", user_id, "history")

    
    history_item = store.get(namespace, "queries")
    history = history_item.value if history_item else []

    if history is None:
        return []

    return history[-limit:]


def append_query(user_id: str, question: str) -> None:
    """
    Append a query to the user's history in the Store.

    TODO:
    - Write the new query to the Store.
    """
    namespace = ("users", user_id, "history")

    history_item = store.get(namespace, "queries") or []
    history = history_item.value if history_item else []

    history.append(question)

    store.put(namespace, "queries", history)



if __name__ == "__main__":
    user_id = "Steve"

    print("\n--- Testing Preferences ---")

    save_user_preferences(user_id, {
        "verbosity": "low",
        "trusted_sources": ["wikipedia", "arxiv"]
    })

    prefs = get_user_preferences(user_id)
    print("Preferences:", prefs)

    print("\n--- Testing Query History ---")

    append_query(user_id, "What is minecraft?")
    append_query(user_id, "Explain how to beat minecraft.")
    append_query(user_id, "How does one make a crafting table?")

    history = get_query_history(user_id, limit=5)
    print("History:", history)

    print("\n--- Testing Sliding Window ---")

    for i in range(30):
        append_query(user_id, f"query {i}")

    history = get_query_history(user_id, limit=5)
    print("Last 5 queries:", history)