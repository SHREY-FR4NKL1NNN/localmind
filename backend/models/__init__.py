"""Local model clients for LocalMind.

Each client wraps a single Ollama-served model behind a uniform ``generate``
function so the router can call any backend interchangeably.
"""
