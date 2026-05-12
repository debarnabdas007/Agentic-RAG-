from backend.app.memory import MemoryManager


def test_semantic_memory_from_remember_phrase():
    m = MemoryManager()
    m.ingest_user_stated_facts("Remember that I want short answers.")
    ctx = m.get_system_prompt_context()
    assert "short answers" in ctx.lower()
