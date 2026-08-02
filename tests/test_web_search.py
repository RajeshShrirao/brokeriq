from brokeriq.tools.web_search import web_search


async def test_web_search_uses_ddg(monkeypatch):
    import importlib

    ddgs_module = importlib.import_module("brokeriq.tools.web_search")
    captured = {}

    async def fake_ddgs(query, max_results):
        captured["query"] = query
        return [{"title": "Acme Inc", "url": "https://acme.example", "snippet": "Acme makes widgets"}]

    monkeypatch.setattr(ddgs_module, "_ddgs_search", fake_ddgs)
    results = await web_search("Acme Inc company info", max_results=3)
    assert captured["query"] == "Acme Inc company info"
    assert results[0]["title"] == "Acme Inc"
    assert len(results) == 1
