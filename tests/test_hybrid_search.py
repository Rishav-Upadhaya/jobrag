import asyncio

import pytest

from app.retrieval import hybrid_search as hs


def chunk(chunk_id, job_id, text="text"):
    return {"chunk_id": chunk_id, "job_id": job_id, "chunk_text": text}


def test_rrf_rewards_chunks_found_by_both_retrievers():
    vector = [chunk(1, "A"), chunk(2, "B"), chunk(3, "C")]
    keyword = [chunk(3, "C"), chunk(4, "D")]

    fused = hs.reciprocal_rank_fusion([vector, keyword], k=60)

    # 3 is found by both lists; 2 and 4 tie at rank 2 and keep insertion order.
    assert [c["chunk_id"] for c in fused] == [3, 1, 2, 4]
    assert fused[0]["fused_score"] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_keeps_each_chunk_once():
    fused = hs.reciprocal_rank_fusion([[chunk(1, "A")], [chunk(1, "A")]], k=60)
    assert len(fused) == 1


def test_rrf_handles_empty_lists():
    assert hs.reciprocal_rank_fusion([[], []]) == []


def test_deduplicate_caps_chunks_per_job_and_keeps_order():
    chunks = [chunk(i, "A") for i in range(4)] + [chunk(9, "B")]
    out = hs.deduplicate_by_job(chunks, max_chunks_per_job=2)
    assert [c["chunk_id"] for c in out] == [0, 1, 9]


def test_rerank_falls_back_to_original_order_when_reranker_fails(monkeypatch):
    class Broken:
        def rerank(self, **_):
            raise RuntimeError("jina down")

    monkeypatch.setattr(hs, "get_jina_reranker", lambda: Broken())
    chunks = [chunk(i, str(i)) for i in range(5)]
    assert hs.rerank_chunks(chunks, "q", top_k=3, threshold=0.3) == chunks[:3]


def test_rerank_reorders_by_reranker_indices(monkeypatch):
    class Fake:
        def rerank(self, **_):
            return [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.5}]

    monkeypatch.setattr(hs, "get_jina_reranker", lambda: Fake())
    out = hs.rerank_chunks([chunk(0, "A"), chunk(1, "B"), chunk(2, "C")], "q", top_k=2, threshold=0.3)
    assert [c["chunk_id"] for c in out] == [2, 0]
    assert out[0]["rerank_score"] == 0.9


class FakeEmbedder:
    def embed_query(self, _):
        return [0.0]


def run_hybrid(monkeypatch, vector, keyword, whitelist=None, final_top_k=5):
    calls = {}

    def fake_vector(conn, query_embedding, job_id_whitelist, top_k):
        calls["vector_whitelist"] = job_id_whitelist
        return vector

    def fake_keyword(conn, query_text, job_id_whitelist, top_k):
        calls["keyword_query"] = query_text
        calls["keyword_whitelist"] = job_id_whitelist
        return keyword

    monkeypatch.setattr(hs.repository, "get_matching_job_ids", lambda conn, f: whitelist)
    monkeypatch.setattr(hs.repository, "vector_search", fake_vector)
    monkeypatch.setattr(hs.repository, "keyword_search", fake_keyword)
    monkeypatch.setattr(hs, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(hs.settings, "JINA_API_KEY", "")  # skip the network reranker

    out = asyncio.run(hs.hybrid_search(None, "python backend", {}, final_top_k=final_top_k))
    return out, calls


def test_hybrid_search_surfaces_keyword_only_hits(monkeypatch):
    # An exact title match the embedding missed must still be retrievable.
    out, calls = run_hybrid(monkeypatch, vector=[chunk(1, "A")], keyword=[chunk(7, "Z")])
    assert {c["chunk_id"] for c in out} == {1, 7}
    assert calls["keyword_query"] == "python backend"


def test_hybrid_search_passes_metadata_whitelist_to_both_retrievers(monkeypatch):
    _, calls = run_hybrid(monkeypatch, vector=[], keyword=[], whitelist=["A", "B"])
    assert calls["vector_whitelist"] == ["A", "B"]
    assert calls["keyword_whitelist"] == ["A", "B"]


def test_hybrid_search_respects_final_top_k(monkeypatch):
    vector = [chunk(i, f"J{i}") for i in range(10)]
    out, _ = run_hybrid(monkeypatch, vector=vector, keyword=[], final_top_k=3)
    assert len(out) == 3


def test_keyword_rank_never_surfaces_as_relevance_score():
    # ts_rank_cd is unbounded; a keyword-only hit must report its fused score.
    from app.orchestration.langgraph.utils.prompt_builder import build_sources_from_chunks

    keyword_hit = {**chunk(5, "K"), "keyword_score": 3.0}
    fused = hs.reciprocal_rank_fusion([[], [keyword_hit]], k=60)
    [source] = build_sources_from_chunks(fused)
    assert source["relevance_score"] == pytest.approx(1 / 61)
