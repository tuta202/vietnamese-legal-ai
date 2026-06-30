"""
Unit tests for the generator module.
All tests run OFFLINE — no vLLM server required.
"""
from __future__ import annotations

import re

import pytest

from legal_rag.generation.llm_generator import LegalGenerator, _DIEU_PATTERN
from legal_rag.generation.prompt_builder import PromptBuilder, _DISCLAIMER, _SYSTEM_PROMPT
from legal_rag.retrieval.config import RetrievalConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_config() -> RetrievalConfig:
    return RetrievalConfig()


@pytest.fixture()
def sample_articles() -> list[dict]:
    return [
        {
            "chunk_id": "aa" * 16,
            "law_id":   "04/2017/QH14",
            "law_type": "Luật",
            "law_name": "Hỗ trợ doanh nghiệp nhỏ và vừa",
            "dieu_number": "Điều 4",
            "dieu_title":  "Tiêu chí xác định doanh nghiệp nhỏ và vừa",
            "content": (
                "Doanh nghiệp nhỏ và vừa là doanh nghiệp đáp ứng tiêu chí về "
                "lao động, doanh thu và nguồn vốn."
            ),
            "relevant_doc_str":     "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
            "relevant_article_str": "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4",
        },
        {
            "chunk_id": "bb" * 16,
            "law_id":   "80/2021/NĐ-CP",
            "law_type": "Nghị định",
            "law_name": "Hỗ trợ DNNVV",
            "dieu_number": "Điều 5",
            "dieu_title":  "Đối tượng áp dụng",
            "content": (
                "Nghị định này áp dụng đối với doanh nghiệp nhỏ và vừa được thành "
                "lập và hoạt động theo quy định của pháp luật về doanh nghiệp."
            ),
            "relevant_doc_str":     "80/2021/NĐ-CP|Nghị định Hỗ trợ DNNVV",
            "relevant_article_str": "80/2021/NĐ-CP|Nghị định Hỗ trợ DNNVV|Điều 5",
        },
        {
            "chunk_id": "cc" * 16,
            "law_id":   "59/2020/QH14",
            "law_type": "Luật",
            "law_name": "Doanh nghiệp",
            "dieu_number": "Điều 7",
            "dieu_title":  "Quyền của doanh nghiệp",
            "content": (
                "Tự do kinh doanh ngành, nghề mà pháp luật không cấm; "
                "tự chủ kinh doanh và lựa chọn hình thức tổ chức kinh doanh."
            ),
            "relevant_doc_str":     "59/2020/QH14|Luật Doanh nghiệp",
            "relevant_article_str": "59/2020/QH14|Luật Doanh nghiệp|Điều 7",
        },
    ]


# ---------------------------------------------------------------------------
# PromptBuilder tests
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_returns_system_and_user_keys(self, sample_articles):
        pb = PromptBuilder()
        result = pb.build_prompt("Câu hỏi mẫu?", sample_articles)
        assert "system" in result
        assert "user" in result

    def test_system_prompt_contains_citation_instruction(self, sample_articles):
        pb = PromptBuilder()
        result = pb.build_prompt("Câu hỏi mẫu?", sample_articles)
        system = result["system"]
        assert "trích dẫn" in system.lower() or "Điều" in system
        assert "exact legal basis" in system.lower()

    def test_system_prompt_contains_no_hallucinate_instruction(self, sample_articles):
        pb = PromptBuilder()
        result = pb.build_prompt("test", sample_articles)
        system = result["system"].lower()
        assert "do not invent" in system
        assert "use only the supplied articles" in system

    def test_system_prompt_prevents_cross_clause_consequence_transfer(self, sample_articles):
        system = PromptBuilder().build_prompt("test", sample_articles)["system"].lower()

        assert "separate rule" in system
        assert "same khoản/điểm" in system
        assert "neighboring khoản/điểm" in system
        assert "interest payment" in system
        assert "never turn \"khoản 2\" into \"điều 2\"" in system
        assert "keep penalty amounts and remedies separate" in system

    def test_user_prompt_contains_all_articles(self, sample_articles):
        pb = PromptBuilder()
        result = pb.build_prompt("Câu hỏi?", sample_articles)
        user = result["user"]
        for art in sample_articles:
            assert art["dieu_number"] in user
            assert art["dieu_title"] in user

    def test_user_prompt_contains_question(self, sample_articles):
        question = "Doanh nghiệp nhỏ cần bao nhiêu vốn?"
        pb = PromptBuilder()
        result = pb.build_prompt(question, sample_articles)
        assert question in result["user"]

    def test_user_prompt_article_format(self, sample_articles):
        pb = PromptBuilder()
        result = pb.build_prompt("q", sample_articles)
        user = result["user"]
        assert user.count("\nSource:") == len(sample_articles)
        for index, art in enumerate(sample_articles, start=1):
            assert f"[A{index}]" in user
            assert art["law_id"] in user
            assert art["law_name"] in user

    def test_prompt_builder_max_articles(self, sample_articles):
        """With 10 articles and max_articles=7, only 7 should appear."""
        arts_10 = sample_articles * 4  # 12 articles (unique chunk_ids not required here)
        pb = PromptBuilder()
        result = pb.build_prompt("q", arts_10, max_articles=7)
        user = result["user"]
        assert user.count("\nSource:") == 7
        assert "[A7]" in user
        assert "[A8]" not in user

    def test_content_truncated_with_head_and_tail(self, default_config):
        pb = PromptBuilder()
        long_art = {
            "chunk_id": "x" * 32,
            "law_id": "test",
            "law_type": "Luật",
            "law_name": "Test",
            "dieu_number": "Điều 1",
            "dieu_title": "Title",
            "content": "HEAD" + "X" * 3000 + "TAIL",
        }
        result = pb.build_prompt("q", [long_art])
        user = result["user"]
        assert "HEAD" in user
        assert "TAIL" in user
        assert "phần giữa được rút gọn" in user

    def test_default_includes_all_articles(self, sample_articles):
        articles = []
        for index in range(12):
            article = dict(sample_articles[index % len(sample_articles)])
            article["dieu_title"] = f"Unique title {index}"
            articles.append(article)

        user = PromptBuilder().build_prompt("q", articles)["user"]

        assert user.count("\nSource:") == 12
        assert "Unique title 11" in user

    def test_total_content_budget_is_shared_across_all_articles(self):
        articles = [
            {
                "law_id": str(index),
                "law_type": "Luật",
                "law_name": "Test",
                "dieu_number": f"Điều {index}",
                "dieu_title": f"Title {index}",
                "content": f"HEAD-{index}-" + "X" * 500 + f"-TAIL-{index}",
            }
            for index in range(4)
        ]

        user = PromptBuilder().build_prompt(
            "q",
            articles,
            content_max_chars=500,
            total_content_max_chars=400,
        )["user"]

        for index in range(4):
            assert f"[A{index + 1}]" in user


# ---------------------------------------------------------------------------
# LegalGenerator tests
# ---------------------------------------------------------------------------

class TestExtractDieuMentions:
    """AC-5 coverage for extract_dieu_mentions()."""

    def _extract(self, text: str) -> list[str]:
        gen = LegalGenerator(RetrievalConfig(), mock=True)
        return gen.extract_dieu_mentions(text)

    def test_basic_two_mentions(self):
        result = self._extract("Theo Điều 4 và Điều 5 của Luật này.")
        assert result == ["Điều 4", "Điều 5"]

    def test_dieu_nay_excluded(self):
        result = self._extract("Điều này quy định về nghĩa vụ.")
        assert result == []

    def test_letter_suffix(self):
        result = self._extract("tại Điều 24a của Nghị định.")
        assert result == ["Điều 24a"]

    def test_three_mentions(self):
        result = self._extract("Điều 1, Điều 2, Điều 3 đều liên quan.")
        assert result == ["Điều 1", "Điều 2", "Điều 3"]

    def test_empty_string(self):
        result = self._extract("")
        assert result == []

    def test_deduplication(self):
        result = self._extract("Điều 5 và Điều 5 được áp dụng.")
        assert result == ["Điều 5"]   # deduplicated

    def test_dieu_khoan_dieu_kien_excluded(self):
        # "điều khoản", "điều kiện" have no digit → not matched
        result = self._extract("điều khoản này và điều kiện đó.")
        assert result == []

    def test_uppercase_dieu_only(self):
        # Pattern is case-sensitive for "Điều" (mixed case)
        # "điều 5" (all lowercase) should NOT be matched
        result = self._extract("điều 5 áp dụng.")
        assert result == []

    def test_mixed_valid_and_invalid(self):
        text = "Căn cứ Điều 10, Điều 11 và Điều này, điều khoản 3."
        result = self._extract(text)
        assert "Điều 10" in result
        assert "Điều 11" in result
        assert len(result) == 2   # "Điều này" and "điều khoản" not counted


class TestGeneratorMock:
    def test_mock_contains_dieu_pattern(self, default_config, sample_articles):
        gen = LegalGenerator(default_config, mock=True)
        answer = gen.generate("Câu hỏi?", sample_articles)
        mentions = gen.extract_dieu_mentions(answer)
        assert len(mentions) >= 1, f"No Điều X found in: {answer!r}"

    def test_mock_contains_disclaimer(self, default_config, sample_articles):
        gen = LegalGenerator(default_config, mock=True)
        answer = gen.generate("Câu hỏi?", sample_articles)
        assert "lưu ý" in answer.lower()
        assert "chuyên gia pháp lý" in answer.lower() or "thẩm quyền" in answer.lower()

    def test_mock_cites_first_article_dieu(self, default_config, sample_articles):
        gen = LegalGenerator(default_config, mock=True)
        answer = gen.generate("Câu hỏi?", sample_articles)
        first_dieu = sample_articles[0]["dieu_number"]
        assert first_dieu in answer

    def test_mock_empty_articles(self, default_config):
        gen = LegalGenerator(default_config, mock=True)
        answer = gen.generate("Câu hỏi?", [])
        assert "lưu ý" in answer.lower()

    def test_mock_no_client_init(self, default_config, sample_articles):
        gen = LegalGenerator(default_config, mock=True)
        gen.generate("q", sample_articles)
        assert gen._client is None


class TestGeneratorConfig:
    def test_config_generator_section_loaded(self):
        from pathlib import Path
        from legal_rag.retrieval.config import load_config
        cfg_path = Path(__file__).resolve().parent.parent / "config_gpu_clean.yaml"
        if not cfg_path.exists():
            pytest.skip("config_gpu_clean.yaml not found")
        cfg = load_config(cfg_path)
        assert cfg.generator.temperature == 0.0
        assert cfg.generator.max_tokens == 3000
        assert cfg.generator.max_articles == 0
        assert cfg.generator.content_max_chars == 2400
        assert cfg.generator.total_content_max_chars == 48000

    def test_default_generator_config(self):
        cfg = RetrievalConfig()
        assert cfg.generator.temperature == 0.3
        assert cfg.generator.max_tokens == 3000
        assert cfg.generator.top_p == 0.9


class TestNoCircularImports:
    def test_generator_imports_do_not_trigger_model_load(self, default_config):
        """Importing generator modules must not load any ML model."""
        gen = LegalGenerator(default_config, mock=True)
        assert gen._client is None
        pb = PromptBuilder()
        assert pb is not None

    def test_retrieval_config_unchanged_by_generator_import(self):
        """generator imports should not mutate retrieval.config state."""
        from legal_rag.retrieval.config import RetrievalConfig as RC
        cfg = RC()
        assert cfg.qdrant.host == "localhost"
        assert cfg.retrieval.rrf_k == 60
