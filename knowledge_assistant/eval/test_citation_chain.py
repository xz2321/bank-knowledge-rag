"""Deterministic regressions for page routing and citation integrity."""
import unittest
import sys
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_knowledge_base import classify_page, clean_routed_markdown, split_page_runs
from build_local_index import IMAGE_PLACEHOLDER, inline_image_notes, split_sidecar_page_text
from qa_assistant import validate_citations
from eval_rag import configured_judge, evaluate_item, load_gold, load_index, summarize, validate


class PageRoutingRegressionTests(unittest.TestCase):
    def test_mineru_runs_are_contiguous_and_bounded(self):
        self.assertEqual(
            split_page_runs([0, 1, 2, 5, 6, 7, 8], max_pages=2),
            [[0, 1], [2], [5, 6], [7, 8]],
        )

    def test_plain_equations_and_prose_stay_with_text_extractor(self):
        text = "会计等式 资产 = 负债 + 所有者权益。" * 30
        words = [(i * 10, row * 12, i * 10 + 8, row * 12 + 8, "词", row, row, i)
                 for row in range(8) for i in range(6)]
        routed, reasons = classify_page(text, words)
        self.assertFalse(routed)
        self.assertEqual(reasons, set())

    def test_strong_formula_image_and_repeated_table_are_routed(self):
        self.assertIn("formula", classify_page("∑ √ ∫" + "公式内容" * 40, [])[1])
        self.assertIn("image", classify_page("图表说明" * 30, [], 0.03)[1])
        words = []
        for row in range(5):
            for col, x in enumerate((10, 160)):
                words.append((x, row * 15, x + 35, row * 15 + 8, f"单元{row}{col}", row * 2 + col, row, 0))
        self.assertIn("table", classify_page("表格内容" * 30, words)[1])

    def test_page_markers_and_repeated_footer_are_not_body_text(self):
        source = (
            "<!-- PyMuPDF page 3 -->\n正文第一段\n"
            "重点可视化 · 内容保真 · 非学习信息已清理 001 / 100\n\n"
            "<!-- MinerU pages 4-5 -->\n公式 A = B\n002 / 100\n"
        )
        cleaned, page_map = clean_routed_markdown(source)
        self.assertNotIn("<!--", cleaned)
        self.assertNotIn("001 / 100", cleaned)
        self.assertNotIn("002 / 100", cleaned)
        self.assertNotIn("重点可视化", cleaned)
        self.assertEqual(cleaned.strip(), "正文第一段\n\n公式 A = B")
        self.assertEqual(
            [(item["page_start"], item["page_end"]) for item in page_map], [(3, 3), (4, 5)]
        )
        self.assertEqual(cleaned[page_map[0]["start"]:page_map[0]["end"]], "正文第一段")

    def test_sidecar_page_ranges_preserve_index_citation_pages(self):
        text = "第一页正文\n\n第二页正文"
        with tempfile.TemporaryDirectory() as temp:
            sidecar = Path(temp) / "document.pages.json"
            sidecar.write_text(json.dumps([
                {"start": 0, "end": 5, "page_start": 1, "page_end": 1},
                {"start": 7, "end": 12, "page_start": 2, "page_end": 2},
            ]), encoding="utf-8")
            blocks = split_sidecar_page_text(text, sidecar)
        self.assertEqual(blocks, [(1, 1, "第一页正文"), (2, 2, "第二页正文")])

    def test_image_text_has_no_generated_labels_or_empty_placeholders(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "image.md"
            target.write_text(f"正文\n{IMAGE_PLACEHOLDER}\n后文", encoding="utf-8")
            inline_image_notes(target, {"image.jpg": "图中表格：资产 | 负债"})
            text = target.read_text(encoding="utf-8")
            self.assertIn("图中表格：资产 | 负债", text)
            self.assertNotIn("【图 1", text)
            self.assertNotIn(IMAGE_PLACEHOLDER, text)
            target.write_text(f"正文\n{IMAGE_PLACEHOLDER}\n后文", encoding="utf-8")
            inline_image_notes(target, {})
            self.assertNotIn(IMAGE_PLACEHOLDER, target.read_text(encoding="utf-8"))


class CitationChainRegressionTests(unittest.TestCase):
    def citation(self, citation_id, text):
        return {"id": citation_id, "text": text}

    def test_invalid_unused_ids_are_removed_and_remaining_ids_reindexed(self):
        answer, citations, check = validate_citations(
            "资产负债率反映偿债能力。[3]", [self.citation(1, "未引用材料"), self.citation(3, "资产负债率反映企业偿债能力。")]
        )
        self.assertIn("[1]", answer)
        self.assertEqual([item["id"] for item in citations], [1])
        self.assertEqual(check["mode"], "repaired")
        self.assertEqual(check["support_check"]["status"], "checked")


class EndToEndEvaluationRegressionTests(unittest.TestCase):
    def citation(self, citation_id, text):
        return {"id": citation_id, "text": text}

    @patch("eval_rag.judge_client")
    def test_answer_judge_is_not_initialized_by_default(self, judge_client):
        self.assertIsNone(configured_judge(False))
        judge_client.assert_not_called()

    def test_failed_closed_answer_is_counted_as_unsuccessful(self):
        item = {"id": "x1", "type": "answer", "question": "指标怎么算", "expect_any": ["权益乘数"]}
        reply = {
            "mode": "answer",
            "answer": "资料中未能为每项结论找到明确引用，暂不作答。",
            "citations": [],
            "retrieved": [{"content": "权益乘数=资产÷权益"}],
            "citation_check": {"support_review": "failed_closed_uncited", "support_check": {"status": "review"}},
        }
        row = evaluate_item(item, reply, 1.0)
        self.assertFalse(row["answered"])
        self.assertTrue(row["refused"])
        self.assertTrue(row["failed_closed"])
        summary = summarize([row])
        self.assertEqual(summary["answer_rate"], 0.0)
        self.assertEqual(summary["false_refusal_rate"], 1.0)
        self.assertEqual(summary["support_review_failed_closed"], 1)

    def test_citation_source_and_page_are_checked_against_expected_pdf(self):
        item = {"id": "x2", "type": "answer", "question": "问题", "expect_source": "财务管理"}
        reply = {
            "mode": "answer", "answer": "公式[1]",
            "citations": [{"id": 1, "title": "财务会计-财务管理.pdf", "page": 101, "snippet": "公式"}],
            "retrieved": [{"title": "财务会计-财务管理.pdf", "content": "公式"}],
        }
        row = evaluate_item(item, reply, 1.0, {"财务会计-财务管理.pdf": 100})
        self.assertTrue(row["source_hit"])
        self.assertEqual(row["invalid_citation_pages"][0]["reason"], "out_of_range")

    def test_expanded_gold_set_is_two_hundred_source_validated_questions(self):
        gold_path = Path(__file__).with_name("gold.json")
        if not gold_path.exists():
            self.skipTest("本地私有评测集 gold.json 未随公共仓库发布")
        gold = load_gold(gold_path)
        generated = [item for item in gold["items"] if item.get("generated_from") == "indexed_excerpt"]
        self.assertEqual(len(gold["items"]), 200)
        self.assertEqual(len(generated), 160)
        self.assertEqual(len({item["expect_source"] for item in generated}), 14)
        self.assertTrue(all(
            phrase in item.get("reference", "")
            for item in generated
            for phrase in item.get("expect_any", [])
        ))
        self.assertEqual(validate(gold, load_index()), [])

    def test_uncited_and_unrelated_claims_require_review(self):
        answer, _, check = validate_citations(
            "资产负债率反映偿债能力。[1]\n银行资本充足率越高，利润一定越高。",
            [self.citation(1, "资产负债率反映企业偿债能力。")],
        )
        self.assertTrue(answer)
        self.assertEqual(check["support_check"]["status"], "review")
        self.assertEqual(len(check["support_check"]["uncited_claims"]), 1)

    def test_cited_but_unrelated_claim_requires_review(self):
        _, _, check = validate_citations(
            "银行资本充足率越高，利润一定越高。[1]",
            [self.citation(1, "资产负债率反映企业长期偿债能力。")],
        )
        self.assertEqual(check["support_check"]["status"], "review")
        self.assertEqual(len(check["support_check"]["no_overlap_claims"]), 1)

    def test_newline_marker_and_semicolon_claim_are_kept_together(self):
        _, _, check = validate_citations(
            "资产负债率反映偿债能力；可用于分析长期偿债能力。\n[1]",
            [self.citation(1, "资产负债率可以反映企业偿债能力，可用于分析长期偿债能力。")],
        )
        self.assertEqual(check["support_check"]["status"], "checked")


if __name__ == "__main__":
    unittest.main()
