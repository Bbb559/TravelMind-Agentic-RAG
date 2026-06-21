import unittest

from _helpers import FakeLLM, fake_doc, test_settings
from travelmind.graphs import AgenticRAGWorkflow
from travelmind.llm.prompt_loader import load_prompt
from travelmind.runtime.rag_helpers import generate_answer_text, grade_results, parse_json_object, rewrite_query
from travelmind.schemas import RetrieverResult


class LLMOptionalTests(unittest.TestCase):
    def sample_result(self, mode="faiss", content="大理 双廊 交通"):
        return RetrieverResult(
            content=content,
            source_type="csv",
            source_path="assets/travel_guide.csv",
            title="大理",
            score=1.0,
            metadata={"retrieval_mode": mode},
            retriever_name="fake",
        )

    def test_prompt_loader_reads_generation_prompt(self):
        self.assertIn("sources", load_prompt("generation_prompt.md"))

    def test_prompt_loader_reads_agent_prompt(self):
        self.assertIn("NaiveTravelAgent", load_prompt("agents/naive_final_answer.md"))

    def test_prompt_loader_rejects_absolute_path(self):
        with self.assertRaises(ValueError):
            load_prompt("D:/x.md")

    def test_prompt_loader_rejects_parent_path(self):
        with self.assertRaises(ValueError):
            load_prompt("../.env")

    def test_parse_json_object_plain(self):
        self.assertEqual(parse_json_object('{"a":1}')["a"], 1)

    def test_parse_json_object_fenced(self):
        self.assertEqual(parse_json_object('```json\n{"a":2}\n```')["a"], 2)

    def test_grade_default_deterministic(self):
        trace = []
        usable = grade_results("大理", [self.sample_result()], trace, False, None)
        self.assertEqual(len(usable), 1)
        self.assertTrue(trace[0].startswith("grade:deterministic"))

    def test_grade_fake_llm_pass(self):
        trace = []
        fake = FakeLLM(['{"overall_grade":"pass","need_rewrite":false,"results":[{"index":0,"grade":"pass","reason":"ok","usable_for_answer":true}]}'])
        usable = grade_results("大理", [self.sample_result()], trace, True, fake)
        self.assertEqual(len(usable), 1)
        self.assertIn("grade:llm:pass", trace)

    def test_grade_fake_llm_invalid_json_fallbacks(self):
        trace = []
        usable = grade_results("大理", [self.sample_result()], trace, True, FakeLLM(["bad"]))
        self.assertEqual(len(usable), 1)
        self.assertTrue(any(step.startswith("grade:llm_fallback_deterministic") for step in trace))

    def test_grade_graphrag_wrapper_cannot_upgrade_to_pass(self):
        result = self.sample_result("graphrag_wrapper")
        result.source_type = "graphrag_index"
        trace = []
        fake = FakeLLM(['{"overall_grade":"pass","need_rewrite":false,"results":[{"index":0,"grade":"pass","reason":"ok","usable_for_answer":true}]}'])
        usable = grade_results("上海周末", [result], trace, True, fake)
        self.assertEqual(usable, [])
        self.assertEqual(result.metadata["grade"], "weak")

    def test_rewrite_default_deterministic(self):
        trace = []
        rewritten = rewrite_query("大理", trace, False, None)
        self.assertIn("大理", rewritten)
        self.assertTrue(trace[0].startswith("rewrite:deterministic"))

    def test_rewrite_fake_llm_success(self):
        trace = []
        fake = FakeLLM(['{"rewritten_query":"大理 双廊 交通","rewrite_strategy":"add_context","reason":"ok"}'])
        self.assertEqual(rewrite_query("大理", trace, True, fake), "大理 双廊 交通")
        self.assertTrue(trace[0].startswith("rewrite:llm:"))

    def test_rewrite_fake_llm_unsafe_fallbacks(self):
        trace = []
        fake = FakeLLM(['{"rewritten_query":"","rewrite_strategy":"no_rewrite","reason":"bad"}'])
        rewrite_query("大理", trace, True, fake)
        self.assertTrue(any("fallback" in step or "rejected" in step for step in trace))

    def test_generate_default_template(self):
        trace = []
        answer = generate_answer_text("大理", "naive_rag", "high", [self.sample_result()], trace, False, None)
        self.assertTrue(answer)
        self.assertIn("generate:template", trace)

    def test_multimodal_template_joins_unique_complete_sentences(self):
        repeated = "第一天先游览澳门历史城区，步行串联议事亭前地、大三巴和恋爱巷。"
        results = [
            self.sample_result(
                "markdown_vector",
                repeated
                + "下午转往路氹区域，安排龙环葡韵和官也街，并为长者预留休息时间。",
            ),
            self.sample_result(
                "markdown_vector",
                repeated
                + "第二天可根据体力选择澳门博物馆或海事博物馆，避免连续安排过多台阶。",
            ),
            self.sample_result(
                "markdown_vector",
                "酒店可优先选择接驳方便的区域，返程前留出充足的口岸通关时间。"
                "两天行程之间不要重复跨区往返，遇到炎热或降雨天气可适当减少户外步行。",
            ),
        ]
        for result in results:
            result.source_type = "pdf_markdown"
            result.metadata.update(
                {
                    "retrieval_mode": "markdown_vector",
                    "evidence_valid": True,
                }
            )

        trace = []
        answer = generate_answer_text(
            "澳门路氹和老城区两天怎么安排？",
            "multimodal_rag",
            "high",
            results,
            trace,
            False,
            None,
        )

        self.assertGreater(len(answer), 160)
        self.assertEqual(answer.count(repeated), 1)
        self.assertIn("第二天可根据体力选择", answer)
        self.assertIn("返程前留出充足的口岸通关时间。", answer)
        self.assertTrue(answer.endswith(("。", "！", "？", ".", "!", "?")))
        self.assertIn("generate:template", trace)

    def test_generate_fake_llm_success(self):
        trace = []
        answer = generate_answer_text("大理", "naive_rag", "high", [self.sample_result()], trace, True, FakeLLM(["LLM answer"]))
        self.assertEqual(answer, "LLM answer")
        self.assertIn("generate:llm", trace)

    def test_generate_fake_llm_error_fallbacks(self):
        trace = []
        answer = generate_answer_text("大理", "naive_rag", "high", [self.sample_result()], trace, True, FakeLLM(error=RuntimeError("boom")))
        self.assertTrue(answer)
        self.assertIn("generate:llm_fallback_template", trace)

    def test_workflow_default_does_not_call_fake_llm(self):
        fake = FakeLLM(["LLM answer"])
        answer = AgenticRAGWorkflow(test_settings(), fake).run("大理到双廊怎么去？")
        self.assertNotEqual(answer.answer, "LLM answer")
        self.assertEqual(fake.prompts, [])

    def test_workflow_generate_subswitch_controls_fake_llm(self):
        settings = test_settings(llm_enabled=True, llm_generate_enabled=False)
        fake = FakeLLM(["LLM answer"])
        answer = AgenticRAGWorkflow(settings, fake).run("大理到双廊怎么去？")
        self.assertNotEqual(answer.answer, "LLM answer")
        self.assertEqual(fake.prompts, [])

    def test_workflow_grade_subswitch_controls_fake_llm(self):
        settings = test_settings(llm_enabled=True, llm_grade_enabled=False)
        fake = FakeLLM(['{"overall_grade":"pass","need_rewrite":false,"results":[]}'])
        answer = AgenticRAGWorkflow(settings, fake).run("大理到双廊怎么去？")
        self.assertFalse(any(step.startswith("grade:llm:") for step in answer.trace))

    def test_workflow_rewrite_not_triggered_when_results_good(self):
        settings = test_settings(llm_enabled=True, llm_rewrite_enabled=True)
        answer = AgenticRAGWorkflow(settings, FakeLLM()).run("大理到双廊怎么去？")
        self.assertFalse(any(step.startswith("rewrite:llm:") for step in answer.trace))
