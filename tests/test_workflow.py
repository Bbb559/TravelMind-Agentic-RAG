import unittest
import json
import time
import tempfile
from dataclasses import replace
from pathlib import Path

from _helpers import FakeLLM, test_settings
from travelmind.agents import GraphRAGAgent, SystemAgent
from travelmind.graphs import AgenticRAGWorkflow
from travelmind.schemas import RouteDecision


class TimeoutGlobalSearchAdapter:
    def readiness(self):
        return True, ""

    def search(self, query: str):
        raise TimeoutError("global_search_timeout")


class CountingGlobalSearchAdapter:
    def __init__(self, content: str):
        self.content = content
        self.search_calls = 0

    def readiness(self):
        return True, ""

    def search(self, query: str):
        self.search_calls += 1
        return {
            "content": self.content,
            "raw_preview": self.content,
            "source_path": "assets/graphrag_output/community_reports.parquet",
            "source_summary": [
                {
                    "section": "reports",
                    "row_count": 1,
                    "titles_or_ids": ["fake-report"],
                }
            ],
        }


class CountingLocalSearchAdapter:
    def __init__(self, content: str):
        self.content = content
        self.search_calls = 0

    def readiness(self):
        return True, ""

    def search(self, query: str):
        self.search_calls += 1
        return {
            "content": self.content,
            "raw_preview": self.content,
            "source_path": "assets/graphrag_output/lancedb",
            "source_summary": [
                {
                    "section": "sources",
                    "row_count": 2,
                    "titles_or_ids": ["source-1", "source-2"],
                }
            ],
        }


class SlowLocalSearchAdapter(CountingLocalSearchAdapter):
    def search(self, query: str):
        time.sleep(0.2)
        return super().search(query)


class WorkflowContractTests(unittest.TestCase):
    def assert_answer_shape(self, answer):
        data = answer.to_dict()
        for key in ["answer", "route", "confidence", "sources", "retrieved", "fallback_reason", "trace"]:
            self.assertIn(key, data)
        self.assertIsInstance(data["trace"], list)
        self.assertIsInstance(data["retrieved"], list)

    def test_naive_workflow_shape(self):
        answer = AgenticRAGWorkflow(test_settings()).run("大理到双廊怎么去？")
        self.assertEqual(answer.route, "naive_rag")
        self.assert_answer_shape(answer)

    def test_invalid_input_stops_before_router_retriever_and_llm(self):
        fake = FakeLLM(["不应调用"])

        answer = AgenticRAGWorkflow(
            test_settings(
                llm_enabled=True,
                system_agent_llm_router_enabled=True,
                llm_grade_enabled=True,
                llm_rewrite_enabled=True,
                llm_generate_enabled=True,
            ),
            fake,
        ).run(" ！！！ ")

        self.assertEqual(answer.route, "invalid_input")
        self.assertEqual(answer.answer, "请先输入具体旅游问题。")
        self.assertEqual(answer.fallback_reason, "invalid_input")
        self.assertEqual(answer.sources, [])
        self.assertEqual(answer.retrieved, [])
        self.assertEqual(fake.prompts, [])
        self.assertFalse(
            any(
                step.startswith(("system:route", "agent:", "retrieve:", "grade:", "rewrite:", "generate:"))
                for step in answer.trace
            )
        )

    def test_naive_uses_csv_or_faiss_metadata(self):
        answer = AgenticRAGWorkflow(test_settings()).run("成都有什么美食？")
        modes = {item.metadata.get("retrieval_mode") for item in answer.retrieved}
        self.assertTrue(modes <= {"csv", "faiss"})

    def test_naive_itinerary_uses_destination_and_intent_matched_evidence(self):
        answer = AgenticRAGWorkflow(test_settings()).run("成都都江堰适合怎么玩？")

        self.assertEqual(answer.route, "naive_rag")
        self.assertEqual(answer.fallback_reason, None)
        self.assertTrue(answer.retrieved)
        self.assertTrue(all("都江堰" in (item.title or "") for item in answer.retrieved))
        self.assertTrue(all(item.metadata.get("evidence_valid") is True for item in answer.retrieved))
        self.assertTrue(all("itinerary" in item.metadata.get("matched_intents", []) for item in answer.retrieved))
        self.assertIn("游玩时可以优先考虑", answer.answer)
        self.assertIn("交通方面", answer.answer)
        self.assertNotIn("推荐玩法", answer.answer)
        self.assertNotIn("交通衔接", answer.answer)
        self.assertNotIn("注意事项", answer.answer)
        self.assertNotIn("住宿推荐", answer.answer)
        self.assertNotIn("四姑娘山", answer.answer)

    def test_naive_half_day_template_uses_natural_paragraphs(self):
        answer = AgenticRAGWorkflow(test_settings()).run("大理双廊怎么安排半天游？")

        self.assertEqual(answer.route, "naive_rag")
        self.assertEqual(answer.execution_status["generation_mode"], "template")
        self.assertTrue(answer.answer.startswith("如果安排半天游，可以优先考虑："))
        self.assertGreaterEqual(len(answer.answer.split("\n\n")), 2)
        self.assertNotIn("推荐玩法\n", answer.answer)
        self.assertNotIn("交通衔接\n", answer.answer)
        self.assertNotIn("注意事项\n", answer.answer)

    def test_naive_one_day_template_reflects_requested_duration(self):
        answer = AgenticRAGWorkflow(test_settings()).run("成都都江堰一天怎么玩？")

        self.assertEqual(answer.route, "naive_rag")
        self.assertTrue(answer.answer.startswith("如果安排一天游，可以优先考虑："))

    def test_naive_unknown_destination_does_not_generate_from_unrelated_sources(self):
        answer = AgenticRAGWorkflow(test_settings()).run("新加坡圣淘沙怎么玩比较合适？")

        self.assertEqual(answer.route, "naive_rag")
        self.assertEqual(answer.fallback_reason, "destination_not_covered")
        self.assertEqual(answer.sources, [])
        self.assertEqual(answer.retrieved, [])
        self.assertIn("当前资料不足", answer.answer)
        self.assertNotIn("舟山", answer.answer)
        self.assertNotIn("福建", answer.answer)

    def test_graphrag_workflow_shape(self):
        answer = AgenticRAGWorkflow(test_settings()).run("对比西安和南京的人文景点")
        self.assertEqual(answer.route, "graphrag")
        self.assert_answer_shape(answer)

    def test_official_local_search_is_default_formal_graphrag_answer(self):
        adapter = CountingLocalSearchAdapter(
            "阳朔以漓江喀斯特山水见长，张家界以砂岩峰林见长，可按偏好选择。"
        )
        fake = FakeLLM(["不应执行二次生成"])

        answer = SystemAgent(
            test_settings(),
            fake,
            graphrag_local_adapter=adapter,
        ).run(
            "阳朔和张家界哪个更适合看山水风景？",
            allow_global_search=False,
        )

        self.assertEqual(adapter.search_calls, 1)
        self.assertEqual(answer.answer, adapter.content)
        self.assertEqual(answer.fallback_reason, None)
        self.assertEqual(
            answer.retrieved[0].metadata["retrieval_mode"],
            "graphrag_local_search",
        )
        self.assertTrue(answer.retrieved[0].metadata["source_summary"])
        self.assertIn("graphrag:official_local_called", answer.trace)
        self.assertIn("graphrag:official_local_succeeded", answer.trace)
        self.assertIn("generate:official_local_response", answer.trace)
        self.assertNotIn("grade:llm", " ".join(answer.trace))
        self.assertNotIn("generate:llm", answer.trace)
        self.assertEqual(fake.prompts, [])

    def test_final_answer_removes_internal_graphrag_references_only(self):
        adapter = CountingLocalSearchAdapter(
            "阳朔（建议2天）与张家界都适合山水游，预算约300元。"
            " [Data: Reports (43); Entities (1274, 1848); Relationships (12)]"
            " Sources(59)"
        )

        answer = SystemAgent(
            test_settings(),
            None,
            graphrag_local_adapter=adapter,
        ).run("阳朔和张家界哪个更适合看山水风景？")

        self.assertNotIn("[Data:", answer.answer)
        self.assertNotIn("Reports(", answer.answer)
        self.assertNotIn("Entities(", answer.answer)
        self.assertNotIn("Relationships(", answer.answer)
        self.assertNotIn("Sources(", answer.answer)
        self.assertIn("阳朔（建议2天）", answer.answer)
        self.assertIn("300元", answer.answer)

    def test_local_evidence_never_enters_formal_llm_answer_chain(self):
        settings = test_settings(
            llm_enabled=True,
            llm_grade_enabled=True,
            llm_generate_enabled=True,
            graphrag_global_search_enabled=False,
        )
        fake = FakeLLM(
            [
                json.dumps(
                    {
                        "results": [
                            {
                                "index": 0,
                                "grade": "pass",
                                "usable_for_answer": True,
                                "reason": "实体覆盖",
                            }
                        ]
                    }
                ),
                "不应生成的正式 GraphRAG 答案",
            ]
        )

        answer = AgenticRAGWorkflow(settings, fake).run(
            "对比西安和南京的人文景点",
            allow_global_search=False,
        )

        self.assertEqual(
            answer.answer,
            "官方 GraphRAG Local Search 未成功，以下仅为本地 GraphRAG 产物证据预览，不代表官方 Local Search 正式回答。",
        )
        self.assertEqual(answer.confidence, "low")
        self.assertEqual(answer.fallback_reason, "official_local_failed")
        self.assertIn("grade:skipped:evidence_preview_only", answer.trace)
        self.assertIn("generate:skipped:evidence_preview_only", answer.trace)
        self.assertNotIn("generate:llm", answer.trace)
        self.assertEqual(fake.prompts, [])
        self.assertEqual(
            answer.retrieved[0].metadata["answer_policy"],
            "evidence_preview_only",
        )

    def test_workflow_request_gate_is_forwarded_to_graphrag(self):
        workflow = AgenticRAGWorkflow(
            test_settings(graphrag_global_search_enabled=True)
        )

        answer = workflow.run(
            "对比西安和南京的人文景点",
            allow_global_search=False,
        )

        self.assertIn("graphrag:request_not_allowed", answer.trace)

    def test_authorized_global_search_has_priority_over_available_local_evidence(self):
        adapter = CountingGlobalSearchAdapter(
            "阳朔和张家界都拥有典型山水景观，可用于比较两地风景特点。"
        )
        settings = test_settings(
            llm_enabled=True,
            llm_grade_enabled=True,
            llm_generate_enabled=True,
            graphrag_llm_api_key="fake",
            graphrag_global_search_enabled=True,
        )
        fake = FakeLLM(
            [
                json.dumps(
                    {
                        "results": [
                            {
                                "index": 0,
                                "grade": "pass",
                                "usable_for_answer": True,
                                "reason": "覆盖两个目的地",
                            }
                        ]
                    }
                ),
                "阳朔与张家界的正式比较答案。",
            ]
        )

        answer = SystemAgent(
            settings,
            fake,
            graphrag_adapter=adapter,
        ).run(
            "阳朔和张家界哪个更适合看山水风景？",
            allow_global_search=True,
        )

        self.assertEqual(adapter.search_calls, 1)
        self.assertEqual(
            answer.retrieved[0].metadata["retrieval_mode"],
            "graphrag_global_search",
        )
        self.assertEqual(answer.answer, "阳朔与张家界的正式比较答案。")
        self.assertIn("graphrag:global_search_called", answer.trace)
        self.assertIn("graphrag:global_search_succeeded", answer.trace)

    def test_authorized_global_without_llm_generate_uses_complete_official_response(self):
        content = (
            "北京适合第一次带父母了解皇家建筑与都城历史，可围绕故宫、景山和天坛安排。"
            "南京更适合体验明城墙、民国建筑与江南文脉，可围绕中山陵、明孝陵和南京博物院安排。"
            "如果父母偏好结构清晰、代表性强的国家历史地标，北京更合适；"
            "如果偏好节奏舒缓、近现代史与江南文化并重，南京更合适。"
            "北京热门景点步行量较大，应提前预约并为父母预留休息时间。"
            "南京主要人文景点相对分散，可按钟山风景区与老城两片安排，减少往返。"
            " [Data: Reports (12); Entities (34); Relationships (56)]"
        )
        adapter = CountingGlobalSearchAdapter(content)

        answer = SystemAgent(
            test_settings(
                graphrag_llm_api_key="fake",
                graphrag_global_search_enabled=True,
            ),
            None,
            graphrag_adapter=adapter,
        ).run(
            "北京和南京哪个更适合第一次带父母看历史文化？",
            allow_global_search=True,
        )

        self.assertEqual(adapter.search_calls, 1)
        self.assertGreater(len(answer.answer), 160)
        self.assertIn("北京更合适", answer.answer)
        self.assertIn("南京更合适", answer.answer)
        self.assertNotIn("[Data:", answer.answer)
        self.assertNotIn("Entities(", answer.answer)
        self.assertIn("generate:official_global_response", answer.trace)
        self.assertNotIn("generate:template", answer.trace)
        self.assertEqual(
            answer.execution_status["generation_mode"],
            "official_response",
        )

    def test_direct_graphrag_agent_output_sanitizes_official_global_citations(self):
        adapter = CountingGlobalSearchAdapter(
            "阳朔适合漓江山水体验，张家界适合峰林徒步。"
            " [Data: Reports (127, 434); Entities (1261); Relationships (4225)]"
        )
        settings = test_settings(
            graphrag_llm_api_key="fake",
            graphrag_global_search_enabled=True,
        )
        answer = GraphRAGAgent(
            settings,
            None,
            allow_global_search=True,
            graphrag_adapter=adapter,
        ).run(
            "阳朔和张家界哪个更适合看山水风景？",
            RouteDecision(
                query="阳朔和张家界哪个更适合看山水风景？",
                route="graphrag",
                confidence="medium",
                reason="direct GraphRAG contract",
                query_type="comparison",
            ),
        )

        self.assertEqual(adapter.search_calls, 1)
        self.assertNotIn("[Data:", answer.answer)
        self.assertNotIn("Reports(", answer.answer)
        self.assertNotIn("Entities(", answer.answer)
        self.assertNotIn("Relationships(", answer.answer)
        self.assertIn("阳朔适合漓江山水体验", answer.answer)
        self.assertIn("张家界适合峰林徒步", answer.answer)

    def test_global_grade_rejection_does_not_repeat_paid_search(self):
        adapter = CountingGlobalSearchAdapter(
            "阳朔和张家界都拥有典型山水景观，可用于比较两地风景特点。"
        )
        settings = test_settings(
            llm_enabled=True,
            llm_grade_enabled=True,
            llm_generate_enabled=True,
            graphrag_llm_api_key="fake",
            graphrag_global_search_enabled=True,
        )
        fake = FakeLLM(
            [
                json.dumps(
                    {
                        "results": [
                            {
                                "index": 0,
                                "grade": "fail",
                                "usable_for_answer": False,
                                "reason": "拒绝生成",
                            }
                        ]
                    }
                )
            ]
        )

        answer = SystemAgent(
            settings,
            fake,
            graphrag_adapter=adapter,
        ).run(
            "阳朔和张家界哪个更适合看山水风景？",
            allow_global_search=True,
        )

        self.assertEqual(adapter.search_calls, 1)
        self.assertEqual(answer.fallback_reason, "graphrag_grade_rejected")
        self.assertIn("未通过回答质量门禁", answer.answer)
        self.assertNotEqual(answer.answer, adapter.content)
        self.assertNotIn("generate:official_global_response", answer.trace)
        self.assertNotIn("generate:llm", answer.trace)
        self.assertNotIn("rewrite:", " ".join(answer.trace))

    def test_failed_global_search_falls_back_to_preview_without_formal_generate(self):
        settings = test_settings(
            llm_enabled=True,
            llm_grade_enabled=True,
            llm_generate_enabled=True,
            graphrag_llm_api_key="fake",
            graphrag_global_search_enabled=True,
        )
        fake = FakeLLM(["不应调用"])

        answer = SystemAgent(
            settings,
            fake,
            graphrag_adapter=TimeoutGlobalSearchAdapter(),
        ).run(
            "对比西安和南京的人文景点",
            allow_global_search=True,
        )

        self.assertEqual(
            answer.retrieved[0].metadata["retrieval_mode"],
            "graphrag_local_evidence",
        )
        self.assertEqual(answer.fallback_reason, "official_local_failed")
        self.assertEqual(
            answer.retrieved[0].metadata["global_search_error"],
            "timeout",
        )
        self.assertIn("graphrag:global_search_called", answer.trace)
        self.assertNotIn("graphrag:global_search_succeeded", answer.trace)
        self.assertIn("grade:skipped:evidence_preview_only", answer.trace)
        self.assertIn("generate:skipped:evidence_preview_only", answer.trace)
        self.assertEqual(fake.prompts, [])
        self.assertNotIn("global_search_timeout", answer.answer)

    def test_graphrag_source_type(self):
        answer = AgenticRAGWorkflow(test_settings()).run("阳朔和张家界哪个更适合看山水风景？")
        self.assertTrue(all(item.source_type == "graphrag_index" for item in answer.retrieved))

    def test_multimodal_workflow_shape(self):
        answer = AgenticRAGWorkflow(test_settings()).run("香港迪士尼怎么玩？")
        self.assertEqual(answer.route, "multimodal_rag")
        self.assert_answer_shape(answer)

    def test_multimodal_source_type(self):
        answer = AgenticRAGWorkflow(test_settings()).run("澳门大三巴牌坊在哪里？")
        self.assertTrue(any(item.source_type == "pdf_markdown" for item in answer.retrieved))

    def test_multimodal_without_relevant_sources_does_not_generate_advice(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = replace(
                test_settings(
                    llm_enabled=True,
                    llm_generate_enabled=True,
                ),
                multimodal_markdown_dir=Path(tmp),
            )
            fake = FakeLLM(["不应调用"])
            answer = AgenticRAGWorkflow(settings, fake).run("香港迪士尼怎么玩？")

        self.assertEqual(answer.route, "multimodal_rag")
        self.assertEqual(answer.retrieved, [])
        self.assertEqual(answer.sources, [])
        self.assertEqual(answer.fallback_reason, "no_relevant_evidence")
        self.assertEqual(
            answer.answer,
            "离线资料中没有找到足够证据，无法生成具体旅游建议。",
        )
        self.assertNotIn("generate:llm", answer.trace)
        self.assertEqual(fake.prompts, [])

    def test_hybrid_workflow_shape(self):
        answer = AgenticRAGWorkflow(test_settings()).run("台湾和西安哪个更适合亲子游？")
        self.assertEqual(answer.route, "hybrid_rag")
        self.assert_answer_shape(answer)
        self.assertIn("agent:hybrid_aggregator:start", answer.trace)
        self.assertIn("agent:hybrid_aggregator:end", answer.trace)

    def test_hybrid_only_aggregates_valid_branch_evidence(self):
        answer = AgenticRAGWorkflow(test_settings()).run(
            "香港和成都哪个更适合亲子游？"
        )

        self.assertEqual(answer.route, "hybrid_rag")
        self.assertEqual(answer.fallback_reason, "hybrid_partial_fallback")
        self.assertTrue(answer.retrieved)
        self.assertTrue(all(item.metadata.get("evidence_valid") is True for item in answer.retrieved))
        self.assertFalse(any(item.source_type == "graphrag_index" for item in answer.retrieved))
        self.assertEqual(
            answer.hybrid_branch_status["graphrag"]["evidence_valid"],
            False,
        )
        self.assertEqual(
            answer.hybrid_branch_status["multimodal"]["evidence_valid"],
            True,
        )
        self.assertIn("仅基于已命中的", answer.answer)

    def test_hybrid_rejects_graph_branch_that_only_declares_missing_entity_data(self):
        answer = SystemAgent(
            test_settings(),
            None,
            graphrag_local_adapter=CountingLocalSearchAdapter(
                "目前没有关于澳门的任何资料，因此无法评估澳门。"
                "重庆有较完整的慢游与交通资料。"
            ),
        ).run("澳门和重庆哪个更适合带老人慢游？")

        self.assertEqual(answer.route, "hybrid_rag")
        self.assertEqual(answer.fallback_reason, "hybrid_partial_fallback")
        self.assertFalse(
            answer.hybrid_branch_status["graphrag"]["evidence_valid"]
        )
        self.assertTrue(
            answer.hybrid_branch_status["multimodal"]["evidence_valid"]
        )
        self.assertFalse(
            any(item.source_type == "graphrag_index" for item in answer.retrieved)
        )
        self.assertTrue(
            all(item.source_type == "pdf_markdown" for item in answer.retrieved)
        )
        self.assertIn("仅基于已命中的 Multimodal 离线资料", answer.answer)
        self.assertNotIn("已完成深度融合", answer.answer)

    def test_hybrid_does_not_claim_deep_fusion(self):
        answer = AgenticRAGWorkflow(test_settings()).run("香港和成都哪个更适合周末游？")
        self.assertEqual(answer.route, "hybrid_rag")
        self.assertIn("多源候选聚合", answer.answer)
        self.assertNotIn("已完成深度融合", answer.answer)

    def test_hybrid_keeps_local_graph_results_as_preview_only(self):
        answer = AgenticRAGWorkflow(test_settings()).run(
            "台湾和西安哪个更适合亲子游？"
        )

        self.assertFalse(
            any(item.source_type == "graphrag_index" for item in answer.retrieved)
        )
        self.assertFalse(answer.hybrid_branch_status["graphrag"]["evidence_valid"])
        self.assertTrue(
            set(answer.hybrid_branch_status["graphrag"]["retrieval_modes"])
            & {"graphrag_local_evidence", "graphrag_wrapper"}
        )
        self.assertIn("hybrid:graphrag_evidence_preview_only", answer.trace)

    def test_hybrid_timeout_keeps_multimodal_and_safe_graph_candidates(self):
        settings = test_settings(
            graphrag_llm_api_key="fake",
            graphrag_global_search_enabled=True,
        )
        answer = SystemAgent(
            settings,
            None,
            graphrag_adapter=TimeoutGlobalSearchAdapter(),
        ).run(
            "台湾和西安哪个更适合亲子游？",
            allow_global_search=True,
        )

        modes = {
            item.metadata.get("retrieval_mode")
            for item in answer.retrieved
        }
        self.assertEqual(answer.route, "hybrid_rag")
        self.assertIn("markdown_keyword", modes)
        self.assertFalse(
            modes & {"graphrag_local_evidence", "graphrag_wrapper"}
        )
        self.assertFalse(answer.hybrid_branch_status["graphrag"]["evidence_valid"])
        self.assertNotIn("graphrag_global_search", modes)
        self.assertTrue(any("global_search_failed:timeout" in step for step in answer.trace))

    def test_hybrid_branch_timeout_returns_completed_multimodal_candidates(self):
        settings = test_settings(hybrid_branch_timeout_seconds=0.05)
        started = time.perf_counter()

        answer = SystemAgent(
            settings,
            None,
            graphrag_local_adapter=SlowLocalSearchAdapter(
                "香港和成都都适合亲子游。"
            ),
        ).run("香港和成都哪个更适合亲子游？")

        elapsed = time.perf_counter() - started
        modes = {
            item.metadata.get("retrieval_mode")
            for item in answer.retrieved
        }
        self.assertLess(elapsed, 0.18)
        self.assertIn("markdown_keyword", modes)
        self.assertIn("hybrid:branch:graphrag:timeout", answer.trace)
        self.assertIn("hybrid:branch:multimodal:completed", answer.trace)
        self.assertIn("hybrid:partial_fallback", answer.trace)
        self.assertEqual(answer.fallback_reason, "hybrid_partial_fallback")

    def test_hybrid_without_any_completed_candidates_reports_no_usable_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = replace(
                test_settings(hybrid_branch_timeout_seconds=0.05),
                multimodal_markdown_dir=Path(tmp),
            )
            answer = SystemAgent(
                settings,
                None,
                graphrag_local_adapter=SlowLocalSearchAdapter(
                    "香港和成都都适合亲子游。"
                ),
            ).run("香港和成都哪个更适合亲子游？")

        self.assertEqual(answer.retrieved, [])
        self.assertIn("hybrid:branch:graphrag:timeout", answer.trace)
        self.assertIn("hybrid:branch:multimodal:completed", answer.trace)
        self.assertEqual(
            answer.fallback_reason,
            "hybrid_no_usable_results",
        )

    def test_fallback_workflow_shape(self):
        answer = AgenticRAGWorkflow(test_settings()).run("qwxjkp")
        self.assertEqual(answer.route, "invalid_input")
        self.assertEqual(answer.fallback_reason, "invalid_input")
        self.assert_answer_shape(answer)

    def test_fallback_has_no_sources(self):
        answer = AgenticRAGWorkflow(test_settings()).run("帮我写一段 Python 快排")
        self.assertEqual(answer.sources, [])
        self.assertEqual(answer.retrieved, [])

    def test_route_method_matches_workflow_route(self):
        workflow = AgenticRAGWorkflow(test_settings())
        query = "香港迪士尼怎么玩？"
        self.assertEqual(workflow.route(query).route, workflow.run(query).route)

    def test_workflow_keeps_old_trace_fragments(self):
        answer = AgenticRAGWorkflow(test_settings()).run("大理到双廊怎么去？")
        self.assertTrue(any(step.startswith("route:") for step in answer.trace))
        self.assertTrue(any(step.startswith("retrieve:") for step in answer.trace))
        self.assertTrue(any(step.startswith("grade:") for step in answer.trace))
        self.assertTrue(any(step.startswith("generate:") for step in answer.trace))

    def test_llm_generate_fake_changes_answer(self):
        settings = test_settings(llm_enabled=True, llm_generate_enabled=True)
        fake = FakeLLM(["fake generated answer"])
        answer = AgenticRAGWorkflow(settings, fake).run("大理到双廊怎么去？")
        self.assertEqual(answer.answer, "fake generated answer")
        self.assertIn("generate:llm", answer.trace)

    def test_llm_generate_empty_fallbacks(self):
        settings = test_settings(llm_enabled=True, llm_generate_enabled=True)
        answer = AgenticRAGWorkflow(settings, FakeLLM([""])).run("大理到双廊怎么去？")
        self.assertIn("generate:llm_fallback_template", answer.trace)

    def test_llm_router_fake_route_keeps_workflow_shape(self):
        settings = test_settings(llm_enabled=True, system_agent_llm_router_enabled=True)
        fake = FakeLLM(['{"route":"multimodal_rag","confidence":"high","reason":"","query_type":"","entities":[],"matched_terms":[]}'])
        answer = AgenticRAGWorkflow(settings, fake).run("香港迪士尼怎么玩？")
        self.assertEqual(answer.route, "multimodal_rag")
        self.assertIn("system:route_source:llm", answer.trace)

    def test_no_graphrag_trace_for_naive(self):
        answer = AgenticRAGWorkflow(test_settings()).run("成都有什么美食？")
        self.assertFalse(any(step.startswith("graphrag:") for step in answer.trace))

    def test_no_multimodal_trace_for_naive(self):
        answer = AgenticRAGWorkflow(test_settings()).run("成都有什么美食？")
        self.assertFalse(any(step.startswith("multimodal:") for step in answer.trace))

    def test_unsupported_upload_photo_is_not_multimodal(self):
        answer = AgenticRAGWorkflow(test_settings()).run("上传一张景区照片并询问适合的游玩路线")
        self.assertEqual(answer.route, "fallback")

    def test_unsupported_pdf_parse_is_not_multimodal(self):
        answer = AgenticRAGWorkflow(test_settings()).run("现在解析这个 PDF 里的图片和文字")
        self.assertEqual(answer.route, "fallback")
