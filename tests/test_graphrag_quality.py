import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from _helpers import test_settings
from travelmind.graphrag import (
    GraphRAGGlobalSearchAdapter,
    GraphRAGOfficialLocalSearchAdapter,
)
from travelmind.graphrag.relevance import assess_graphrag_relevance
from travelmind.retrievers.graphrag_wrapper import GraphRAGSearchRetriever
from travelmind.runtime.rag_helpers import grade_results
from travelmind.schemas import RetrieverResult


class FakeGlobalSearchAdapter:
    def __init__(self, content: str):
        self.content = content
        self.readiness_calls = 0
        self.search_calls = 0

    def readiness(self):
        self.readiness_calls += 1
        return True, ""

    def search(self, query: str):
        self.search_calls += 1
        return {
            "content": self.content,
            "raw_preview": self.content[:200],
            "source_path": "assets/graphrag_output/community_reports.parquet",
            "source_summary": [
                {
                    "section": "reports",
                    "row_count": 1,
                    "titles_or_ids": ["fake-report"],
                }
            ],
        }


class FakeLocalSearchAdapter:
    def __init__(self, content: str, source_summary=None):
        self.content = content
        self.source_summary = (
            [
                {
                    "section": "sources",
                    "row_count": 1,
                    "titles_or_ids": ["fake-source"],
                }
            ]
            if source_summary is None
            else source_summary
        )
        self.search_calls = 0
        self.last_diagnostics = {
            "official_local_called": False,
            "official_local_succeeded": False,
            "official_local_error": None,
        }

    def readiness(self):
        return True, ""

    def search(self, query: str):
        self.search_calls += 1
        self.last_diagnostics.update(
            {
                "official_local_called": True,
                "official_local_succeeded": True,
            }
        )
        return {
            "content": self.content,
            "raw_preview": self.content[:200],
            "source_path": "assets/graphrag_output/lancedb",
            "source_summary": self.source_summary,
        }


class UnavailableLocalSearchAdapter:
    last_diagnostics = {
        "official_local_called": False,
        "official_local_succeeded": False,
        "official_local_error": "missing_key",
    }

    def readiness(self):
        return False, "missing_key"

    def search(self, query: str):
        raise AssertionError("ordinary tests must not call official local search")


def graph_settings(**overrides):
    values = {
        "graphrag_llm_api_key": "test-placeholder",
        "graphrag_llm_base_url": "https://example.invalid/v1",
        "graphrag_llm_chat_model": "chat-placeholder",
        "graphrag_llm_embedding_model": "embedding-placeholder",
    }
    values.update(overrides)
    return test_settings(**values)


class GraphRAGQualityTests(unittest.TestCase):
    def result(self, content: str, mode: str, score: float = 1.0):
        return RetrieverResult(
            content=content,
            source_type="graphrag_index",
            source_path="assets/graphrag_output/community_reports.parquet",
            title="GraphRAG evidence",
            score=score,
            metadata={"retrieval_mode": mode},
            retriever_name="fake",
        )

    def test_positive_score_alone_does_not_make_graphrag_result_relevant(self):
        assessment = assess_graphrag_relevance(
            "从上海出发有哪些景点适合周末去？",
            self.result("荔波大七孔和天生桥自然景观。", "graphrag_global_search"),
        )
        self.assertFalse(assessment.relevant)
        self.assertEqual(assessment.matched_entities, [])

    def test_matching_query_entity_makes_graphrag_result_relevant(self):
        assessment = assess_graphrag_relevance(
            "贵州有哪些适合自然风光游的地方？",
            self.result("贵州荔波大七孔适合自然风光游。", "graphrag_global_search"),
        )
        self.assertTrue(assessment.relevant)
        self.assertIn("贵州", assessment.matched_entities)

    def test_specific_attraction_alias_counts_as_full_coverage(self):
        assessment = assess_graphrag_relevance(
            "荔波小七孔怎么玩比较合适？",
            self.result("小七孔景区适合安排一日游。", "graphrag_local_search"),
        )

        self.assertTrue(assessment.relevant)
        self.assertEqual(assessment.query_entities, ["荔波小七孔"])
        self.assertEqual(assessment.matched_entities, ["荔波小七孔"])

    def test_parent_destination_does_not_cover_specific_attraction(self):
        assessment = assess_graphrag_relevance(
            "荔波小七孔怎么玩比较合适？",
            self.result("荔波自然风光丰富，适合山水旅行。", "graphrag_local_search"),
        )

        self.assertFalse(assessment.relevant)
        self.assertEqual(assessment.query_entities, ["荔波小七孔"])
        self.assertEqual(assessment.matched_entities, [])
        self.assertEqual(assessment.reason, "core_entity_missing")

    def test_missing_information_statement_does_not_cover_entity(self):
        assessment = assess_graphrag_relevance(
            "澳门和重庆哪个更适合带老人慢游？",
            self.result(
                "目前提供的数据中没有关于澳门的任何信息，因此无法评估澳门。"
                "重庆有较完整的交通、景点和慢游资料。",
                "graphrag_local_search",
            ),
        )

        self.assertFalse(assessment.relevant)
        self.assertEqual(assessment.query_entities, ["澳门", "重庆"])
        self.assertEqual(assessment.matched_entities, ["重庆"])
        self.assertEqual(assessment.reason, "core_entity_missing")

    def test_missing_detail_statement_with_parenthetical_alias_does_not_cover_entity(self):
        assessment = assess_graphrag_relevance(
            "大理、丽江、香格里拉适合怎么串成一条云南路线？",
            self.result(
                "目前提供的 data tables 中未包含丽江（Lijiang）的详细信息，"
                "因此只能围绕大理和香格里拉构建路线建议。",
                "graphrag_local_search",
            ),
        )

        self.assertFalse(assessment.relevant)
        self.assertEqual(
            assessment.query_entities,
            ["大理", "丽江", "香格里拉"],
        )
        self.assertEqual(assessment.matched_entities, ["大理", "香格里拉"])
        self.assertEqual(assessment.reason, "core_entity_missing")

    def test_broad_missing_entity_data_cannot_be_restored_by_meta_or_external_knowledge(self):
        assessment = assess_graphrag_relevance(
            "大理、丽江、香格里拉适合怎么串成一条云南路线？",
            self.result(
                "目前没有关于丽江的实质性信息，无法基于现有数据构建完整路线。"
                "后文只能指出丽江在常规旅游认知中的位置，并补充外部常识。"
                "大理和香格里拉之间有明确的旅游交通关系。",
                "graphrag_local_search",
            ),
        )

        self.assertFalse(assessment.relevant)
        self.assertEqual(assessment.matched_entities, ["大理", "香格里拉"])
        self.assertEqual(assessment.reason, "core_entity_missing")

    def test_positive_description_after_missing_statement_restores_coverage(self):
        assessment = assess_graphrag_relevance(
            "澳门和重庆哪个更适合带老人慢游？",
            self.result(
                "当前资料没有澳门住宿价格信息。"
                "澳门历史城区道路集中，可缩短单日步行距离。"
                "重庆山地坡道较多，带老人出行需要增加交通接驳。",
                "graphrag_global_search",
            ),
        )

        self.assertTrue(assessment.relevant)
        self.assertEqual(assessment.matched_entities, ["澳门", "重庆"])
        self.assertEqual(assessment.reason, "entity_coverage_sufficient")

    def test_travel_fact_negation_is_not_treated_as_missing_coverage(self):
        assessment = assess_graphrag_relevance(
            "澳门适合怎么安排公共交通？",
            self.result(
                "澳门没有地铁，但公交和酒店接驳车覆盖主要旅游区域。",
                "graphrag_local_search",
            ),
        )

        self.assertTrue(assessment.relevant)
        self.assertEqual(assessment.matched_entities, ["澳门"])

    def test_preview_weak_result_is_not_cautious_usable(self):
        result = self.result("贵州荔波候选内容", "community_reports_preview")
        result.metadata.update({"grade": "weak", "usable_for_answer": False})
        usable = grade_results("贵州自然风光", [result], [], False, None)
        self.assertEqual(usable, [])

    def test_local_evidence_requires_llm_pass_before_answering(self):
        result = self.result("西安和南京均有人文景点资料。", "graphrag_local_evidence")
        result.metadata.update(
            {
                "graphrag_relevance": True,
                "query_entities": ["西安", "南京"],
            }
        )
        usable = grade_results("对比西安和南京的人文景点", [result], [], False, None)
        self.assertEqual(usable, [])
        self.assertEqual(result.metadata["grade"], "weak")

    def test_local_evidence_always_declares_low_confidence_policy(self):
        result = GraphRAGSearchRetriever(
            test_settings(graphrag_llm_api_key="")
        ).retrieve("对比西安和南京的人文景点")[0]

        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_local_evidence")
        self.assertEqual(result.metadata["confidence_policy"], "low")

    def test_service_gate_disabled_never_touches_global_search_adapter(self):
        adapter = FakeGlobalSearchAdapter("不应被调用")
        retriever = GraphRAGSearchRetriever(
            test_settings(
                graphrag_llm_api_key="fake",
                graphrag_global_search_enabled=False,
            ),
            adapter=adapter,
            allow_global_search=True,
        )

        result = retriever.retrieve("对比西安和南京的人文景点")[0]

        self.assertEqual(adapter.readiness_calls, 0)
        self.assertEqual(adapter.search_calls, 0)
        self.assertEqual(result.metadata["fallback_reason"], "official_local_failed")
        self.assertEqual(result.metadata["official_local_error"], "missing_base_url")

    def test_request_gate_disabled_never_touches_global_search_adapter(self):
        adapter = FakeGlobalSearchAdapter("不应被调用")
        retriever = GraphRAGSearchRetriever(
            test_settings(
                graphrag_llm_api_key="fake",
                graphrag_global_search_enabled=True,
            ),
            adapter=adapter,
            allow_global_search=False,
        )

        result = retriever.retrieve("对比西安和南京的人文景点")[0]

        self.assertEqual(adapter.readiness_calls, 0)
        self.assertEqual(adapter.search_calls, 0)
        self.assertEqual(result.metadata["fallback_reason"], "official_local_failed")
        self.assertEqual(result.metadata["official_local_error"], "missing_base_url")

    def test_disabled_gate_without_local_entity_evidence_uses_wrapper(self):
        result = GraphRAGSearchRetriever(
            test_settings(graphrag_global_search_enabled=False),
            allow_global_search=True,
        ).retrieve("比较海岛和古城哪种旅行更合适？")[0]

        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_wrapper")
        self.assertEqual(result.metadata["fallback_reason"], "official_local_failed")

    def test_fake_real_adapter_can_mark_global_search(self):
        retriever = GraphRAGSearchRetriever(
            graph_settings(
                graphrag_global_search_enabled=True,
            ),
            adapter=FakeGlobalSearchAdapter("云南大理、丽江和香格里拉可形成滇西北路线。"),
            allow_global_search=True,
        )
        result = retriever.retrieve("大理、丽江、香格里拉适合怎么串成一条云南路线？")[0]
        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_global_search")
        self.assertTrue(result.metadata["global_search_available"])
        self.assertTrue(result.metadata["graphrag_relevance"])

    def test_official_api_success_records_invocation_diagnostics(self):
        async def fake_global_search(**kwargs):
            return "阳朔和张家界都拥有典型山水景观。", {
                "reports": pd.DataFrame([{"title": "阳朔与张家界"}])
            }

        settings = graph_settings(
            graphrag_global_search_enabled=True,
        )
        adapter = GraphRAGGlobalSearchAdapter(
            settings,
            global_search_callable=fake_global_search,
        )
        retriever = GraphRAGSearchRetriever(
            settings,
            adapter=adapter,
            local_adapter=UnavailableLocalSearchAdapter(),
            allow_global_search=True,
        )

        result = retriever.retrieve("阳朔和张家界哪个更适合看山水风景？")[0]

        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_global_search")
        self.assertTrue(retriever.last_diagnostics["global_search_called"])
        self.assertTrue(retriever.last_diagnostics["global_search_succeeded"])
        self.assertEqual(retriever.last_diagnostics["raw_result_type"], "str")
        self.assertGreater(retriever.last_diagnostics["raw_result_length"], 0)
        self.assertGreaterEqual(retriever.last_diagnostics["elapsed_ms"], 0)

    def test_official_api_success_with_low_relevance_falls_back_honestly(self):
        async def fake_global_search(**kwargs):
            return "贵州荔波自然风光资料。", {
                "reports": pd.DataFrame([{"title": "贵州荔波"}])
            }

        settings = graph_settings(
            graphrag_global_search_enabled=True,
        )
        adapter = GraphRAGGlobalSearchAdapter(
            settings,
            global_search_callable=fake_global_search,
        )
        retriever = GraphRAGSearchRetriever(
            settings,
            adapter=adapter,
            local_adapter=UnavailableLocalSearchAdapter(),
            allow_global_search=True,
        )

        result = retriever.retrieve("阳朔和张家界哪个更适合看山水风景？")[0]

        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_local_evidence")
        self.assertTrue(retriever.last_diagnostics["global_search_called"])
        self.assertFalse(retriever.last_diagnostics["global_search_succeeded"])
        self.assertEqual(retriever.last_diagnostics["quality_status"], "FAIL")
        self.assertEqual(retriever.last_diagnostics["global_search_error"], "low_coverage")

    def test_official_api_empty_result_falls_back_with_specific_reason(self):
        async def fake_global_search(**kwargs):
            return "", {}

        settings = graph_settings(
            graphrag_global_search_enabled=True,
        )
        adapter = GraphRAGGlobalSearchAdapter(
            settings,
            global_search_callable=fake_global_search,
        )
        retriever = GraphRAGSearchRetriever(
            settings,
            adapter=adapter,
            local_adapter=UnavailableLocalSearchAdapter(),
            allow_global_search=True,
        )

        result = retriever.retrieve("阳朔和张家界哪个更适合看山水风景？")[0]

        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_local_evidence")
        self.assertTrue(retriever.last_diagnostics["global_search_called"])
        self.assertFalse(retriever.last_diagnostics["global_search_succeeded"])
        self.assertEqual(retriever.last_diagnostics["global_search_error"], "empty_response")

    def test_official_api_exception_falls_back_without_claiming_success(self):
        async def fake_global_search(**kwargs):
            raise RuntimeError("provider failed")

        settings = graph_settings(
            graphrag_global_search_enabled=True,
        )
        adapter = GraphRAGGlobalSearchAdapter(
            settings,
            global_search_callable=fake_global_search,
        )
        retriever = GraphRAGSearchRetriever(
            settings,
            adapter=adapter,
            local_adapter=UnavailableLocalSearchAdapter(),
            allow_global_search=True,
        )

        result = retriever.retrieve("阳朔和张家界哪个更适合看山水风景？")[0]

        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_local_evidence")
        self.assertTrue(retriever.last_diagnostics["global_search_called"])
        self.assertFalse(retriever.last_diagnostics["global_search_succeeded"])
        self.assertEqual(
            retriever.last_diagnostics["global_search_error"],
            "sdk_error:RuntimeError",
        )

    def test_official_api_timeout_falls_back_with_timeout_reason(self):
        async def fake_global_search(**kwargs):
            await asyncio.sleep(0.05)
            return "不会返回", {}

        settings = graph_settings(
            graphrag_timeout_seconds=0,
            graphrag_global_search_enabled=True,
        )
        adapter = GraphRAGGlobalSearchAdapter(
            settings,
            global_search_callable=fake_global_search,
        )
        retriever = GraphRAGSearchRetriever(
            settings,
            adapter=adapter,
            local_adapter=UnavailableLocalSearchAdapter(),
            allow_global_search=True,
        )

        result = retriever.retrieve("阳朔和张家界哪个更适合看山水风景？")[0]

        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_local_evidence")
        self.assertTrue(retriever.last_diagnostics["global_search_called"])
        self.assertFalse(retriever.last_diagnostics["global_search_succeeded"])
        self.assertEqual(
            retriever.last_diagnostics["global_search_error"],
            "timeout",
        )

    def test_official_api_invalid_response_falls_back_with_specific_reason(self):
        async def fake_global_search(**kwargs):
            return {"unexpected": "shape"}

        settings = graph_settings(
            graphrag_global_search_enabled=True,
        )
        adapter = GraphRAGGlobalSearchAdapter(
            settings,
            global_search_callable=fake_global_search,
        )
        retriever = GraphRAGSearchRetriever(
            settings,
            adapter=adapter,
            local_adapter=UnavailableLocalSearchAdapter(),
            allow_global_search=True,
        )

        result = retriever.retrieve("阳朔和张家界哪个更适合看山水风景？")[0]

        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_local_evidence")
        self.assertEqual(
            retriever.last_diagnostics["global_search_error"],
            "invalid_response",
        )

    def test_incomplete_runtime_config_never_claims_real_global_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "travelmind_runtime.yaml").write_text(
                "global_search:\n  chat_model_id: default_chat_model\n",
                encoding="utf-8",
            )
            settings = replace(
                graph_settings(
                    graphrag_global_search_enabled=True,
                ),
                graphrag_config_dir=config_dir,
            )
            result = GraphRAGSearchRetriever(
                settings,
                allow_global_search=True,
            ).retrieve(
                "对比西安和南京的人文景点"
            )[0]

        self.assertNotEqual(result.metadata["retrieval_mode"], "graphrag_global_search")
        self.assertFalse(result.metadata["global_search_available"])

    def test_official_local_nonempty_low_coverage_falls_back_honestly(self):
        retriever = GraphRAGSearchRetriever(
            test_settings(),
            local_adapter=FakeLocalSearchAdapter("贵州荔波自然风光资料。"),
        )

        result = retriever.retrieve(
            "阳朔和张家界哪个更适合看山水风景？"
        )[0]

        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_local_evidence")
        self.assertEqual(result.metadata["official_local_error"], "low_coverage")
        self.assertFalse(retriever.last_diagnostics["official_local_succeeded"])

    def test_official_local_parent_only_answer_is_low_coverage_for_attraction(self):
        retriever = GraphRAGSearchRetriever(
            test_settings(),
            local_adapter=FakeLocalSearchAdapter("荔波自然风光丰富。"),
        )

        result = retriever.retrieve("荔波小七孔怎么玩比较合适？")[0]

        self.assertNotEqual(
            result.metadata["retrieval_mode"],
            "graphrag_local_search",
        )
        self.assertEqual(result.metadata["official_local_error"], "low_coverage")
        self.assertFalse(retriever.last_diagnostics["official_local_succeeded"])

    def test_official_local_missing_information_statement_is_low_coverage(self):
        retriever = GraphRAGSearchRetriever(
            test_settings(),
            local_adapter=FakeLocalSearchAdapter(
                "目前没有关于澳门的任何资料，因此无法评估澳门。"
                "重庆有较完整的慢游与交通资料。"
            ),
        )

        result = retriever.retrieve("澳门和重庆哪个更适合带老人慢游？")[0]

        self.assertNotEqual(
            result.metadata["retrieval_mode"],
            "graphrag_local_search",
        )
        self.assertEqual(result.metadata["official_local_error"], "low_coverage")
        self.assertFalse(retriever.last_diagnostics["official_local_succeeded"])

    def test_official_local_requires_source_summary(self):
        retriever = GraphRAGSearchRetriever(
            test_settings(),
            local_adapter=FakeLocalSearchAdapter(
                "阳朔与张家界都适合山水风景游。",
                source_summary=[],
            ),
        )

        result = retriever.retrieve(
            "阳朔和张家界哪个更适合看山水风景？"
        )[0]

        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_local_evidence")
        self.assertEqual(
            result.metadata["official_local_error"],
            "missing_source_summary",
        )

    def test_official_local_rejects_malformed_source_summary(self):
        retriever = GraphRAGSearchRetriever(
            test_settings(),
            local_adapter=FakeLocalSearchAdapter(
                "阳朔与张家界都适合山水风景游。",
                source_summary=[{"section": "sources", "row_count": 0}],
            ),
        )

        result = retriever.retrieve(
            "阳朔和张家界哪个更适合看山水风景？"
        )[0]

        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_local_evidence")
        self.assertEqual(
            result.metadata["official_local_error"],
            "missing_source_summary",
        )

    def test_low_coverage_global_continues_to_official_local(self):
        global_adapter = FakeGlobalSearchAdapter("贵州荔波自然风光资料。")
        local_adapter = FakeLocalSearchAdapter(
            "阳朔以喀斯特水乡见长，张家界以砂岩峰林见长。"
        )
        retriever = GraphRAGSearchRetriever(
            graph_settings(graphrag_global_search_enabled=True),
            adapter=global_adapter,
            local_adapter=local_adapter,
            allow_global_search=True,
        )

        result = retriever.retrieve(
            "阳朔和张家界哪个更适合看山水风景？"
        )[0]

        self.assertEqual(global_adapter.search_calls, 1)
        self.assertEqual(local_adapter.search_calls, 1)
        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_local_search")
        self.assertEqual(
            retriever.last_diagnostics["global_search_error"],
            "low_coverage",
        )

    def test_global_unrelated_attraction_answer_falls_back_with_low_coverage(self):
        retriever = GraphRAGSearchRetriever(
            graph_settings(graphrag_global_search_enabled=True),
            adapter=FakeGlobalSearchAdapter(
                "太平山位于香港，可俯瞰维多利亚港。"
            ),
            local_adapter=UnavailableLocalSearchAdapter(),
            allow_global_search=True,
        )

        result = retriever.retrieve("荔波小七孔怎么玩比较合适？")[0]

        self.assertNotEqual(
            result.metadata["retrieval_mode"],
            "graphrag_global_search",
        )
        self.assertEqual(result.metadata["global_search_error"], "low_coverage")
        self.assertFalse(retriever.last_diagnostics["global_search_succeeded"])

    def test_official_local_readiness_uses_real_output_lancedb(self):
        adapter = GraphRAGOfficialLocalSearchAdapter(graph_settings())

        report = adapter.readiness_report()

        self.assertTrue(report["config_ready"])
        self.assertTrue(report["vector_store_ready"])
        self.assertTrue(report["vector_table_ready"])
        self.assertEqual(
            Path(report["vector_store_path"]).resolve(),
            (graph_settings().graphrag_output_dir / "lancedb").resolve(),
        )

    def test_official_local_adapter_normalizes_safe_source_summary(self):
        async def fake_local_search(**kwargs):
            return (
                "阳朔以喀斯特山水见长，张家界以砂岩峰林见长。",
                {
                    "text_units": pd.DataFrame(
                        [
                            {
                                "id": "unit-1",
                                "text": "阳朔和张家界资料",
                                "in_context": True,
                            }
                        ]
                    )
                },
            )

        adapter = GraphRAGOfficialLocalSearchAdapter(
            graph_settings(),
            local_search_callable=fake_local_search,
        )
        retriever = GraphRAGSearchRetriever(
            graph_settings(),
            local_adapter=adapter,
        )

        result = retriever.retrieve(
            "阳朔和张家界哪个更适合看山水风景？"
        )[0]

        self.assertEqual(result.metadata["retrieval_mode"], "graphrag_local_search")
        self.assertEqual(
            result.metadata["source_summary"],
            [
                {
                    "section": "text_units",
                    "row_count": 1,
                    "titles_or_ids": ["unit-1"],
                }
            ],
        )
        self.assertTrue(retriever.last_diagnostics["official_local_called"])
        self.assertTrue(retriever.last_diagnostics["official_local_succeeded"])


if __name__ == "__main__":
    unittest.main()
