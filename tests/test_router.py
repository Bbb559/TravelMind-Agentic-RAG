import unittest

from _helpers import FakeLLM, test_settings
from travelmind.agents.router import SupervisorRouter, SystemRouter


class RouterTests(unittest.TestCase):
    def test_rule_routes_single_attraction_comparative_wording_to_naive(self):
        decision = SupervisorRouter().route("荔波小七孔怎么玩比较合适？")

        self.assertEqual(decision.route, "naive_rag")
        self.assertEqual(decision.query_type, "detail")

    def test_rule_collapses_province_city_attraction_hierarchy(self):
        queries = [
            "贵州荔波小七孔怎么玩比较合适？",
            "成都都江堰适合怎么玩？",
            "大理双廊怎么安排半天游？",
        ]

        for query in queries:
            with self.subTest(query=query):
                decision = SupervisorRouter().route(query)
                self.assertEqual(decision.route, "naive_rag")
                self.assertEqual(len(set(decision.entities)), 1)

    def test_rule_routes_same_destination_scope_comparison_to_naive(self):
        decision = SupervisorRouter().route("大理和双廊哪个更适合住宿？")

        self.assertEqual(decision.route, "naive_rag")

    def test_rule_routes_single_destination_detail_matrix_to_naive(self):
        queries = [
            "荔波小七孔怎么玩比较合适？",
            "阳朔有哪些必打卡景点？",
            "大理到双廊怎么去？",
            "成都有什么美食？",
            "西安有哪些历史景点？",
            "总结一下云南亲子游推荐",
            "小七孔和大七孔哪个更值得去？",
        ]

        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(SupervisorRouter().route(query).route, "naive_rag")

    def test_rule_routes_dali_to_naive(self):
        self.assertEqual(SupervisorRouter().route("大理到双廊怎么去？").route, "naive_rag")

    def test_rule_routes_hongkong_to_multimodal(self):
        self.assertEqual(SupervisorRouter().route("香港迪士尼怎么玩？").route, "multimodal_rag")

    def test_rule_routes_shanghai_weekend_to_naive(self):
        self.assertEqual(SupervisorRouter().route("从上海出发有哪些景点适合周末去？").route, "naive_rag")

    def test_rule_routes_guizhou_nature_recommendation_to_naive(self):
        self.assertEqual(SupervisorRouter().route("贵州有哪些适合自然风光游的地方？").route, "naive_rag")

    def test_rule_keeps_destination_comparison_on_graphrag(self):
        self.assertEqual(SupervisorRouter().route("阳朔和张家界哪个更适合看山水风景？").route, "graphrag")

    def test_rule_keeps_multi_destination_route_summary_on_graphrag(self):
        query = "大理、丽江、香格里拉适合怎么串成一条云南路线？"
        self.assertEqual(SupervisorRouter().route(query).route, "graphrag")

    def test_rule_routes_taiwan_xian_to_hybrid(self):
        self.assertEqual(SupervisorRouter().route("台湾和西安哪个更适合亲子游？").route, "hybrid_rag")

    def test_rule_routes_noise_to_fallback(self):
        self.assertEqual(SupervisorRouter().route("qwxjkp").route, "fallback")

    def test_rule_routes_weather_to_fallback(self):
        self.assertEqual(SupervisorRouter().route("今天天气怎么样？").route, "fallback")

    def test_rule_routes_code_to_fallback(self):
        self.assertEqual(SupervisorRouter().route("帮我写一段 Python 快排").route, "fallback")

    def test_rule_routes_upload_image_to_fallback(self):
        self.assertEqual(SupervisorRouter().route("上传一张景区照片并询问适合的游玩路线").route, "fallback")

    def test_llm_router_success(self):
        settings = test_settings(llm_enabled=True, system_agent_llm_router_enabled=True)
        fake = FakeLLM(['{"route":"multimodal_rag","confidence":"high","reason":"ok","query_type":"gat","entities":["香港"],"matched_terms":["香港"]}'])
        router = SystemRouter(settings, fake)
        decision = router.route("香港迪士尼怎么玩？")
        self.assertEqual(decision.route, "multimodal_rag")
        self.assertEqual(router.last_route_source, "llm")

    def test_llm_router_invalid_json_fallbacks(self):
        settings = test_settings(llm_enabled=True, system_agent_llm_router_enabled=True)
        router = SystemRouter(settings, FakeLLM(["not-json"]))
        self.assertEqual(router.route("香港迪士尼怎么玩？").route, "multimodal_rag")
        self.assertEqual(router.last_route_source, "rule_fallback")

    def test_llm_router_invalid_route_fallbacks(self):
        settings = test_settings(llm_enabled=True, system_agent_llm_router_enabled=True)
        router = SystemRouter(settings, FakeLLM(['{"route":"bad","confidence":"high","reason":"","query_type":"","entities":[],"matched_terms":[]}']))
        self.assertEqual(router.route("大理到双廊怎么去？").route, "naive_rag")
        self.assertEqual(router.last_route_source, "rule_fallback")

    def test_llm_router_low_confidence_fallbacks(self):
        settings = test_settings(llm_enabled=True, system_agent_llm_router_enabled=True)
        router = SystemRouter(settings, FakeLLM(['{"route":"graphrag","confidence":"low","reason":"","query_type":"","entities":[],"matched_terms":[]}']))
        self.assertEqual(router.route("大理到双廊怎么去？").route, "naive_rag")
        self.assertEqual(router.last_fallback_reason, "low_confidence")

    def test_llm_router_field_type_error_fallbacks(self):
        settings = test_settings(llm_enabled=True, system_agent_llm_router_enabled=True)
        router = SystemRouter(settings, FakeLLM(['{"route":"naive_rag","confidence":"high","reason":"","query_type":"","entities":"bad","matched_terms":[]}']))
        self.assertEqual(router.route("大理到双廊怎么去？").route, "naive_rag")
        self.assertEqual(router.last_route_source, "rule_fallback")

    def test_unsupported_guard_prevents_llm_call(self):
        settings = test_settings(llm_enabled=True, system_agent_llm_router_enabled=True)
        fake = FakeLLM(['{"route":"multimodal_rag","confidence":"high","reason":"","query_type":"","entities":[],"matched_terms":[]}'])
        router = SystemRouter(settings, fake)
        self.assertEqual(router.route("现在解析这个 PDF 里的图片和文字").route, "fallback")
        self.assertEqual(fake.prompts, [])

    def test_rule_guard_corrects_llm_wrong_route(self):
        settings = test_settings(llm_enabled=True, system_agent_llm_router_enabled=True)
        fake = FakeLLM(['{"route":"graphrag","confidence":"high","reason":"","query_type":"","entities":[],"matched_terms":[]}'])
        decision = SystemRouter(settings, fake).route("从上海出发有哪些景点适合周末去？")
        self.assertEqual(decision.route, "naive_rag")

    def test_disabled_llm_router_uses_rule(self):
        settings = test_settings(llm_enabled=True, system_agent_llm_router_enabled=False)
        fake = FakeLLM(['{"route":"multimodal_rag","confidence":"high","reason":"","query_type":"","entities":[],"matched_terms":[]}'])
        router = SystemRouter(settings, fake)
        self.assertEqual(router.route("大理到双廊怎么去？").route, "naive_rag")
        self.assertEqual(fake.prompts, [])

    def test_llm_router_exception_fallbacks(self):
        settings = test_settings(llm_enabled=True, system_agent_llm_router_enabled=True)
        router = SystemRouter(settings, FakeLLM(error=RuntimeError("boom")))
        self.assertEqual(router.route("香港迪士尼怎么玩？").route, "multimodal_rag")
        self.assertEqual(router.last_route_source, "rule_fallback")

    def test_all_rule_routes_are_known(self):
        for query in ["大理", "香港", "上海周边", "台湾和西安", "qwxjkp"]:
            self.assertIn(SupervisorRouter().route(query).route, {"naive_rag", "graphrag", "multimodal_rag", "hybrid_rag", "fallback"})
