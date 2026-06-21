import unittest

from _helpers import test_settings
from travelmind.agents import GraphRAGAgent, MultimodalTravelAgent, NaiveTravelAgent, SystemAgent


class AgentsContractTests(unittest.TestCase):
    def setUp(self):
        self.settings = test_settings()

    def test_naive_agent_tool_contract(self):
        agent = NaiveTravelAgent(self.settings, None)
        self.assertEqual(agent.agent_name, "naive_travel_agent")
        self.assertEqual(agent.tool_spec.name, "national_retriever_tool")
        self.assertIn("CSV", agent.tool_spec.description)

    def test_graphrag_agent_tool_contract(self):
        agent = GraphRAGAgent(self.settings, None)
        self.assertEqual(agent.agent_name, "graphrag_agent")
        self.assertEqual(agent.tool_spec.name, "national_graphrag_retriever_tool")
        self.assertIn("local_search", agent.tool_spec.boundary)
        self.assertIn("global_search", agent.tool_spec.boundary)

    def test_multimodal_agent_tool_contract(self):
        agent = MultimodalTravelAgent(self.settings, None)
        self.assertEqual(agent.agent_name, "multimodal_travel_agent")
        self.assertEqual(agent.tool_spec.name, "gang_ao_tai_retriever_tool")
        self.assertIn("Markdown", agent.tool_spec.description)

    def test_naive_prompts_are_agent_specific(self):
        agent = NaiveTravelAgent(self.settings, None)
        self.assertEqual(agent.prompt_config.generate_response, "agents/naive_generate_response.md")
        self.assertEqual(agent.prompt_config.final_answer, "agents/naive_final_answer.md")

    def test_graphrag_prompts_are_agent_specific(self):
        agent = GraphRAGAgent(self.settings, None)
        self.assertEqual(agent.prompt_config.generate_response, "agents/graphrag_generate_response.md")
        self.assertEqual(agent.prompt_config.final_answer, "agents/graphrag_final_answer.md")

    def test_multimodal_prompts_are_agent_specific(self):
        agent = MultimodalTravelAgent(self.settings, None)
        self.assertEqual(agent.prompt_config.generate_response, "agents/multimodal_generate_response.md")
        self.assertEqual(agent.prompt_config.final_answer, "agents/multimodal_final_answer.md")

    def test_tool_spec_serializes(self):
        data = NaiveTravelAgent(self.settings, None).tool_spec.to_dict()
        for key in ["name", "description", "input_schema", "output_schema", "execution_mode", "boundary"]:
            self.assertIn(key, data)

    def test_system_agent_route_returns_trace(self):
        decision, trace = SystemAgent(self.settings, None).route("大理到双廊怎么去？")
        self.assertEqual(decision.route, "naive_rag")
        self.assertIn("workflow:start", trace)
        self.assertIn("system:route:naive_rag", trace)

    def test_system_agent_fallback_trace(self):
        decision, trace = SystemAgent(self.settings, None).route("qwxjkp")
        self.assertEqual(decision.route, "invalid_input")
        self.assertEqual(trace, ["workflow:start", "input:invalid"])

    def test_naive_agent_trace_contract(self):
        answer = SystemAgent(self.settings, None).run("大理到双廊怎么去？")
        for marker in [
            "agent:naive_travel_agent:generate_response",
            "agent:naive_travel_agent:retrieve_tool:national_retriever_tool",
            "agent:naive_travel_agent:grade_search_docs",
            "agent:naive_travel_agent:generate_final_answer",
        ]:
            self.assertIn(marker, answer.trace)

    def test_multimodal_agent_trace_contract(self):
        answer = SystemAgent(self.settings, None).run("香港迪士尼怎么玩？")
        self.assertIn("agent:multimodal_travel_agent:retrieve_tool:gang_ao_tai_retriever_tool", answer.trace)
        self.assertTrue(any(step.startswith("multimodal:retriever_mode:") for step in answer.trace))

    def test_graphrag_agent_trace_contract(self):
        answer = SystemAgent(self.settings, None).run("对比西安和南京的人文景点")
        self.assertIn("agent:graphrag_agent:retrieve_tool:national_graphrag_retriever_tool", answer.trace)
        self.assertTrue(any(step.startswith("graphrag:retriever_mode:") for step in answer.trace))

    def test_graphrag_trace_records_service_cost_gate(self):
        answer = SystemAgent(self.settings, None).run("对比西安和南京的人文景点")

        self.assertIn("graphrag:global_search_disabled", answer.trace)

    def test_hybrid_trace_contains_two_agents(self):
        answer = SystemAgent(self.settings, None).run("台湾和西安哪个更适合亲子游？")
        self.assertIn("agent:hybrid_aggregator:start", answer.trace)
        self.assertIn("agent:graphrag_agent:start", answer.trace)
        self.assertIn("agent:multimodal_travel_agent:start", answer.trace)
        self.assertIn("hybrid:multi_source_candidate_aggregation", answer.trace)
        self.assertIn("agent:hybrid_aggregator:end", answer.trace)

    def test_fallback_does_not_enter_child_agent(self):
        answer = SystemAgent(self.settings, None).run("今天天气怎么样？")
        self.assertEqual(answer.route, "fallback")
        self.assertFalse(any(step.startswith("agent:") for step in answer.trace))

    def test_multimodal_boundary_says_no_ocr(self):
        boundary = MultimodalTravelAgent(self.settings, None).tool_spec.boundary
        self.assertIn("OCR", boundary)

    def test_graphrag_boundary_blocks_wrapper_conclusions(self):
        boundary = GraphRAGAgent(self.settings, None).tool_spec.boundary
        self.assertIn("evidence", boundary)
        self.assertIn("global_search", boundary)
