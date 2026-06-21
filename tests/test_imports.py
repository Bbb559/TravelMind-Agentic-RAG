import importlib
import unittest

from _helpers import clean_env, restore_env


class ImportIsolationTests(unittest.TestCase):
    def setUp(self):
        self.old_env = clean_env()

    def tearDown(self):
        restore_env(self.old_env)

    def test_import_travelmind_exposes_version(self):
        import travelmind

        self.assertTrue(hasattr(travelmind, "__version__"))

    def test_import_api_does_not_create_workflow_result(self):
        module = importlib.import_module("travelmind.api")
        self.assertTrue(hasattr(module, "app"))

    def test_import_cli_exposes_main(self):
        module = importlib.import_module("travelmind.cli")
        self.assertTrue(callable(module.main))

    def test_import_graph_facade(self):
        module = importlib.import_module("travelmind.graphs")
        self.assertTrue(hasattr(module, "AgenticRAGWorkflow"))

    def test_import_agents_public_api(self):
        module = importlib.import_module("travelmind.agents")
        for name in ["SystemAgent", "NaiveTravelAgent", "GraphRAGAgent", "MultimodalTravelAgent"]:
            self.assertTrue(hasattr(module, name))

    def test_import_retrievers_public_api(self):
        module = importlib.import_module("travelmind.retrievers")
        for name in ["NaiveAutoTravelRetriever", "GraphRAGSearchRetriever", "MultimodalVectorMarkdownRetriever"]:
            self.assertTrue(hasattr(module, name))

    def test_import_schemas_public_api(self):
        module = importlib.import_module("travelmind.schemas")
        for name in ["RouteDecision", "RetrieverResult", "SourceRef", "RAGAnswer", "GraphState"]:
            self.assertTrue(hasattr(module, name))

    def test_import_runtime_helpers(self):
        module = importlib.import_module("travelmind.runtime.rag_helpers")
        self.assertTrue(callable(module.parse_json_object))
        self.assertTrue(callable(module.generate_answer_text))
