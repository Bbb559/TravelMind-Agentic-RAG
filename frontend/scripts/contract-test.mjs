import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import ts from 'typescript';

const sourceUrl = new URL('../src/presentation.ts', import.meta.url);
const source = await readFile(sourceUrl, 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ES2022,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`;
const {
  extractAgent,
  generationModeLabel,
  retrievalModeLabel,
  shouldShowGlobalSearchNotice,
} = await import(moduleUrl);

assert.equal(
  extractAgent(
    [
      'agent:hybrid_aggregator:start',
      'agent:graphrag_agent:start',
      'agent:multimodal_travel_agent:start',
    ],
    'hybrid_rag',
  ),
  'HybridAggregator',
);
assert.equal(
  extractAgent(['agent:graphrag_agent:start'], 'graphrag'),
  'GraphRAGAgent',
);
assert.equal(extractAgent([], 'naive_rag'), 'NaiveTravelAgent');
assert.equal(shouldShowGlobalSearchNotice('naive_rag'), false);
assert.equal(shouldShowGlobalSearchNotice('multimodal_rag'), false);
assert.equal(shouldShowGlobalSearchNotice('graphrag'), true);
assert.equal(shouldShowGlobalSearchNotice('hybrid_rag'), true);
assert.equal(
  retrievalModeLabel('multi_source_candidate_aggregation'),
  '多源候选聚合',
);
assert.equal(generationModeLabel('llm'), 'LLM 生成');
assert.equal(generationModeLabel('template'), '模板生成');
assert.equal(generationModeLabel('official_response'), '官方响应');
assert.equal(generationModeLabel('none'), '未生成');

const css = await readFile(new URL('../src/App.css', import.meta.url), 'utf8');
for (const declaration of [
  'white-space: pre-wrap',
  'overflow-wrap: anywhere',
  'word-break: break-word',
  'max-width: 100%',
  'max-height: 360px',
  'overflow-y: auto',
  'overflow-x: hidden',
]) {
  assert.ok(
    css.includes(declaration),
    `answer panel is missing ${declaration}`,
  );
}

console.log('frontend contract: PASS');
