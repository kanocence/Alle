import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import test from 'node:test';
import ts from 'typescript';

const root = new URL('../', import.meta.url).pathname;
function loadExtractor({ completion, requestError } = {}) {
  const errors = [];
  const fakeOpenAI = class {
    constructor() {
      this.chat = { completions: { create: async () => {
        if (requestError) throw requestError;
        return completion;
      } } };
    }
  };
  function load(relative) {
    const path = resolve(root, relative);
    const { outputText } = ts.transpileModule(readFileSync(path, 'utf8'), {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
    });
    const loadedModule = { exports: {} };
    const require = (name) => {
      if (name === 'openai') return fakeOpenAI;
      if (name === '@/const/prompt') return load('src/const/prompt.ts');
      if (name === '@/types') return load('src/types/email.ts');
      throw new Error('Unexpected dependency: ' + name);
    };
    new Function('require', 'module', 'exports', 'console', outputText)(
      require, loadedModule, loadedModule.exports, { error: (...args) => errors.push(args) });
    return loadedModule.exports;
  }
  return { extract: load('src/lib/email/extract.ts').default, errors };
}

const wanted = { type: 'auth_code', result: '482761', result_text: '' };
const empty = { type: 'none', result: '', result_text: '' };
const glm = '@cf/zai-org/glm-4.7-flash';
function completion(content = JSON.stringify(wanted)) {
  return {
    id: 'synthetic-completion', object: 'chat.completion', created: 1792570000,
    model: glm, choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content } }],
    usage: { prompt_tokens: 30, completion_tokens: 20, total_tokens: 50 },
  };
}

test('GLM JSON schema request yields extracted verification code', async () => {
  const { extract, errors } = loadExtractor();
  const actual = await extract('测试验证码 482761', {
    EXTRACT_MODEL: glm, AI: { run: async (model, request) => {
      assert.equal(model, glm);
      assert.equal(request.response_format.type, 'json_schema');
      const wrapped = request.response_format.json_schema;
      assert.equal(typeof wrapped.name, 'string');
      assert.equal(wrapped.schema.type, 'object');
      assert.deepEqual(wrapped.schema.required, ['type', 'result', 'result_text']);
      assert.equal(wrapped.schema.additionalProperties, false);
      assert.equal(request.chat_template_kwargs.enable_thinking, false);
      assert.ok(request.max_tokens > 0 && request.max_tokens <= 512);
      assert.ok(request.messages[0].content.includes('expert email analyzer'));
      assert.equal(request.messages[1].content, '测试验证码 482761');
      return completion();
    } },
  });
  assert.deepEqual(actual, wanted);
  assert.deepEqual(errors, []);
});

for (const response of [JSON.stringify(wanted), wanted]) {
  test('legacy response remains supported: ' + typeof response, async () => {
    const { extract, errors } = loadExtractor();
    const actual = await extract('验证码 482761', {
      EXTRACT_MODEL: '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
      AI: { run: async (_model, request) => {
        assert.equal(request.response_format.json_schema.type, 'object');
        return { response };
      } },
    });
    assert.deepEqual(actual, wanted);
    assert.deepEqual(errors, []);
  });
}

test('empty GLM output retains fallback and records useful error context', async () => {
  const { extract, errors } = loadExtractor();
  const actual = await extract('private synthetic body', {
    EXTRACT_MODEL: glm, AI: { run: async () => completion(null) },
  });
  assert.deepEqual(actual, empty);
  assert.equal(errors.length, 1);
  assert.equal(errors[0][1].model, glm);
  assert.match(errors[0][1].message, /response format|empty/i);
  assert.ok(!JSON.stringify(errors).includes('private synthetic body'));
});

test('upstream failure retains fallback and logs error name and message', async () => {
  const { extract, errors } = loadExtractor();
  const actual = await extract('private synthetic body', {
    EXTRACT_MODEL: glm, AI: { run: async () => { throw new Error('Model unavailable'); } },
  });
  assert.deepEqual(actual, empty);
  assert.equal(errors.length, 1);
  assert.equal(errors[0][1].model, glm);
  assert.equal(errors[0][1].name, 'Error');
  assert.equal(errors[0][1].message, 'Model unavailable');
  assert.ok(!JSON.stringify(errors).includes('private synthetic body'));
});

test('existing external provider selection remains supported', async () => {
  const { extract, errors } = loadExtractor({ completion: completion() });
  const actual = await extract('验证码 482761', {
    EXTRACT_MODEL: 'external-model',
    OPENAI_BASE_URL: 'https://synthetic.invalid/v1',
    OPENAI_API_KEY: 'synthetic-test-value',
    AI: { run: async () => { throw new Error('Wrong provider selected'); } },
  });
  assert.deepEqual(actual, wanted);
  assert.deepEqual(errors, []);
});

for (const external of [false, true]) {
  test('malformed AI JSON does not expose model output in logs: ' + (external ? 'external' : 'Workers AI'), async () => {
    const privateOutput = 'Sensitive synthetic email body: account 935174';
    const { extract, errors } = loadExtractor({ completion: completion(privateOutput) });
    const actual = await extract('sensitive synthetic input', {
      EXTRACT_MODEL: glm,
      ...(external ? {
        OPENAI_BASE_URL: 'https://synthetic.invalid/v1',
        OPENAI_API_KEY: 'synthetic-test-value',
      } : {}),
      AI: { run: async () => completion(privateOutput) },
    });
    assert.deepEqual(actual, empty);
    assert.equal(errors.length, 1);
    const logged = JSON.stringify(errors);
    assert.ok(!logged.includes('Sensitive'));
    assert.ok(!logged.includes('935174'));
    assert.match(errors[0][1].message, /invalid.*JSON/i);
  });
}

