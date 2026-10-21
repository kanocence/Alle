import PROMPT from '@/const/prompt';
import OpenAI from 'openai';

import { DEFAULT_EXTRACT_RESULT } from '@/types';

import type { ExtractResult } from '@/types';

function parseExtractionJson(text: string): ExtractResult {
  try {
    return JSON.parse(text) as ExtractResult;
  } catch {
    // JSON.parse errors can include email content echoed in malformed model output.
    throw new Error('AI returned invalid extraction JSON');
  }
}

async function extractWithOpenAI(
  content: string,
  env: CloudflareEnv
): Promise<ExtractResult> {
  const client = new OpenAI({
    apiKey: env.OPENAI_API_KEY,
    baseURL: env.OPENAI_BASE_URL,
  });

  const response = await client.chat.completions.create({
    model: env.EXTRACT_MODEL,
    messages: [
      { role: 'system', content: PROMPT },
      { role: 'user', content },
    ],
    response_format: {
      type: 'json_schema',
      json_schema: {
        name: 'response_object',
        schema: {
          type: 'object',
          properties: {
            type: {
              type: 'string',
              enum: ['auth_code', 'auth_link', 'service_link', 'subscription_link', 'other_link', 'none']
            },
            result: { type: 'string' },
            result_text: { type: 'string' },
          },
          required: ['type', 'result', 'result_text'],
        },
      },
    },
  });

  const jsonText = response.choices[0].message.content;
  if (!jsonText) {
    throw new Error('OpenAI returned empty response');
  }

  return parseExtractionJson(jsonText);
}

async function extractWithCloudflareAI(
  content: string,
  env: CloudflareEnv
): Promise<ExtractResult> {
  const modern = env.EXTRACT_MODEL === '@cf/zai-org/glm-4.7-flash';
  const schema = {
    type: 'object',
    properties: {
      type: {
        type: 'string',
        enum: ['auth_code', 'auth_link', 'service_link', 'subscription_link', 'other_link', 'none'],
      },
      result: { type: 'string' },
      result_text: { type: 'string' },
    },
    required: ['type', 'result', 'result_text'],
    additionalProperties: false,
  };

  const result = await env.AI.run(env.EXTRACT_MODEL as keyof AiModels, {
    messages: [
      { role: 'system', content: PROMPT },
      { role: 'user', content },
    ],
    response_format: {
      type: 'json_schema',
      json_schema: modern
        ? { name: 'email_extraction', schema, strict: true }
        : schema,
    },
    stream: false,
    ...(modern ? {
      max_tokens: 512,
      temperature: 0.1,
      chat_template_kwargs: { enable_thinking: false },
    } : {}),
  });

  // Workers AI exposes both its legacy response and Chat Completions formats.
  const output = result as {
    response?: unknown;
    choices?: { message?: { content?: string | null } }[];
  };
  const response = modern
    ? output.choices?.[0]?.message?.content
    : output.response;

  if (typeof response === 'string') {
    return parseExtractionJson(response);
  }

  if (response && typeof response === 'object') {
    return response as ExtractResult;
  }

  throw new Error('Unexpected response format from Cloudflare AI');
}

export default async function extract(
  content: string,
  env: CloudflareEnv
): Promise<ExtractResult> {
  try {
    let result: ExtractResult = DEFAULT_EXTRACT_RESULT;
    if (env.OPENAI_BASE_URL && env.OPENAI_API_KEY) {
      result = await extractWithOpenAI(content, env);
    } else {
      result = await extractWithCloudflareAI(content, env);
    }
    return result;
  } catch (e) {
    console.error('Extraction error:', {
      model: env.EXTRACT_MODEL,
      name: e instanceof Error ? e.name : 'Error',
      message: e instanceof Error ? e.message : String(e),
      stack: e instanceof Error ? e.stack : undefined,
    });
    return DEFAULT_EXTRACT_RESULT;
  }
}
