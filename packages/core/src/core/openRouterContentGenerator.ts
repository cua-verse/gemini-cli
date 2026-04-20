/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * OpenRouter ContentGenerator. Ported from heartyguy's PR #2319
 * (https://github.com/google-gemini/gemini-cli/pull/2319) and adapted to
 * the v0.38.1 ContentGenerator interface, which threads `userPromptId` and
 * `role` through generateContent{,Stream} for telemetry. Those parameters
 * are accepted and ignored here since OpenRouter has no equivalent.
 *
 * This adapter bridges two type systems (OpenAI SDK ↔ @google/genai); the
 * casts below are structural conversions between those type universes and
 * cannot be expressed via the stricter ESLint rules.
 */
/* eslint-disable @typescript-eslint/no-unsafe-type-assertion */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */

import OpenAI from 'openai';
import type {
  ContentGenerator,
  ContentGeneratorConfig,
} from './contentGenerator.js';
import {
  GenerateContentResponse,
  FinishReason,
  GenerateContentResponsePromptFeedback,
  BlockedReason,
  GenerateContentResponseUsageMetadata,
} from '@google/genai';
import type {
  CountTokensResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Content,
  Part,
  Candidate,
  ContentUnion,
  ToolListUnion,
  Tool as GenaiTool,
} from '@google/genai';

interface OpenRouterUsage {
  completion_tokens?: number;
  prompt_tokens?: number;
  total_tokens?: number;
}

export function createOpenRouterContentGenerator(
  config: ContentGeneratorConfig,
  httpOptions: { headers: Record<string, string> },
): ContentGenerator {
  const openRouterClient = new OpenAI({
    baseURL: config.baseUrl || 'https://openrouter.ai/api/v1',
    apiKey: config.apiKey,
    defaultHeaders: {
      ...httpOptions.headers,
      'HTTP-Referer': 'https://github.com/google-gemini/gemini-cli',
      'X-Title': 'Gemini CLI',
    },
  });

  async function* doGenerateContentStream(
    request: GenerateContentParameters,
  ): AsyncGenerator<GenerateContentResponse> {
    try {
      const messages = convertToOpenAIFormat(request);
      const systemInstruction = extractSystemInstruction(request);

      const stream = await openRouterClient.chat.completions.create({
        model: request.model || (config as { model?: string }).model || '',
        messages: systemInstruction
          ? [{ role: 'system', content: systemInstruction }, ...messages]
          : messages,
        temperature: request.config?.temperature,
        top_p: request.config?.topP,
        max_tokens: request.config?.maxOutputTokens || 20000,
        tools: convertTools(request.config?.tools),
        stream: true,
        stream_options: { include_usage: true },
      });

      // OpenAI streams tool calls as fragments:
      //   chunk 1: {index: 0, function: {name: "list_directory", arguments: ""}}
      //   chunk 2: {index: 0, function: {arguments: "{\"dir_path"}}
      //   chunk 3: {index: 0, function: {arguments: "\":\"/tmp\"}"}}
      // The Gemini @google/genai interface expects each yielded functionCall
      // to be COMPLETE (name + parsed JSON args). Emitting per-chunk produces
      // a "name=undefined, args={}" call followed by orphan arg fragments,
      // which gemini-cli's tool loop treats as `undefined_tool_name`.
      //
      // Fix: accumulate tool_call fragments across chunks keyed by index;
      // emit a completed functionCall only when finish_reason or stream-end
      // signals that the tool call is done.
      type PendingCall = { name: string; argsText: string; id?: string };
      const pending = new Map<number, PendingCall>();
      let finalUsage: OpenAI.Completions.CompletionUsage | undefined;
      let finalFinishReason: string | null | undefined;

      for await (const chunk of stream) {
        const choice = chunk.choices?.[0];
        const delta = choice?.delta;

        // Text deltas — yield immediately so the caller can stream them.
        if (delta?.content) {
          const parts: Part[] = [{ text: delta.content }];
          const resp = new GenerateContentResponse();
          resp.candidates = [
            {
              content: { role: 'model', parts },
              finishReason: FinishReason.STOP,
              avgLogprobs: 0,
            },
          ];
          const fb = new GenerateContentResponsePromptFeedback();
          fb.blockReason = BlockedReason.BLOCKED_REASON_UNSPECIFIED;
          fb.safetyRatings = [];
          resp.promptFeedback = fb;
          yield resp;
        }

        // Tool call deltas — accumulate, do not yield yet.
        if (delta?.tool_calls) {
          for (const tc of delta.tool_calls) {
            const idx = typeof tc.index === 'number' ? tc.index : 0;
            const cur = pending.get(idx) || { name: '', argsText: '' };
            if (tc.id) cur.id = tc.id;
            if (tc.function?.name) cur.name = tc.function.name;
            if (tc.function?.arguments) {
              cur.argsText += tc.function.arguments;
            }
            pending.set(idx, cur);
          }
        }

        if (choice?.finish_reason) finalFinishReason = choice.finish_reason;
        if (chunk.usage) finalUsage = chunk.usage;
      }

      // Stream ended. If we accumulated tool calls, emit them now as one
      // response containing all complete functionCalls.
      if (pending.size > 0) {
        const toolParts: Part[] = [];
        for (const call of pending.values()) {
          if (!call.name) continue; // skip malformed
          let parsed: Record<string, unknown> = {};
          if (call.argsText) {
            try {
              parsed = JSON.parse(call.argsText);
            } catch {
              parsed = {};
            }
          }
          toolParts.push({
            functionCall: {
              id: call.id,
              name: call.name,
              args: parsed,
            },
          });
        }

        if (toolParts.length > 0) {
          const resp = new GenerateContentResponse();
          resp.candidates = [
            {
              content: { role: 'model', parts: toolParts },
              finishReason: finalFinishReason
                ? mapFinishReason(finalFinishReason)
                : FinishReason.STOP,
              avgLogprobs: 0,
            },
          ];
          const fb = new GenerateContentResponsePromptFeedback();
          fb.blockReason = BlockedReason.BLOCKED_REASON_UNSPECIFIED;
          fb.safetyRatings = [];
          resp.promptFeedback = fb;
          if (finalUsage) {
            const um = new GenerateContentResponseUsageMetadata();
            const u = finalUsage as OpenRouterUsage;
            um.promptTokenCount = u.prompt_tokens || 0;
            um.candidatesTokenCount = u.completion_tokens || 0;
            um.totalTokenCount = u.total_tokens || 0;
            um.cachedContentTokenCount = 0;
            resp.usageMetadata = um;
          }
          yield resp;
          return;
        }
      }

      // No tool calls: emit a final empty-parts response carrying usage so
      // caller can close the stream with token counts.
      if (finalUsage) {
        const resp = new GenerateContentResponse();
        resp.candidates = [];
        const fb = new GenerateContentResponsePromptFeedback();
        fb.blockReason = BlockedReason.BLOCKED_REASON_UNSPECIFIED;
        fb.safetyRatings = [];
        resp.promptFeedback = fb;
        const um = new GenerateContentResponseUsageMetadata();
        const u = finalUsage as OpenRouterUsage;
        um.promptTokenCount = u.prompt_tokens || 0;
        um.candidatesTokenCount = u.completion_tokens || 0;
        um.totalTokenCount = u.total_tokens || 0;
        um.cachedContentTokenCount = 0;
        resp.usageMetadata = um;
        yield resp;
      }
    } catch (error) {
      throw convertError(error);
    }
  }

  const openRouterContentGenerator: ContentGenerator = {
    async generateContent(
      request: GenerateContentParameters,
      _userPromptId: string,
      _role: unknown,
    ): Promise<GenerateContentResponse> {
      try {
        const messages = convertToOpenAIFormat(request);
        const systemInstruction = extractSystemInstruction(request);

        const completion = await openRouterClient.chat.completions.create({
          model: request.model || (config as { model?: string }).model || '',
          messages: systemInstruction
            ? [{ role: 'system', content: systemInstruction }, ...messages]
            : messages,
          temperature: request.config?.temperature,
          top_p: request.config?.topP,
          max_tokens: request.config?.maxOutputTokens || 20000,
          tools: convertTools(request.config?.tools),
          response_format:
            request.config?.responseMimeType === 'application/json'
              ? { type: 'json_object' }
              : undefined,
          stream: false,
          stream_options: { include_usage: true },
        });

        return convertToGeminiResponse(
          completion as OpenAI.Chat.ChatCompletion,
        );
      } catch (error) {
        throw convertError(error);
      }
    },

    async generateContentStream(
      request: GenerateContentParameters,
      _userPromptId: string,
      _role: unknown,
    ): Promise<AsyncGenerator<GenerateContentResponse>> {
      return doGenerateContentStream(request);
    },

    async countTokens(
      request: CountTokensParameters,
    ): Promise<CountTokensResponse> {
      // OpenRouter doesn't expose a dedicated token-counting endpoint.
      // Rough estimate: 1 token ≈ 4 characters of concatenated text.
      const contents = normalizeContents(request.contents);
      const totalText = contents
        .map(
          (content: Content) =>
            content.parts
              ?.map((part: Part) => {
                if ('text' in part && part.text) return part.text;
                return '';
              })
              .join(' ') || '',
        )
        .join(' ');

      const estimatedTokens = Math.ceil(totalText.length / 4);

      return {
        totalTokens: estimatedTokens,
        cachedContentTokenCount: 0,
      };
    },

    async embedContent(
      _request: EmbedContentParameters,
    ): Promise<EmbedContentResponse> {
      throw new Error(
        'Embeddings are not supported through OpenRouter for Gemini models',
      );
    },
  };

  return openRouterContentGenerator;
}

function normalizeContents(contents: ContentUnion | ContentUnion[]): Content[] {
  if (typeof contents === 'string') {
    return [{ role: 'user', parts: [{ text: contents }] }];
  }

  if (Array.isArray(contents)) {
    return contents.map((content) => {
      if (typeof content === 'string') {
        return { role: 'user', parts: [{ text: content }] };
      }
      if (Array.isArray(content)) {
        const parts: Part[] = content.map((part) => {
          if (typeof part === 'string') {
            return { text: part };
          }
          return part;
        });
        return { role: 'user', parts };
      }
      return content as Content;
    });
  }

  return [contents as Content];
}

function extractSystemInstruction(
  request: GenerateContentParameters,
): string | undefined {
  const instruction = request.config?.systemInstruction;
  if (!instruction) return undefined;

  if (typeof instruction === 'string') {
    return instruction;
  }

  if ('parts' in instruction && instruction.parts) {
    return instruction.parts
      .map((part: Part) => ('text' in part && part.text ? part.text : ''))
      .join('\n');
  }

  return undefined;
}

function convertToOpenAIFormat(
  request: GenerateContentParameters,
): OpenAI.Chat.ChatCompletionMessageParam[] {
  const contents = normalizeContents(request.contents);

  const result = contents
    .map((content: Content) => {
      const role =
        content.role === 'model' ? 'assistant' : (content.role as string);
      const parts = content.parts || [];

      if (parts.length === 1 && parts[0] && 'text' in parts[0]) {
        return {
          role: role as 'user' | 'assistant',
          content: parts[0].text || '',
        };
      }

      const functionCalls = parts.filter(
        (part: Part) => part && 'functionCall' in part,
      );

      if (functionCalls.length > 0 && role === 'assistant') {
        const toolCalls = functionCalls
          .map((part: Part, index: number) => {
            const functionCall = part.functionCall;
            if (!functionCall) return null;

            return {
              id: functionCall.id || `call_${index}`,
              type: 'function' as const,
              function: {
                name: functionCall.name || '',
                arguments: JSON.stringify(functionCall.args || {}),
              },
            };
          })
          .filter(Boolean);

        return {
          role: 'assistant' as const,
          content: null,
          tool_calls: toolCalls as OpenAI.Chat.ChatCompletionMessageToolCall[],
        };
      }

      const functionResponses = parts.filter(
        (part: Part) => part && 'functionResponse' in part,
      );

      if (functionResponses.length > 0 && role === 'function') {
        return functionResponses.map((part: Part, index: number) => ({
          role: 'tool' as const,
          tool_call_id: part.functionResponse?.name || `call_${index}`,
          content: JSON.stringify(part.functionResponse?.response || {}),
        }));
      }

      const textParts = parts.filter((part: Part) => part && 'text' in part);
      const text = textParts
        .map((part: Part) => ('text' in part ? part.text || '' : ''))
        .join('\n');

      return {
        role: role === 'user' ? 'user' : 'assistant',
        content: text,
      };
    })
    .flat();
  return result as OpenAI.Chat.ChatCompletionMessageParam[];
}

function convertTools(
  tools?: ToolListUnion,
): OpenAI.Chat.ChatCompletionTool[] | undefined {
  if (!tools) return undefined;

  const toolsArray = Array.isArray(tools) ? tools : [tools];
  if (toolsArray.length === 0) return undefined;

  const firstTool = toolsArray[0];
  if (!firstTool || typeof firstTool === 'string') return undefined;

  const functionDeclarations = (firstTool as GenaiTool).functionDeclarations;
  if (!functionDeclarations) return undefined;

  return functionDeclarations.map((func) => ({
    type: 'function' as const,
    function: {
      name: func.name || '',
      description: func.description,
      parameters: (func.parameters || {}) as Record<string, unknown>,
    },
  }));
}

function convertToGeminiResponse(
  completion: OpenAI.Chat.ChatCompletion,
): GenerateContentResponse {
  const choice = completion.choices[0];
  const message = choice.message;

  const parts: Part[] = [];

  if (message.content) {
    parts.push({ text: message.content });
  }

  if (message.tool_calls) {
    for (const toolCall of message.tool_calls) {
      if (toolCall.function) {
        parts.push({
          functionCall: {
            name: toolCall.function.name,
            args: JSON.parse(toolCall.function.arguments),
          },
        });
      }
    }
  }

  const candidates: Candidate[] = [
    {
      content: {
        role: 'model',
        parts,
      },
      finishReason: mapFinishReason(choice.finish_reason),
      avgLogprobs: 0,
    },
  ];

  const promptFeedback = new GenerateContentResponsePromptFeedback();
  promptFeedback.blockReason = BlockedReason.BLOCKED_REASON_UNSPECIFIED;
  promptFeedback.safetyRatings = [];

  const usage = completion.usage as OpenRouterUsage | undefined;

  const usageMetadata = new GenerateContentResponseUsageMetadata();
  usageMetadata.promptTokenCount = usage?.prompt_tokens || 0;
  usageMetadata.candidatesTokenCount = usage?.completion_tokens || 0;
  usageMetadata.totalTokenCount = usage?.total_tokens || 0;
  usageMetadata.cachedContentTokenCount = 0;

  const response = new GenerateContentResponse();
  response.candidates = candidates;
  response.promptFeedback = promptFeedback;
  response.usageMetadata = usageMetadata;

  return response;
}

function mapFinishReason(reason: string | null | undefined): FinishReason {
  switch (reason) {
    case 'stop':
      return FinishReason.STOP;
    case 'length':
      return FinishReason.MAX_TOKENS;
    case 'tool_calls':
    case 'function_call':
      return FinishReason.STOP;
    case 'content_filter':
      return FinishReason.SAFETY;
    default:
      return FinishReason.OTHER;
  }
}

function convertError(error: unknown): Error {
  if (error instanceof OpenAI.APIError) {
    const message = `OpenRouter API Error: ${error.status} - ${error.message}`;
    const newError = new Error(message);
    (newError as Error & { status?: number }).status = error.status;
    return newError;
  }
  return error instanceof Error ? error : new Error(String(error));
}
