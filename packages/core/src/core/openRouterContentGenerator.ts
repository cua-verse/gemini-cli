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
  // OpenRouter passes the upstream provider's cache accounting through here.
  // For Gemini models this carries Google's implicit-cache read count
  // (`cached_tokens`) and, when available, the cache-write count. The base
  // OpenAI SDK type omits these, so we declare them for the cast below.
  prompt_tokens_details?: {
    cached_tokens?: number;
    cache_write_tokens?: number;
  };
}

const DEFAULT_OPENROUTER_COMPRESSION_MODEL = 'google/gemini-3-flash-preview';

/**
 * Resolve Gemini CLI model IDs to OpenRouter model IDs.
 *
 * Chat compression is a utility request. Gemini CLI's built-in compression
 * aliases currently resolve to Google-side IDs such as `gemini-3-pro-preview`,
 * which OpenRouter may not expose. Keep normal chat model selection intact, but
 * use an OpenRouter-supported flash model for compression by default.
 */
export function resolveOpenRouterModel(
  model: string | undefined,
  role?: unknown,
): string {
  if (role === 'utility_compressor') {
    return (
      process.env['OPENROUTER_COMPRESSION_MODEL'] ||
      DEFAULT_OPENROUTER_COMPRESSION_MODEL
    );
  }

  if (!model) return '';
  if (model.startsWith('google/')) return model;
  return `google/${model}`;
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
    role?: unknown,
  ): AsyncGenerator<GenerateContentResponse> {
    try {
      const messages = convertToOpenAIFormat(request);
      const systemInstruction = extractSystemInstruction(request);

      const stream = await openRouterClient.chat.completions.create({
        model: resolveOpenRouterModel(
          request.model || (config as { model?: string }).model,
          role,
        ),
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
      // OpenRouter stamps every chunk with the generation id (`gen-...`).
      // Capture it so it can be surfaced on the Gemini responses below as
      // `responseId`; gemini-cli threads that into the stream-json transcript
      // (as the message `response_id`), letting us reconcile each turn against
      // OpenRouter's generation/cost API after the run.
      let responseId: string | undefined;

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
          if (chunk.id) resp.responseId = chunk.id;
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
        if (chunk.id) responseId = chunk.id;
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
            } catch (e) {
              // Debug: surface the malformed payload so we can diagnose
              // whether OpenRouter sent fragmented/invalid JSON.
              if (process.env['OPENROUTER_DEBUG']) {
                process.stderr.write(
                  `[openrouter] tool_call ${call.name} arg parse failed: ` +
                    `${String(e)}; raw=${JSON.stringify(call.argsText)}\n`,
                );
              }
              parsed = {};
            }
          }
          if (process.env['OPENROUTER_DEBUG']) {
            process.stderr.write(
              `[openrouter] tool_call ${call.name} args=` +
                `${JSON.stringify(parsed)}\n`,
            );
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
            um.cachedContentTokenCount =
              u.prompt_tokens_details?.cached_tokens || 0;
            resp.usageMetadata = um;
          }
          if (responseId) resp.responseId = responseId;
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
        um.cachedContentTokenCount =
          u.prompt_tokens_details?.cached_tokens || 0;
        resp.usageMetadata = um;
        if (responseId) resp.responseId = responseId;
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
      role: unknown,
    ): Promise<GenerateContentResponse> {
      try {
        const messages = convertToOpenAIFormat(request);
        const systemInstruction = extractSystemInstruction(request);

        const completion = await openRouterClient.chat.completions.create({
          model: resolveOpenRouterModel(
            request.model || (config as { model?: string }).model,
            role,
          ),
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
      role: unknown,
    ): Promise<AsyncGenerator<GenerateContentResponse>> {
      return doGenerateContentStream(request, role);
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

export function convertToOpenAIFormat(
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

      // gemini-cli packs tool results as their own turn with role 'user'
      // (Gemini's native convention), not role 'function'. Gating on
      // role === 'function' here dropped native-tool results (read_file etc.)
      // over OpenRouter: the parts carried no `text`, so they fell through to
      // the text branch and emitted an empty message — the model never saw the
      // tool output and hallucinated file contents. Fire on functionResponse
      // presence regardless of role so the result reaches the model.
      if (functionResponses.length > 0) {
        const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

        // Collect image parts that ride along with tool results. MCP tools
        // (e.g. the CUA `screenshot` tool) return their image as an
        // `inlineData` Part. Depending on `supportsMultimodalFunctionResponse`
        // it is either nested under `functionResponse.parts` or emitted as a
        // sibling Part in this same content turn. Either way it must reach the
        // model — OpenAI `role:"tool"` messages can only carry a string, so we
        // re-emit the images as a following `role:"user"` message with
        // `image_url` content. Without this, the model is blind to every
        // screenshot over OpenRouter and hallucinates GUI contents.
        const imageParts: Part[] = [];

        functionResponses.forEach((part: Part, index: number) => {
          messages.push({
            role: 'tool' as const,
            // Prefer functionResponse.id to match the original tool_call.id
            // emitted on the assistant turn (OpenAI requires strict matching).
            // Fall back to name only when id is missing (older gemini-cli paths).
            tool_call_id:
              part.functionResponse?.id ||
              part.functionResponse?.name ||
              `call_${index}`,
            content: JSON.stringify(part.functionResponse?.response || {}),
          });
          // Nested multimodal-function-response case.
          const nested = (
            part.functionResponse as unknown as { parts?: Part[] }
          )?.parts;
          if (Array.isArray(nested)) {
            for (const p of nested) {
              if (p?.inlineData?.data) imageParts.push(p);
            }
          }
        });

        // Sibling case: inlineData Parts in the same content turn.
        for (const p of parts) {
          if (p?.inlineData?.data) imageParts.push(p);
        }

        const imageContent = imageParts.map((p: Part) => ({
          type: 'image_url' as const,
          image_url: {
            url: `data:${p.inlineData?.mimeType || 'image/png'};base64,${
              p.inlineData?.data
            }`,
          },
        }));
        if (imageContent.length > 0) {
          messages.push({ role: 'user', content: imageContent });
        }

        return messages;
      }

      const textParts = parts.filter((part: Part) => part && 'text' in part);
      const text = textParts
        .map((part: Part) => ('text' in part ? part.text || '' : ''))
        .join('\n');

      // Forward inline images on plain user turns too (e.g. a user-supplied
      // screenshot). OpenAI only accepts image content on user messages, so
      // assistant turns keep their text-only form.
      const imageParts = parts.filter((part: Part) => part?.inlineData?.data);
      if (role === 'user' && imageParts.length > 0) {
        const content: OpenAI.Chat.ChatCompletionContentPart[] = [];
        if (text) content.push({ type: 'text', text });
        for (const p of imageParts) {
          content.push({
            type: 'image_url',
            image_url: {
              url: `data:${p.inlineData?.mimeType || 'image/png'};base64,${
                p.inlineData?.data
              }`,
            },
          });
        }
        return { role: 'user' as const, content };
      }

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

  // Collect function declarations from ALL tool objects, not just the first.
  // gemini-cli registers built-in and MCP tools as separate Tool entries.
  const allDecls: GenaiTool['functionDeclarations'] = [];
  for (const t of toolsArray) {
    if (!t || typeof t === 'string') continue;
    const decls = (t as GenaiTool).functionDeclarations;
    if (decls) allDecls.push(...decls);
  }
  if (!allDecls || allDecls.length === 0) return undefined;

  return allDecls.map((func) => {
    // FunctionDeclaration has two mutually-exclusive schema fields:
    //   parameters:          Schema (Gemini internal type with UPPER-CASE types)
    //   parametersJsonSchema: raw JSON Schema object
    // gemini-cli core tools use parametersJsonSchema — drop it directly into
    // the OpenAI tool definition. Falling back to `parameters` preserves
    // compatibility with tools that set only the Gemini-Schema variant.
    const jsonSchema = (func as { parametersJsonSchema?: unknown })
      .parametersJsonSchema;
    const parameters =
      jsonSchema !== undefined
        ? (jsonSchema as Record<string, unknown>)
        : ((func.parameters || {}) as Record<string, unknown>);
    return {
      type: 'function' as const,
      function: {
        name: func.name || '',
        description: func.description,
        parameters,
      },
    };
  });
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
  usageMetadata.cachedContentTokenCount =
    usage?.prompt_tokens_details?.cached_tokens || 0;

  const response = new GenerateContentResponse();
  response.candidates = candidates;
  response.promptFeedback = promptFeedback;
  response.usageMetadata = usageMetadata;
  // Surface the OpenRouter generation id for transcript reconciliation.
  if (completion.id) response.responseId = completion.id;

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
