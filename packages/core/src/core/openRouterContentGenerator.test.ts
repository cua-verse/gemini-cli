/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  convertToOpenAIFormat,
  resolveOpenRouterModel,
} from './openRouterContentGenerator.js';

describe('resolveOpenRouterModel', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('prefixes normal Gemini model IDs for OpenRouter', () => {
    expect(resolveOpenRouterModel('gemini-2.5-flash')).toBe(
      'google/gemini-2.5-flash',
    );
  });

  it('keeps already-qualified OpenRouter model IDs unchanged', () => {
    expect(resolveOpenRouterModel('google/gemini-3.1-pro-preview')).toBe(
      'google/gemini-3.1-pro-preview',
    );
  });

  it('uses a supported OpenRouter model for compression utility calls', () => {
    expect(
      resolveOpenRouterModel('gemini-3-pro-preview', 'utility_compressor'),
    ).toBe('google/gemini-3-flash-preview');
  });

  it('allows overriding the OpenRouter compression model', () => {
    vi.stubEnv('OPENROUTER_COMPRESSION_MODEL', 'google/gemini-2.5-flash');

    expect(
      resolveOpenRouterModel('gemini-3-pro-preview', 'utility_compressor'),
    ).toBe('google/gemini-2.5-flash');
  });
});

describe('convertToOpenAIFormat — tool-result forwarding', () => {
  // gemini-cli packs native tool results (read_file/write_file/...) as their
  // own turn with role 'user', not role 'function'. The converter must still
  // emit an OpenAI role:'tool' message carrying the result content, or the
  // model never sees the tool output and hallucinates (e.g. file contents).
  it('forwards a functionResponse delivered under role "user"', () => {
    const msgs = convertToOpenAIFormat({
      contents: [
        {
          role: 'model',
          parts: [
            {
              functionCall: { id: 'call_1', name: 'read_file', args: {} },
            },
          ],
        },
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                id: 'call_1',
                name: 'read_file',
                response: { output: 'hello world' },
              },
            },
          ],
        },
      ],
    } as never);

    const toolMsg = msgs.find((m) => m.role === 'tool');
    expect(toolMsg).toBeDefined();
    expect(toolMsg).toMatchObject({
      role: 'tool',
      tool_call_id: 'call_1',
    });
    expect(String(toolMsg?.content)).toContain('hello world');
  });

  it('still forwards a functionResponse delivered under role "function"', () => {
    const msgs = convertToOpenAIFormat({
      contents: [
        {
          role: 'function',
          parts: [
            {
              functionResponse: {
                id: 'call_2',
                name: 'read_file',
                response: { output: 'alpha' },
              },
            },
          ],
        },
      ],
    } as never);

    const toolMsg = msgs.find((m) => m.role === 'tool');
    expect(toolMsg).toBeDefined();
    expect(toolMsg?.tool_call_id).toBe('call_2');
    expect(String(toolMsg?.content)).toContain('alpha');
  });

  // MCP tools (e.g. the CUA `screenshot` tool) return an image as an
  // `inlineData` Part. OpenAI role:'tool' messages can only carry a string, so
  // the converter must re-emit the image as a following role:'user' message
  // with `image_url` content. Without this the model is blind to every
  // screenshot over OpenRouter and hallucinates GUI contents.
  const expectImageUrl = (msg: unknown, data: string) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parts = (msg as any).content as any[];
    expect(Array.isArray(parts)).toBe(true);
    const img = parts.find((p) => p.type === 'image_url');
    expect(img).toBeDefined();
    expect(img.image_url.url).toContain('data:image/png;base64,');
    expect(img.image_url.url).toContain(data);
  };

  it('forwards a sibling inlineData image alongside a tool result', () => {
    // supportsMultimodalFunctionResponse is false for gemini-3.5-flash, so the
    // screenshot image lands as a sibling Part next to the functionResponse.
    const msgs = convertToOpenAIFormat({
      contents: [
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                id: 'call_1',
                name: 'screenshot',
                response: { output: 'Screenshot captured' },
              },
            },
            { inlineData: { mimeType: 'image/png', data: 'AAAASIBLING' } },
          ],
        },
      ],
    } as never);

    const toolMsg = msgs.find((m) => m.role === 'tool');
    expect(toolMsg?.tool_call_id).toBe('call_1');
    const userMsg = msgs.find(
      (m) => m.role === 'user' && Array.isArray(m.content),
    );
    expectImageUrl(userMsg, 'AAAASIBLING');
  });

  it('forwards a nested multimodal-functionResponse image', () => {
    const msgs = convertToOpenAIFormat({
      contents: [
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                id: 'call_2',
                name: 'screenshot',
                response: { output: 'Screenshot captured' },
                // nested by convertToFunctionResponse when the model supports it
                parts: [
                  { inlineData: { mimeType: 'image/png', data: 'AAAANESTED' } },
                ],
              },
            },
          ],
        },
      ],
    } as never);

    const toolMsg = msgs.find((m) => m.role === 'tool');
    expect(toolMsg?.tool_call_id).toBe('call_2');
    const userMsg = msgs.find(
      (m) => m.role === 'user' && Array.isArray(m.content),
    );
    expectImageUrl(userMsg, 'AAAANESTED');
  });

  it('forwards an inline image on a plain user turn', () => {
    const msgs = convertToOpenAIFormat({
      contents: [
        {
          role: 'user',
          parts: [
            { text: 'look at this' },
            { inlineData: { mimeType: 'image/png', data: 'AAAAUSER' } },
          ],
        },
      ],
    } as never);

    const userMsg = msgs.find(
      (m) => m.role === 'user' && Array.isArray(m.content),
    );
    expectImageUrl(userMsg, 'AAAAUSER');
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parts = (userMsg as any).content as any[];
    expect(
      parts.some((p) => p.type === 'text' && p.text === 'look at this'),
    ).toBe(true);
  });
});
