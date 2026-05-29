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
});
