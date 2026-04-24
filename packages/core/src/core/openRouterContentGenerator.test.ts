/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { resolveOpenRouterModel } from './openRouterContentGenerator.js';

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
