/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SessionMetrics } from '../telemetry/uiTelemetry.js';

export enum OutputFormat {
  TEXT = 'text',
  JSON = 'json',
  STREAM_JSON = 'stream-json',
}

export interface JsonError {
  type: string;
  message: string;
  code?: string | number;
}

export interface JsonOutput {
  session_id?: string;
  response?: string;
  stats?: SessionMetrics;
  error?: JsonError;
}

// Streaming JSON event types
export enum JsonStreamEventType {
  INIT = 'init',
  MESSAGE = 'message',
  TOOL_USE = 'tool_use',
  TOOL_RESULT = 'tool_result',
  ERROR = 'error',
  RESULT = 'result',
}

export interface BaseJsonStreamEvent {
  type: JsonStreamEventType;
  timestamp: string;
}

export interface InitEvent extends BaseJsonStreamEvent {
  type: JsonStreamEventType.INIT;
  session_id: string;
  model: string;
}

export interface MessageEvent extends BaseJsonStreamEvent {
  type: JsonStreamEventType.MESSAGE;
  role: 'user' | 'assistant';
  content: string;
  delta?: boolean;
  // Provider response id for this assistant turn. Over OpenRouter this is the
  // `gen-...` generation id, usable to look up cost/usage via OpenRouter's
  // generation API. Absent for user messages and for providers that don't
  // surface a response id.
  response_id?: string;
}

export interface ToolUseEvent extends BaseJsonStreamEvent {
  type: JsonStreamEventType.TOOL_USE;
  tool_name: string;
  tool_id: string;
  parameters: Record<string, unknown>;
}

export interface ToolResultEvent extends BaseJsonStreamEvent {
  type: JsonStreamEventType.TOOL_RESULT;
  tool_id: string;
  status: 'success' | 'error';
  output?: string;
  /**
   * Inline media (image/pdf/...) the tool returned to the model, e.g. the CUA
   * `screenshot` tool's PNG. `output` carries only the text placeholder
   * (`[Image: image/png]`), so without this the raw bytes never reach the
   * transcript and downstream screenshot persistence has nothing to extract.
   * Each entry mirrors a genai `media` Part: base64 `data` (inline) or `uri`.
   */
  media?: Array<{
    mime_type: string;
    data?: string;
    uri?: string;
  }>;
  error?: {
    type: string;
    message: string;
  };
}

export interface ErrorEvent extends BaseJsonStreamEvent {
  type: JsonStreamEventType.ERROR;
  severity: 'warning' | 'error';
  message: string;
}

export interface ModelStreamStats {
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  cached: number;
  input: number;
}

export interface StreamStats {
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  // Breakdown of input_tokens
  cached: number;
  input: number;
  duration_ms: number;
  tool_calls: number;
  models: Record<string, ModelStreamStats>;
}

export interface ResultEvent extends BaseJsonStreamEvent {
  type: JsonStreamEventType.RESULT;
  status: 'success' | 'error';
  error?: {
    type: string;
    message: string;
  };
  stats?: StreamStats;
}

export type JsonStreamEvent =
  | InitEvent
  | MessageEvent
  | ToolUseEvent
  | ToolResultEvent
  | ErrorEvent
  | ResultEvent;
