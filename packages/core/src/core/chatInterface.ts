/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content, GenerateContentResponse, SendMessageParameters, Tool } from "@google/genai";
import { StructuredError } from "./turn.js";

export interface ChatInterface {
  addHistory(content: Content): void
  getHistory(curated?: boolean): Content[]
  setHistory(history: Content[]): void
  setTools(tools: Tool[]): void
  sendMessage(
    params: SendMessageParameters,
    prompt_id: string,
  ): Promise<GenerateContentResponse>
  sendMessageStream(
    params: SendMessageParameters,
    prompt_id: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>>
  maybeIncludeSchemaDepthContext(error: StructuredError): Promise<void>
}
