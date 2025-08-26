/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';
import {
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  Content,
  FinishReason,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';

export class DeepSeekContentGenerator implements ContentGenerator {
  constructor(
    private openai: OpenAI,
    private model: string,
  ) { }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const contentArray = this.extractContentArray(request.contents);
    const messages = this.convertToOpenAIMessages(contentArray);

    const response = await this.openai.chat.completions.create({
      model: this.model,
      messages,
      temperature: request.config?.temperature,
      max_tokens: request.config?.maxOutputTokens,
      stream: false,
    });

    return this.convertToGeminiResponse(response);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const contentArray = this.extractContentArray(request.contents);
    const messages = this.convertToOpenAIMessages(contentArray);

    const stream = await this.openai.chat.completions.create({
      model: this.model,
      messages,
      temperature: request.config?.temperature,
      max_tokens: request.config?.maxOutputTokens,
      stream: true,
    });

    return this.convertStreamToGenerator(stream);
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    // DeepSeek doesn't have a direct token counting API, so we estimate
    const contentArray = this.extractContentArray(request.contents);
    const messages = this.convertToOpenAIMessages(contentArray);
    const text = messages.map(m => m.content).join(' ');
    const estimatedTokens = Math.ceil(text.length / 4); // Rough estimation

    return {
      totalTokens: estimatedTokens,
    };
  }

  async embedContent(_request: EmbedContentParameters): Promise<EmbedContentResponse> {
    throw new Error('DeepSeek does not support embedding generation');
  }

  private extractContentArray(contents: unknown): Content[] {
    if (Array.isArray(contents)) {
      return contents.filter((item): item is Content =>
        typeof item === 'object' && item !== null && 'parts' in item
      );
    }
    return [];
  }

  private async *convertStreamToGenerator(stream: AsyncIterable<OpenAI.Chat.ChatCompletionChunk>): AsyncGenerator<GenerateContentResponse> {
    let buffer = '';
    let lastFinishReason: string | null | undefined = undefined;
    
    for await (const chunk of stream) {
      const choice = chunk.choices[0];
      const content = choice?.delta?.content;
      const finishReason = choice?.finish_reason;
      
      console.error('xxx-deepseek-chunk:', JSON.stringify({
        content,
        finishReason,
        bufferLength: buffer.length
      }));
      
      if (content) {
        buffer += content;
      }
      
      if (finishReason) {
        lastFinishReason = finishReason;
      }
      
      // Yield buffered content in reasonable chunks
      // Only flush when buffer gets large, we hit newlines, or stream finishes
      const shouldFlush = 
        buffer.length >= 100 || // Buffer getting large (increased threshold)
        (buffer.length > 0 && (
          content?.includes('\n\n') || // Paragraph breaks only
          finishReason // Always flush on finish
        ));
      
      if (shouldFlush && buffer.length > 0) {
        console.error('xxx-deepseek-yielding:', JSON.stringify({
          text: buffer,
          reason: finishReason ? 'finish' : 'breakpoint'
        }));
        const geminiResponse = new GenerateContentResponse();
        geminiResponse.candidates = [{
          content: {
            parts: [{ text: buffer }],
            role: 'model',
          },
          finishReason: this.convertFinishReason(lastFinishReason) as FinishReason,
          index: 0,
        }];
        yield geminiResponse;
        buffer = '';
      }
      
      // If we have a finish reason but no more content, yield final chunk
      if (finishReason && buffer.length === 0) {
        console.error('xxx-deepseek-finish-only');
        const geminiResponse = new GenerateContentResponse();
        geminiResponse.candidates = [{
          content: {
            parts: [],
            role: 'model',
          },
          finishReason: this.convertFinishReason(finishReason) as FinishReason,
          index: 0,
        }];
        yield geminiResponse;
      }
    }
    
    // Flush any remaining buffer
    if (buffer.length > 0) {
      console.error('xxx-deepseek-final-flush:', JSON.stringify({ text: buffer }));
      const geminiResponse = new GenerateContentResponse();
      geminiResponse.candidates = [{
        content: {
          parts: [{ text: buffer }],
          role: 'model',
        },
        finishReason: this.convertFinishReason(lastFinishReason) as FinishReason,
        index: 0,
      }];
      yield geminiResponse;
    }
  }

  private convertToOpenAIMessages(contents: Content[]): OpenAI.Chat.ChatCompletionMessageParam[] {
    return contents.map(content => {
      const role = content.role === 'user' ? 'user' : content.role === 'model' ? 'assistant' : 'system';
      const textContent = content.parts?.map(part => {
        if (part.text) return part.text;
        if (part.functionCall) {
          return `Function call: ${part.functionCall.name} with args: ${JSON.stringify(part.functionCall.args)}`;
        }
        if (part.functionResponse) {
          return `Function response: ${JSON.stringify(part.functionResponse.response)}`;
        }
        return '';
      }).join('\n') || '';

      return {
        role: role as 'user' | 'assistant' | 'system',
        content: textContent,
      };
    });
  }

  private convertToGeminiResponse(response: OpenAI.Chat.ChatCompletion): GenerateContentResponse {
    const choice = response.choices[0];
    const content = choice?.message?.content || '';

    const geminiResponse = new GenerateContentResponse();
    geminiResponse.candidates = [{
      content: {
        parts: [{ text: content }],
        role: 'model',
      },
      finishReason: this.convertFinishReason(choice?.finish_reason) as FinishReason,
      index: 0,
    }];
    geminiResponse.usageMetadata = {
      promptTokenCount: response.usage?.prompt_tokens || 0,
      candidatesTokenCount: response.usage?.completion_tokens || 0,
      totalTokenCount: response.usage?.total_tokens || 0,
    };
    return geminiResponse;
  }

  private convertFinishReason(finishReason: string | null | undefined): string {
    if (finishReason === null || finishReason === undefined) {
      return 'FINISH_REASON_UNSPECIFIED';
    }
    switch (finishReason) {
      case 'stop':
        return 'STOP';
      case 'length':
        return 'MAX_TOKENS';
      case 'content_filter':
        return 'SAFETY';
      default:
        return 'OTHER';
    }
  }
}
