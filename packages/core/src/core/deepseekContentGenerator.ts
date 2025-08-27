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
    const tools = this.convertToolsToOpenAI(request.config?.tools);

    const response = await this.openai.chat.completions.create({
      model: this.model,
      messages,
      temperature: request.config?.temperature,
      max_tokens: request.config?.maxOutputTokens,
      stream: false,
      ...(tools.length > 0 && { tools, tool_choice: 'auto' }),
    });

    return this.convertToGeminiResponse(response);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const contentArray = this.extractContentArray(request.contents);
    const messages = this.convertToOpenAIMessages(contentArray);
    const tools = this.convertToolsToOpenAI(request.config?.tools);

    const stream = await this.openai.chat.completions.create({
      model: this.model,
      messages,
      temperature: request.config?.temperature,
      max_tokens: request.config?.maxOutputTokens,
      stream: true,
      ...(tools.length > 0 && { tools, tool_choice: 'auto' }),
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
    const toolCalls: any[] = [];
    
    for await (const chunk of stream) {
      const choice = chunk.choices[0];
      const delta = choice?.delta;
      const content = delta?.content;
      const finishReason = choice?.finish_reason;
      const deltaToolCalls = delta?.tool_calls;
      
      if (content) {
        buffer += content;
      }
      
      if (deltaToolCalls) {
        for (const deltaToolCall of deltaToolCalls) {
          const index = deltaToolCall.index || 0;
          if (!toolCalls[index]) {
            toolCalls[index] = {
              id: deltaToolCall.id || '',
              type: 'function',
              function: { name: '', arguments: '' }
            };
          }
          
          if (deltaToolCall.function?.name) {
            toolCalls[index].function.name += deltaToolCall.function.name;
          }
          if (deltaToolCall.function?.arguments) {
            toolCalls[index].function.arguments += deltaToolCall.function.arguments;
          }
        }
      }
      
      if (finishReason) {
        lastFinishReason = finishReason;
      }
      
      // Yield buffered content in reasonable chunks
      const shouldFlush = 
        buffer.length >= 100 || // Buffer getting large (increased threshold)
        (buffer.length > 0 && (
          content?.includes('\n\n') || // Paragraph breaks only
          finishReason // Always flush on finish
        ));
      
      if (shouldFlush && buffer.length > 0) {
        const parts: any[] = [{ text: buffer }];
        
        const geminiResponse = new GenerateContentResponse();
        geminiResponse.candidates = [{
          content: { parts, role: 'model' },
          finishReason: this.convertFinishReason(lastFinishReason) as FinishReason,
          index: 0,
        }];
        yield geminiResponse;
        buffer = '';
      }
      
      // Handle tool calls at the end of the stream
      if (finishReason && toolCalls.length > 0) {
        const parts: any[] = [];
        
        if (buffer) {
          parts.push({ text: buffer });
        }
        
        for (const toolCall of toolCalls) {
          if (toolCall.function?.name) {
            parts.push({
              functionCall: {
                name: toolCall.function.name,
                args: JSON.parse(toolCall.function.arguments || '{}'),
              },
            });
          }
        }
        
        const geminiResponse = new GenerateContentResponse();
        geminiResponse.candidates = [{
          content: { parts, role: 'model' },
          finishReason: this.convertFinishReason(finishReason) as FinishReason,
          index: 0,
        }];
        yield geminiResponse;
        return;
      }
      
      // If we have a finish reason but no more content or tools, yield final chunk
      if (finishReason && buffer.length === 0 && toolCalls.length === 0) {
        const geminiResponse = new GenerateContentResponse();
        geminiResponse.candidates = [{
          content: { parts: [], role: 'model' },
          finishReason: this.convertFinishReason(finishReason) as FinishReason,
          index: 0,
        }];
        yield geminiResponse;
      }
    }
    
    // Flush any remaining buffer
    if (buffer.length > 0) {
      const geminiResponse = new GenerateContentResponse();
      geminiResponse.candidates = [{
        content: { parts: [{ text: buffer }], role: 'model' },
        finishReason: this.convertFinishReason(lastFinishReason) as FinishReason,
        index: 0,
      }];
      yield geminiResponse;
    }
  }

  private convertToOpenAIMessages(contents: Content[]): OpenAI.Chat.ChatCompletionMessageParam[] {
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];
    const toolCallIdMap = new Map<string, string>(); // Map function name+args to call ID
    const functionResponseExists = new Set<string>(); // Track which function calls have responses
    
    // First pass: collect all function calls and track which have responses
    for (const content of contents) {
      for (const part of content.parts || []) {
        if (part.functionCall) {
          const key = `${part.functionCall.name}:${JSON.stringify(part.functionCall.args || {})}`;
          if (!toolCallIdMap.has(key)) {
            toolCallIdMap.set(key, `call_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
          }
        } else if (part.functionResponse) {
          functionResponseExists.add(part.functionResponse.name || '');
        }
      }
    }
    
    // Second pass: convert to OpenAI format
    for (const content of contents) {
      const role = content.role === 'user' ? 'user' : content.role === 'model' ? 'assistant' : 'system';
      
      const functionCalls: any[] = [];
      const functionResponses: any[] = [];
      let textContent = '';
      
      for (const part of content.parts || []) {
        if (part.text) {
          textContent += part.text;
        } else if (part.functionCall) {
          // Only include function calls that have corresponding responses
          if (functionResponseExists.has(part.functionCall.name || '')) {
            const key = `${part.functionCall.name}:${JSON.stringify(part.functionCall.args || {})}`;
            const callId = toolCallIdMap.get(key)!;
            functionCalls.push({
              id: callId,
              type: 'function',
              function: {
                name: part.functionCall.name,
                arguments: JSON.stringify(part.functionCall.args || {}),
              },
            });
          } else {
            // Convert incomplete function calls to text
            textContent += `[Function call: ${part.functionCall.name}(${JSON.stringify(part.functionCall.args)})]`;
          }
        } else if (part.functionResponse) {
          // Find the matching function call for this response
          let matchingCallId = null;
          for (const [key, callId] of toolCallIdMap.entries()) {
            const [funcName] = key.split(':');
            if (funcName === part.functionResponse.name) {
              matchingCallId = callId;
              break;
            }
          }
          
          if (matchingCallId) {
            functionResponses.push({
              tool_call_id: matchingCallId,
              role: 'tool' as const,
              name: part.functionResponse.name,
              content: JSON.stringify(part.functionResponse.response || {}),
            });
          }
        }
      }
      
      if (functionCalls.length > 0) {
        messages.push({
          role: role as 'assistant',
          content: textContent || null,
          tool_calls: functionCalls,
        });
      } else if (functionResponses.length > 0) {
        for (const response of functionResponses) {
          messages.push(response);
        }
      } else if (textContent) {
        messages.push({
          role: role as 'user' | 'assistant' | 'system',
          content: textContent,
        });
      }
    }
    
    return messages;
  }

  private convertToGeminiResponse(response: OpenAI.Chat.ChatCompletion): GenerateContentResponse {
    const choice = response.choices[0];
    const message = choice?.message;
    const content = message?.content || '';
    const toolCalls = message?.tool_calls || [];

    const parts: any[] = [];
    
    if (content) {
      parts.push({ text: content });
    }
    
    for (const toolCall of toolCalls) {
      if (toolCall.type === 'function') {
        parts.push({
          functionCall: {
            name: toolCall.function.name,
            args: JSON.parse(toolCall.function.arguments || '{}'),
          },
        });
      }
    }

    const geminiResponse = new GenerateContentResponse();
    geminiResponse.candidates = [{
      content: {
        parts,
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

  private convertToolsToOpenAI(tools?: any[]): any[] {
    if (!tools || !Array.isArray(tools)) {
      return [];
    }

    const openaiTools: any[] = [];
    
    for (const tool of tools) {
      if (tool.functionDeclarations && Array.isArray(tool.functionDeclarations)) {
        // Handle Gemini format: [{ functionDeclarations: [...] }]
        for (const funcDecl of tool.functionDeclarations) {
          if (funcDecl.name) {
            openaiTools.push({
              type: 'function',
              function: {
                name: funcDecl.name,
                description: funcDecl.description || '',
                parameters: funcDecl.parameters || { type: 'object', properties: {} },
              },
            });
          }
        }
      } else if (tool.name) {
        // Handle direct function declaration format
        openaiTools.push({
          type: 'function',
          function: {
            name: tool.name,
            description: tool.description || '',
            parameters: tool.parameters || { type: 'object', properties: {} },
          },
        });
      }
    }

    return openaiTools;
  }

  private convertFinishReason(finishReason: string | null | undefined): string {
    if (finishReason === null || finishReason === undefined) {
      return 'FINISH_REASON_UNSPECIFIED';
    }
    
    // Log unknown finish reasons for debugging
    const knownReasons = ['stop', 'length', 'max_tokens', 'content_filter', 'tool_calls'];
    if (!knownReasons.includes(finishReason)) {
      console.warn(`DeepSeek returned unknown finish_reason: "${finishReason}"`);
    }
    
    switch (finishReason) {
      case 'stop':
        return 'STOP';
      case 'length':
      case 'max_tokens':
        return 'MAX_TOKENS';
      case 'content_filter':
        return 'SAFETY';
      case 'tool_calls':
      case 'function_call':
        return 'STOP'; // Tool calls should be treated as normal completion
      default:
        // For unknown reasons, default to STOP instead of OTHER to avoid warnings
        return 'STOP';
    }
  }
}
