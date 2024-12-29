// HttpAgent.ts

import { Anthropic } from '@anthropic-ai/sdk';
import { OpenAPIV3 } from 'openapi-types';
import * as fs from 'fs';
import * as yaml from 'js-yaml';

interface SchemaEndpoint {
  path: string;
  method: string;
  description: string;
  parameters: OpenAPIV3.ParameterObject[];
  requestBody?: OpenAPIV3.RequestBodyObject;
  responses: OpenAPIV3.ResponsesObject;
}

interface AnalyzedInput {
  keywords: string[];
  entities: Map<string, string>;
  potentialEndpoints: SchemaEndpoint[];
}

export class HttpAgent {
  private schema: OpenAPIV3.Document;
  private anthropicClient: Anthropic;
  private endpoints: SchemaEndpoint[];

  /**
   * Initialize the HttpAgent with a schema file and Anthropic API key
   * @param schemaFile Path to the OpenAPI schema file
   * @param anthropicApiKey Anthropic API key for LLM access
   */
  constructor(schemaFile: string, anthropicApiKey: string) {
    if (!fs.existsSync(schemaFile)) {
      throw new Error('Schema file not found');
    }

    if (!anthropicApiKey) {
      throw new Error('Anthropic API key is required');
    }

    const fileContent = fs.readFileSync(schemaFile, 'utf-8');
    this.schema = yaml.load(fileContent) as OpenAPIV3.Document;
    this.anthropicClient = new Anthropic({ apiKey: anthropicApiKey });
    this.endpoints = this.parseSchema();
  }

  /**
   * Parse the OpenAPI schema and extract relevant information
   * @returns Array of parsed endpoints
   */
  private parseSchema(): SchemaEndpoint[] {
    const endpoints: SchemaEndpoint[] = [];

    for (const [path, pathItem] of Object.entries(this.schema.paths)) {
      for (const [method, operation] of Object.entries(pathItem)) {
        if (operation) {
          endpoints.push({
            path,
            method: method.toUpperCase(),
            description: operation.description || '',
            parameters:
              (operation.parameters as OpenAPIV3.ParameterObject[]) || [],
            requestBody: operation.requestBody as OpenAPIV3.RequestBodyObject,
            responses: operation.responses,
          });
        }
      }
    }

    return endpoints;
  }

  /**
   * Analyze user input to extract relevant information
   * @param userInput User's natural language input
   * @returns Analyzed input with extracted information
   */
  private analyzeUserInput(userInput: string): AnalyzedInput {
    const keywords = userInput
      .toLowerCase()
      .split(' ')
      .filter(
        (word) =>
          word.length > 3 &&
          !['what', 'how', 'the', 'and', 'for'].includes(word)
      );

    const entities = new Map<string, string>();
    const potentialEndpoints = this.endpoints.filter((endpoint) => {
      return keywords.some(
        (keyword) =>
          endpoint.description.toLowerCase().includes(keyword) ||
          endpoint.path.toLowerCase().includes(keyword)
      );
    });

    return {
      keywords,
      entities,
      potentialEndpoints,
    };
  }

  /**
   * Construct a prompt for the Anthropic LLM
   * @param userInput Original user input
   * @param analyzedInput Analyzed input data
   * @returns Constructed prompt string
   */
  private constructPrompt(
    userInput: string,
    analyzedInput: AnalyzedInput
  ): string {
    const endpointDescriptions = analyzedInput.potentialEndpoints
      .map(
        (endpoint) => `
        Path: ${endpoint.path}
        Method: ${endpoint.method}
        Description: ${endpoint.description}
      `
      )
      .join('\n');

    return `
      Context: I have an HTTP API with the following endpoints:
      ${endpointDescriptions}

      User Input: "${userInput}"

      Based on the user input and available endpoints, please:
      1. Identify the most relevant endpoint
      2. Explain why this endpoint matches the user's intent
      3. Provide a structured response with the endpoint details

      Response Format:
      {
        "endpoint": "<path>",
        "method": "<method>",
        "confidence": <0-1>,
        "explanation": "<explanation>"
      }
    `;
  }

  /**
   * Query the Anthropic LLM with the constructed prompt
   * @param prompt Constructed prompt
   * @returns LLM response
   */
  private async queryAnthropicLLM(prompt: string): Promise<string> {
    try {
      const response = await this.anthropicClient.messages.create({
        model: 'claude-2',
        max_tokens: 1000,
        messages: [{ role: 'user', content: prompt }],
      });

      return response.content[0].text;
    } catch (error) {
      throw new Error(`Failed to query Anthropic LLM: ${error.message}`);
    }
  }

  /**
   * Main method to infer intent from user input
   * @param userInput User's natural language input
   * @returns Inferred intent as a structured response
   */
  public async inferIntent(userInput: string): Promise<any> {
    if (!userInput || userInput.trim().length === 0) {
      throw new Error('User input is required');
    }

    try {
      const analyzedInput = this.analyzeUserInput(userInput);
      const prompt = this.constructPrompt(userInput, analyzedInput);
      const llmResponse = await this.queryAnthropicLLM(prompt);

      // Parse the JSON response from the LLM
      return JSON.parse(llmResponse);
    } catch (error) {
      throw new Error(`Failed to infer intent: ${error.message}`);
    }
  }
}
