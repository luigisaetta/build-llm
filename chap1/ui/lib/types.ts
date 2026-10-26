/**
 * Types of the JSON exchanged with the Python backend (chap1/api.py).
 * Field names match the Python models, so they use snake_case.
 */

export type EncodingInfo = {
  name: string;
  description: string;
};

export type EncodingsResponse = {
  default: string;
  encodings: EncodingInfo[];
};

export type Token = {
  id: number;
  text: string;
  byte_values: number[];
  is_partial_character: boolean;
};

export type TokenizeResponse = {
  encoding: string;
  vocabulary_size: number;
  tokens: Token[];
  token_count: number;
  character_count: number;
  word_count: number;
  characters_per_token: number;
};

export type FetchUrlResponse = {
  url: string;
  title: string;
  content_type: string;
  text: string;
  truncated: boolean;
};

/** The error body returned by the backend and by the proxy routes. */
export type ErrorResponse = {
  detail: string | unknown[];
};
