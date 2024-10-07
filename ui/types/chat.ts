export interface Message {
    id: string;
    role: string;
    content: string;
  }
  
  export type Model = {
    id: string;
    name: string;
  };
  
  export interface ArtifactContent {
    id: string;
    type: 'code' | 'html' | 'text' | 'log';
    content: string;
    language?: string;
    name: string;
  }

  export interface ParsedResponse {
    response: string;
  }
  
  export interface MessageContentState {
    thinking: boolean;
    parsed: ParsedResponse;
    error: boolean;
  }