import { apiRequest } from './client';

export interface EyesStatus {
  running?: boolean;
  fps?: number;
  frames?: number;
  [key: string]: unknown;
}

export interface EyesSceneIdentity {
  trace_id?: string;
  identity_id?: string;
  label?: string;
  strength?: number;
  n_observations?: number;
  last_seen_ms?: number;
}

export interface EyesSceneResponse {
  identities?: EyesSceneIdentity[];
  [key: string]: unknown;
}

export async function getEyesStatus(): Promise<EyesStatus> {
  return apiRequest<EyesStatus>('/eyes/status');
}

export async function getEyesScene(topK = 20): Promise<EyesSceneResponse> {
  return apiRequest<EyesSceneResponse>(`/eyes/scene?top_k=${topK}`);
}

export async function startEyes(): Promise<Record<string, unknown>> {
  return apiRequest<Record<string, unknown>>('/eyes/start', { method: 'POST' });
}
