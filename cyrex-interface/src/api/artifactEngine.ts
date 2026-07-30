// Artifact Engine API Client. Typed client for all artifact engine endpoints.

import {
  ArtifactBundle,
  VoiceQueryRequest,
  VoiceQueryResponse,
  Provenance,
  Citation,
  CorrectionRequest,
  CorrectionResponse,
  ProvenanceResponse,
  PredictionRecord,
} from '../types/artifactEngine';
import { apiRequest, apiUpload } from './client';

const ARTIFACTS = '/artifacts';
const RECKONING = '/reckoning';

// Response shape
export interface ReckoningResponse {
  document_id: string;
  records: PredictionRecord[];
  anomalous_count: number;
  novel_count: number;
}

// Upload a document and run the full pipeline. Returns ArtifactBundle.
export async function uploadArtifact(
  file: File,
  documentId?: string,
): Promise<ArtifactBundle> {
  const formData = new FormData();
  formData.append('file', file);
  if (documentId) formData.append('document_id', documentId);
  const data = await apiUpload<{ artifact: ArtifactBundle }>(`${ARTIFACTS}/upload`, formData);
  return data.artifact;
}

// Fetch an artifact bundle by ID.
export async function getArtifact(artifactId: string): Promise<ArtifactBundle> {
  const data = await apiRequest<{ artifact: ArtifactBundle }>(`${ARTIFACTS}/${artifactId}`);
  return data.artifact;
}

export async function getProvenance(artifactId: string): Promise<ProvenanceResponse> {
  return apiRequest<ProvenanceResponse>(`${ARTIFACTS}/${artifactId}/provenance`);
}

// Submit a human correction for a field in an artifact.
export async function submitCorrection(
  artifactId: string,
  correction: CorrectionRequest,
): Promise<CorrectionResponse> {
  return apiRequest<CorrectionResponse>(`${ARTIFACTS}/${artifactId}/corrections`, {
    method: 'POST',
    body: JSON.stringify(correction),
  });
}

// Send a voice query — returns verbatim cited spans or a confession.
// Optional audio_b64 → STT; response may include TTS audio_b64 from deepiri-speech.
export type VoiceQueryResult = VoiceQueryResponse & {
  spoken_text?: string | null;
  audio_b64?: string | null;
  audio_mime_type?: string | null;
  speech?: Record<string, unknown> | null;
  question_used?: string | null;
};

export async function voiceQuery(
  request: VoiceQueryRequest & {
    audio_b64?: string;
    audio_mime_type?: string;
    synthesize_audio?: boolean;
  },
): Promise<VoiceQueryResult> {
  const data = await apiRequest<{
    response: VoiceQueryResponse;
    spoken_text?: string;
    audio_b64?: string;
    audio_mime_type?: string;
    speech?: Record<string, unknown>;
    question_used?: string;
  }>(`${ARTIFACTS}/voice/query`, {
    method: 'POST',
    body: JSON.stringify({
      synthesize_audio: true,
      ...request,
    }),
  });
  return {
    ...data.response,
    spoken_text: data.spoken_text,
    audio_b64: data.audio_b64,
    audio_mime_type: data.audio_mime_type,
    speech: data.speech,
    question_used: data.question_used,
  };
}

export async function speechHealth(): Promise<{
  ok: boolean;
  speech?: Record<string, unknown>;
  error?: string;
}> {
  return apiRequest(`${ARTIFACTS}/voice/speech-health`);
}

// Fetch dead-reckoning prediction records
export async function getReckoning(documentId: string): Promise<ReckoningResponse> {
  return apiRequest<ReckoningResponse>(`${RECKONING}/${documentId}`);
}