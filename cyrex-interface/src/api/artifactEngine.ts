// Artifact Engine API Client. Typed client for all artifact engine endpoints.

import {
  ArtifactBundle,
  VoiceQueryRequest,
  VoiceQueryResponse,
  Provenance,
  Citation,
  CorrectionRequest,
  CorrectionResponse,
  ProvenanceResponse
} from '../types/artifactEngine';
import { apiRequest, apiUpload } from './client';

const ARTIFACTS = '/artifacts';

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

export interface ArtifactGraphEdge {
  from: string;
  to: string;
  ref_type: string;
}

export interface ArtifactGraphResponse {
  success: boolean;
  artifact_id: string;
  nodes: ArtifactBundle[];
  edges: ArtifactGraphEdge[];
}

export async function getArtifactGraph(
  artifactId: string,
  hops = 2,
): Promise<ArtifactGraphResponse> {
  return apiRequest<ArtifactGraphResponse>(`${ARTIFACTS}/${artifactId}/graph?hops=${hops}`);
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
export async function voiceQuery(
  request: VoiceQueryRequest,
): Promise<VoiceQueryResponse> {
  const data = await apiRequest<{ response: VoiceQueryResponse }>(`${ARTIFACTS}/voice/query`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
  return data.response;
}