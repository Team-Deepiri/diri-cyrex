// Artifact Engine API Client. Typed client for all artifact engine endpoints.

import {
  ArtifactBundle,
  VoiceQueryRequest,
  VoiceQueryResponse,
  Provenance,
  Citation,
} from '../types/artifactEngine';

const BASE_URL = '/api/v1/artifacts';

// Upload a document and run the full pipeline. Returns ArtifactBundle.
export async function uploadArtifact(
  file: File,
  documentId?: string,
): Promise<ArtifactBundle> {
  const formData = new FormData();
  formData.append('file', file);
  if (documentId) formData.append('document_id', documentId);

  const res = await fetch(`${BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  const data = await res.json();
  return data.artifact as ArtifactBundle;
}

// Fetch an artifact bundle by ID.
export async function getArtifact(artifactId: string): Promise<ArtifactBundle> {
  const res = await fetch(`${BASE_URL}/${artifactId}`);
  if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
  const data = await res.json();
  return data.artifact as ArtifactBundle;
}

// Walk the artifact graph backward to source PDF spans.
export interface ProvenanceResponse {
  artifact_id: string;
  provenance: Provenance;
  citations: Citation[];
}

export async function getProvenance(artifactId: string): Promise<ProvenanceResponse> {
  const res = await fetch(`${BASE_URL}/${artifactId}/provenance`);
  if (!res.ok) throw new Error(`Provenance fetch failed: ${res.status}`);
  return res.json();
}

// Submit a human correction for a field in an artifact.
export interface CorrectionRequest {
  field_name: string;
  corrected_value: any;
  corrected_citation: Citation;
  actor_id: string;
}

export interface CorrectionResponse {
  artifact_id: string;
  field_name: string;
  corrected_value: any;
  submitted_at: string;
}

export async function submitCorrection(
  artifactId: string,
  correction: CorrectionRequest,
): Promise<CorrectionResponse> {
  const res = await fetch(`${BASE_URL}/${artifactId}/corrections`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(correction),
  });
  if (!res.ok) throw new Error(`Correction failed: ${res.status}`);
  return res.json();
}

// Send a voice query — returns verbatim cited spans or a confession.
export async function voiceQuery(
  request: VoiceQueryRequest,
): Promise<VoiceQueryResponse> {
  const res = await fetch(`${BASE_URL}/voice/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(`Voice query failed: ${res.status}`);
  return res.json();
}