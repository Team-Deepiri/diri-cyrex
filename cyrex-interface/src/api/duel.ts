import { FieldDiscrepancy } from '../types/artifactEngine';
import { apiRequest } from './client';

export interface DuelFieldRow {
  field_name: string;
  agent_a_value: unknown;
  agent_b_value: unknown;
  agent_a_confidence: number | null;
  agent_b_confidence: number | null;
  is_disagreement: boolean;
}

export interface DuelArenaResponse {
  document_id: string;
  artifact_id?: string | null;
  agent_a_id: string;
  agent_b_id: string;
  fields: DuelFieldRow[];
  disagreements: FieldDiscrepancy[];
  resolution_status: string;
}

export async function getDocumentDuel(documentId: string): Promise<DuelArenaResponse> {
  return apiRequest<DuelArenaResponse>(`/duel/${documentId}`);
}
