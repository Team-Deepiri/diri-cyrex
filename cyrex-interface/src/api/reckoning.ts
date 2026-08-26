import { PredictionRecord } from '../types/artifactEngine';
import { apiRequest } from './client';

export interface ReckoningResponse {
  document_id: string;
  records: PredictionRecord[];
  anomalous_count: number;
  novel_count: number;
}

export async function getDocumentReckoning(documentId: string): Promise<ReckoningResponse> {
  return apiRequest<ReckoningResponse>(`/reckoning/${documentId}`);
}
