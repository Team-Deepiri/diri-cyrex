import { PressureCell } from '../types/artifactEngine';
import { apiRequest } from './client';

export interface PressureMapResponse {
  document_id: string;
  cells: PressureCell[];
  fault_zone_count: number;
  max_score: number;
}

export async function getDocumentPressure(documentId: string): Promise<PressureMapResponse> {
  return apiRequest<PressureMapResponse>(`/pressure/${documentId}`);
}

export async function getCorpusPressure(): Promise<PressureMapResponse> {
  return apiRequest<PressureMapResponse>('/pressure');
}
