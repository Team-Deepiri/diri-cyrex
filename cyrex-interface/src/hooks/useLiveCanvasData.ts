import { useCallback, useEffect, useState } from 'react';
import { getDocumentPressure } from '../api/pressure';
import { getDocumentReckoning } from '../api/reckoning';
import { getEyesScene, getEyesStatus, EyesSceneIdentity, EyesStatus } from '../api/eyes';
import { getArtifactGraph } from '../api/artifactEngine';
import { ArtifactBundle, PressureCell, PredictionRecord } from '../types/artifactEngine';
import { ArtifactGraphEdge } from '../api/artifactEngine';
import { DEFAULT_CANVAS_POLL_MS, ELKEDEL_SCENE_DOCUMENT_ID } from '../constants/agi';

export interface LiveCanvasData {
  documentId: string;
  pressureCells: PressureCell[];
  reckoningRecords: PredictionRecord[];
  anomalousCount: number;
  novelCount: number;
  eyesStatus: EyesStatus | null;
  sceneIdentities: EyesSceneIdentity[];
  graphNodes: ArtifactBundle[];
  graphEdges: ArtifactGraphEdge[];
  selectedArtifactId: string | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  setDocumentId: (id: string) => void;
  selectArtifact: (artifactId: string | null) => Promise<void>;
}

export function useLiveCanvasData(
  initialDocumentId = ELKEDEL_SCENE_DOCUMENT_ID,
  pollMs = DEFAULT_CANVAS_POLL_MS,
): LiveCanvasData {
  const [documentId, setDocumentId] = useState(initialDocumentId);
  const [pressureCells, setPressureCells] = useState<PressureCell[]>([]);
  const [reckoningRecords, setReckoningRecords] = useState<PredictionRecord[]>([]);
  const [anomalousCount, setAnomalousCount] = useState(0);
  const [novelCount, setNovelCount] = useState(0);
  const [eyesStatus, setEyesStatus] = useState<EyesStatus | null>(null);
  const [sceneIdentities, setSceneIdentities] = useState<EyesSceneIdentity[]>([]);
  const [graphNodes, setGraphNodes] = useState<ArtifactBundle[]>([]);
  const [graphEdges, setGraphEdges] = useState<ArtifactGraphEdge[]>([]);
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadGraph = useCallback(async (artifactId: string) => {
    try {
      const graph = await getArtifactGraph(artifactId, 2);
      setGraphNodes(graph.nodes);
      setGraphEdges(graph.edges);
    } catch {
      setGraphNodes([]);
      setGraphEdges([]);
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const [pressure, reckoning, status, scene] = await Promise.all([
        getDocumentPressure(documentId),
        getDocumentReckoning(documentId).catch(() => ({
          document_id: documentId,
          records: [],
          anomalous_count: 0,
          novel_count: 0,
        })),
        getEyesStatus().catch(() => null),
        getEyesScene(30).catch(() => ({ identities: [] })),
      ]);
      setPressureCells(pressure.cells);
      setReckoningRecords(reckoning.records);
      setAnomalousCount(reckoning.anomalous_count);
      setNovelCount(reckoning.novel_count);
      setEyesStatus(status);
      setSceneIdentities(scene.identities ?? []);

      const topArtifact = pressure.cells.find(c => c.drill_down_artifact_ids.length > 0)
        ?.drill_down_artifact_ids[0];
      const graphTarget = selectedArtifactId ?? topArtifact ?? null;
      if (graphTarget) {
        await loadGraph(graphTarget);
        if (!selectedArtifactId) {
          setSelectedArtifactId(graphTarget);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load canvas data');
    } finally {
      setLoading(false);
    }
  }, [documentId, loadGraph, selectedArtifactId]);

  const selectArtifact = useCallback(async (artifactId: string | null) => {
    setSelectedArtifactId(artifactId);
    if (artifactId) {
      await loadGraph(artifactId);
    } else {
      setGraphNodes([]);
      setGraphEdges([]);
    }
  }, [loadGraph]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), pollMs);
    return () => window.clearInterval(timer);
  }, [refresh, pollMs]);

  return {
    documentId,
    pressureCells,
    reckoningRecords,
    anomalousCount,
    novelCount,
    eyesStatus,
    sceneIdentities,
    graphNodes,
    graphEdges,
    selectedArtifactId,
    loading,
    error,
    refresh,
    setDocumentId,
    selectArtifact,
  };
}
