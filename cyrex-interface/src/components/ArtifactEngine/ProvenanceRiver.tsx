// Directed flow from ANSWER backward through refs to source PDF highlight.

import React, { useCallback, useEffect, useMemo } from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { ArtifactBundle, ArtifactType } from '../../types/artifactEngine';
import { ArtifactGraphEdge } from '../../api/artifactEngine';

interface ProvenanceRiverProps {
  artifact: ArtifactBundle | null;
  graphNodes?: ArtifactBundle[];
  graphEdges?: ArtifactGraphEdge[];
  onNodeClick?: (artifactId: string) => void;
}

function nodeColor(artifactType: ArtifactType | string): string {
  switch (artifactType) {
    case ArtifactType.ANSWER:
      return '#4a9eff';
    case ArtifactType.REASONING:
      return '#cc44ff';
    case ArtifactType.EXTRACTION:
      return '#ffaa00';
    case ArtifactType.CANONICAL:
      return '#00cc88';
    case ArtifactType.SYSTEM:
      return '#44cccc';
    case ArtifactType.LEARNING:
      return '#ff66aa';
    default:
      return '#888888';
  }
}

const TYPE_ROW: Record<string, number> = {
  [ArtifactType.ANSWER]: 0,
  [ArtifactType.REASONING]: 1,
  [ArtifactType.EXTRACTION]: 2,
  [ArtifactType.SYSTEM]: 3,
  [ArtifactType.CANONICAL]: 4,
  [ArtifactType.LEARNING]: 5,
};

function buildGraphLayout(
  nodes: ArtifactBundle[],
  edges: ArtifactGraphEdge[],
): { nodes: Node[]; edges: Edge[] } {
  if (nodes.length === 0) {
    return { nodes: [], edges: [] };
  }

  const byRow: Record<number, ArtifactBundle[]> = {};
  for (const n of nodes) {
    const row = TYPE_ROW[n.artifact_type] ?? 3;
    byRow[row] = byRow[row] ?? [];
    byRow[row].push(n);
  }

  const flowNodes: Node[] = [];
  for (const [rowKey, rowNodes] of Object.entries(byRow)) {
    const row = Number(rowKey);
    rowNodes.forEach((bundle, idx) => {
      flowNodes.push({
        id: bundle.artifact_id,
        position: { x: idx * 220, y: row * 120 },
        data: {
          label: `${String(bundle.artifact_type).toUpperCase()}\n${bundle.artifact_id}`,
        },
        style: {
          background: nodeColor(bundle.artifact_type),
          color: '#fff',
          borderRadius: '8px',
          border: 'none',
          padding: '10px 16px',
          fontSize: '0.75rem',
          fontWeight: 600,
          whiteSpace: 'pre-wrap' as const,
          textAlign: 'center' as const,
          minWidth: '140px',
        },
      });
    });
  }

  const flowEdges: Edge[] = edges.map((edge, i) => ({
    id: `e-${i}-${edge.from}-${edge.to}`,
    source: edge.from,
    target: edge.to,
    animated: true,
    style: { stroke: '#4a9eff', strokeWidth: 2 },
    label: edge.ref_type,
    labelStyle: { fill: '#888', fontSize: 10 },
  }));

  return { nodes: flowNodes, edges: flowEdges };
}

export const ProvenanceRiver: React.FC<ProvenanceRiverProps> = ({
  artifact,
  graphNodes = [],
  graphEdges = [],
  onNodeClick,
}) => {
  const layout = useMemo(
    () => buildGraphLayout(graphNodes, graphEdges),
    [graphNodes, graphEdges],
  );
  const [nodes, setNodes, onNodesChange] = useNodesState(layout.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layout.edges);

  useEffect(() => {
    setNodes(layout.nodes);
    setEdges(layout.edges);
  }, [layout, setNodes, setEdges]);

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      onNodeClick?.(node.id);
    },
    [onNodeClick],
  );

  const hasData = graphNodes.length > 0;

  return (
    <div style={{ height: '450px', borderRadius: '8px', overflow: 'hidden', border: '1px solid #333', position: 'relative' }}>
      <div style={{
        position: 'absolute',
        zIndex: 10,
        top: '8px',
        left: '8px',
        display: 'flex',
        gap: '0.75rem',
        background: 'rgba(26,26,26,0.85)',
        padding: '0.4rem 0.75rem',
        borderRadius: '6px',
      }}>
        {[
          { color: nodeColor(ArtifactType.ANSWER), label: 'Answer' },
          { color: nodeColor(ArtifactType.REASONING), label: 'Reasoning' },
          { color: nodeColor(ArtifactType.EXTRACTION), label: 'Extraction' },
          { color: nodeColor(ArtifactType.SYSTEM), label: 'Eyes' },
        ].map(({ color, label }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <div style={{ width: '10px', height: '10px', borderRadius: '3px', background: color }} />
            <span style={{ color: '#b0b0b0', fontSize: '0.7rem' }}>{label}</span>
          </div>
        ))}
        {artifact && (
          <span style={{ color: '#666', fontSize: '0.7rem', marginLeft: '0.5rem' }}>
            root: {artifact.artifact_id}
          </span>
        )}
      </div>

      {!hasData ? (
        <div style={{
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#1a1a1a',
        }}>
          <p style={{ color: '#666' }}>No provenance graph — select an artifact or upload a document.</p>
        </div>
      ) : (
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          fitView
          attributionPosition="bottom-right"
          style={{ background: '#1a1a1a' }}
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#333" />
          <Controls style={{ background: '#2a2a2a', border: '1px solid #444' }} />
          <MiniMap
            style={{ background: '#2a2a2a', border: '1px solid #444' }}
            nodeColor={(node) => (node.style?.background as string) ?? '#888'}
          />
        </ReactFlow>
      )}
    </div>
  );
};

export default ProvenanceRiver;
