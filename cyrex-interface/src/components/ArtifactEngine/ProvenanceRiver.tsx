// Directed flow from ANSWER backward through refs to source PDF highlight. Each node is an artifact,
// each edge is a dependency up until CitationLocator.

import React, { useCallback } from 'react';
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

// Properties
interface ProvenanceRiverProps {
  artifact: ArtifactBundle | null;
  onNodeClick?: (artifactId: string) => void;
}

// Color per artifact type 
function nodeColor(artifactType: ArtifactType): string {
  switch (artifactType) {
     // blue — final answer
    case ArtifactType.ANSWER:       return '#4a9eff';
    // purple — reasoning layer
    case ArtifactType.REASONING:    return '#cc44ff';
     // amber — extraction layer
    case ArtifactType.EXTRACTION:   return '#ffaa00';
     // green — source document
    case ArtifactType.CANONICAL:    return '#00cc88';
     // grey — other
    default:                        return '#888888';
  }
}

// Provenance nodes and edges for Week 3
// TODO: Replace with real ArtifactStorePort.get_graph_neighborhood
function buildMockGraph(): { nodes: Node[], edges: Edge[] } {
  const nodes: Node[] = [
    {
      id: 'answer_001',
      position: { x: 400, y: 0 },
      data: { label: 'ANSWER\nart_answer_001' },
      style: {
        background: nodeColor(ArtifactType.ANSWER),
        color: '#fff',
        borderRadius: '8px',
        border: 'none',
        padding: '10px 16px',
        fontSize: '0.75rem',
        fontWeight: 600,
        whiteSpace: 'pre-wrap' as const,
        textAlign: 'center' as const,
      },
    },
    {
      id: 'reasoning_001',
      position: { x: 400, y: 120 },
      data: { label: 'REASONING\nart_reasoning_001' },
      style: {
        background: nodeColor(ArtifactType.REASONING),
        color: '#fff',
        borderRadius: '8px',
        border: 'none',
        padding: '10px 16px',
        fontSize: '0.75rem',
        fontWeight: 600,
        whiteSpace: 'pre-wrap' as const,
        textAlign: 'center' as const,
      },
    },
    {
      id: 'extraction_001',
      position: { x: 200, y: 240 },
      data: { label: 'EXTRACTION\nart_extraction_001' },
      style: {
        background: nodeColor(ArtifactType.EXTRACTION),
        color: '#fff',
        borderRadius: '8px',
        border: 'none',
        padding: '10px 16px',
        fontSize: '0.75rem',
        fontWeight: 600,
        whiteSpace: 'pre-wrap' as const,
        textAlign: 'center' as const,
      },
    },
    {
      id: 'extraction_002',
      position: { x: 600, y: 240 },
      data: { label: 'EXTRACTION\nart_extraction_002' },
      style: {
        background: nodeColor(ArtifactType.EXTRACTION),
        color: '#fff',
        borderRadius: '8px',
        border: 'none',
        padding: '10px 16px',
        fontSize: '0.75rem',
        fontWeight: 600,
        whiteSpace: 'pre-wrap' as const,
        textAlign: 'center' as const,
      },
    },
    {
      id: 'canonical_001',
      position: { x: 400, y: 360 },
      data: { label: 'SOURCE PDF\nchar 1042–1080 · p.1' },
      style: {
        background: nodeColor(ArtifactType.CANONICAL),
        color: '#fff',
        borderRadius: '8px',
        border: '2px solid rgba(0, 204, 136, 0.6)',
        padding: '10px 16px',
        fontSize: '0.75rem',
        fontWeight: 600,
        whiteSpace: 'pre-wrap' as const,
        textAlign: 'center' as const,
      },
    },
  ];

  const edges: Edge[] = [
    {
      id: 'e1',
      source: 'answer_001',
      target: 'reasoning_001',
      animated: true,
      style: { stroke: '#4a9eff', strokeWidth: 2 },
      label: 'depends_on',
      labelStyle: { fill: '#888', fontSize: 10 },
    },
    {
      id: 'e2',
      source: 'reasoning_001',
      target: 'extraction_001',
      animated: true,
      style: { stroke: '#cc44ff', strokeWidth: 2 },
      label: 'cites',
      labelStyle: { fill: '#888', fontSize: 10 },
    },
    {
      id: 'e3',
      source: 'reasoning_001',
      target: 'extraction_002',
      animated: true,
      style: { stroke: '#cc44ff', strokeWidth: 2 },
      label: 'cites',
      labelStyle: { fill: '#888', fontSize: 10 },
    },
    {
      id: 'e4',
      source: 'extraction_001',
      target: 'canonical_001',
      animated: true,
      style: { stroke: '#ffaa00', strokeWidth: 2 },
      label: 'canonical_of',
      labelStyle: { fill: '#888', fontSize: 10 },
    },
    {
      id: 'e5',
      source: 'extraction_002',
      target: 'canonical_001',
      animated: true,
      style: { stroke: '#ffaa00', strokeWidth: 2 },
      label: 'canonical_of',
      labelStyle: { fill: '#888', fontSize: 10 },
    },
  ];

  return { nodes, edges };
}

export const ProvenanceRiver: React.FC<ProvenanceRiverProps> = ({
  artifact,
  onNodeClick,
}) => {
  const { nodes: initialNodes, edges: initialEdges } = buildMockGraph();
  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  const handleNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    onNodeClick?.(node.id);
  }, [onNodeClick]);

  return (
    <div style={{ height: '450px', borderRadius: '8px', overflow: 'hidden', border: '1px solid #333' }}>

      {/* Legend */}
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
          { color: nodeColor(ArtifactType.CANONICAL), label: 'Source' },
        ].map(({ color, label }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <div style={{ width: '10px', height: '10px', borderRadius: '3px', background: color }} />
            <span style={{ color: '#b0b0b0', fontSize: '0.7rem' }}>{label}</span>
          </div>
        ))}
      </div>

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
        <Background
          variant={BackgroundVariant.Dots}
          gap={16}
          size={1}
          color="#333"
        />
        <Controls style={{ background: '#2a2a2a', border: '1px solid #444' }} />
        <MiniMap
          style={{ background: '#2a2a2a', border: '1px solid #444' }}
          nodeColor={(node) => node.style?.background as string ?? '#888'}
        />
      </ReactFlow>
    </div>
  );
};

export default ProvenanceRiver;