import React, { useState } from 'react';
import { TerrainSurvey } from './TerrainSurvey';
import { PressureCell } from '../../types/artifactEngine';
import { FaultDrillDown } from './FaultDrillDown';
import { ProvenanceRiver } from './ProvenanceRiver';
import { DuelArena, MOCK_DUEL } from './DuelArena';
import { ReckoningCompass } from './ReckoningCompass';
import { WitnessStitch } from './WitnessStitch';

// TODO: replace MOCK_CELLS with usePressureMap(documentId) hook
const MOCK_CELLS: PressureCell[] = [
  { document_id: 'doc-1', section_id: 'financial_terms', page: 1, score: 0.85, is_fault_zone: true, discrepancy_count: 3, reflect_failures: 1, low_confidence_count: 2, duel_disagreements: 2, drill_down_artifact_ids: ['art_001'] },
  { document_id: 'doc-1', section_id: 'termination_clause', page: 2, score: 0.4, is_fault_zone: false, discrepancy_count: 1, reflect_failures: 0, low_confidence_count: 1, duel_disagreements: 0, drill_down_artifact_ids: [] },
  { document_id: 'doc-1', section_id: 'obligations', page: 3, score: 0.65, is_fault_zone: false, discrepancy_count: 2, reflect_failures: 0, low_confidence_count: 1, duel_disagreements: 1, drill_down_artifact_ids: [] },
  { document_id: 'doc-1', section_id: 'financial_terms', page: 2, score: 0.9, is_fault_zone: true, discrepancy_count: 4, reflect_failures: 2, low_confidence_count: 3, duel_disagreements: 3, drill_down_artifact_ids: ['art_002', 'art_003'] },
];

interface ArtifactEngineCanvasProps {
  // Which document the Canvas is currently surveying.
  documentId?: string;
}

export const ArtifactEngineCanvas: React.FC<ArtifactEngineCanvasProps> = ({
  documentId = 'lease_001',
}) => {
  const [activePanel] = useState<'terrain' | 'duel' | 'voice' | 'provenance'>('terrain');
  const [selectedFaultCell, setSelectedFaultCell] = useState<PressureCell | null>(null); 

  return (
    <div style={{ padding: '2rem', maxWidth: '1400px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <h2 style={{ color: '#e0e0e0', margin: 0 }}>Artifact Engine Canvas</h2>
        <button style={{
          padding: '0.5rem 1.5rem',
          background: '#4a9eff',
          color: '#fff',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer'
        }}>
          Upload Document.
        </button>
      </div>

      {/*Main*/}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        
        {/*Terrain Survey */}
        <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', minHeight: '300px' }}>
          <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Terrain Survey</h3>
          <TerrainSurvey
          cells={MOCK_CELLS}
          onFaultZoneClick={(cell) => setSelectedFaultCell(cell)}
          />

          {/* Fault Drill-Down. On red zones only. */}
            {selectedFaultCell && (
              <div style={{ marginTop: '1rem' }}>
                <FaultDrillDown
                  cell={selectedFaultCell}
                  onArtifactClick={(id) => console.log('Navigate to artifact:', id)}
                  onClose={() => setSelectedFaultCell(null)}
                />
          </div>
        )}
        </div>

        {/* Duel Arena & Reckoning Compass */}
        <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', minHeight: '300px' }}>
          <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Duel Arena</h3>
          {/* TODO: swap MOCK_DUEL */}
          <DuelArena duelState={MOCK_DUEL} />

          <h3 style={{ color: '#e0e0e0', marginTop: '1.5rem', fontSize: '0.95rem' }}>Reckoning Compass</h3>
          <ReckoningCompass documentId={documentId} />
        </div>
      </div>

      {/* Voice Query — Witness Stitch & Confusion Gap */}
      <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
        <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Voice Query</h3>
        <WitnessStitch documentId={documentId} />
      </div>
      
      {/* Provenance River */}
      <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem', position: 'relative' }}>
        <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Provenance River</h3>
        <ProvenanceRiver
          artifact={null}
          onNodeClick={(id) => console.log('Artifact node clicked:', id)}
        />
      </div>

      {/*Ghost Graph Placeholder*/}
      <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px' }}>
        <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Artifact Graph</h3>
        <div style={{ background: '#1a1a1a', height: '150px', borderRadius: '4px', border: '1px solid #444', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <p style={{ color: '#666' }}>Ghost Graph here</p>
        </div>
      </div>
    </div>
  );
};

export default ArtifactEngineCanvas;