import React, { useRef, useState } from 'react';
import { TerrainSurvey } from './TerrainSurvey';
import { FaultDrillDown } from './FaultDrillDown';
import { ProvenanceRiver } from './ProvenanceRiver';
import { ReckoningCompass } from './ReckoningCompass';
import { DuelArena } from './DuelArena';
import { useLiveCanvasData } from '../../hooks/useLiveCanvasData';
import { uploadArtifact, voiceQuery } from '../../api/artifactEngine';
import { ELKEDEL_SCENE_DOCUMENT_ID } from '../../constants/agi';
import { VoiceQueryResponse } from '../../types/artifactEngine';

export const ArtifactEngineCanvas: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFaultCell, setSelectedFaultCell] = useState<ReturnType<typeof useLiveCanvasData>['pressureCells'][0] | null>(null);
  const [uploading, setUploading] = useState(false);
  const [voiceQuestion, setVoiceQuestion] = useState('');
  const [voiceLoading, setVoiceLoading] = useState(false);
  const [voiceResult, setVoiceResult] = useState<VoiceQueryResponse | null>(null);
  const [voiceError, setVoiceError] = useState<string | null>(null);

  const canvas = useLiveCanvasData(ELKEDEL_SCENE_DOCUMENT_ID);

  const handleUpload = async (file: File) => {
    setUploading(true);
    try {
      const bundle = await uploadArtifact(file, canvas.documentId);
      await canvas.selectArtifact(bundle.artifact_id);
      await canvas.refresh();
    } catch (err) {
      console.error('Upload failed', err);
    } finally {
      setUploading(false);
    }
  };

  const handleVoiceAsk = async () => {
    if (!voiceQuestion.trim()) return;
    setVoiceLoading(true);
    setVoiceError(null);
    try {
      const response = await voiceQuery({
        document_id: canvas.documentId,
        question: voiceQuestion.trim(),
        persona_scope: {
          witness_set_only: true,
          hard_citation_gate: true,
          corpus_filter: [canvas.documentId],
        },
      });
      setVoiceResult(response);
    } catch (err) {
      setVoiceError(err instanceof Error ? err.message : 'Voice query failed');
      setVoiceResult(null);
    } finally {
      setVoiceLoading(false);
    }
  };

  const selectedArtifact = canvas.graphNodes.find(n => n.artifact_id === canvas.selectedArtifactId) ?? null;

  return (
    <div style={{ padding: '2rem', maxWidth: '1400px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <div>
          <h2 style={{ color: '#e0e0e0', margin: 0 }}>Artifact Engine Canvas</h2>
          <p style={{ color: '#888', fontSize: '0.85rem', margin: '0.25rem 0 0' }}>
            Live pressure · reckoning · Elkedel eyes
            {canvas.eyesStatus?.running ? (
              <span style={{ color: '#00cc88', marginLeft: '0.5rem' }}>● eyes running</span>
            ) : (
              <span style={{ color: '#666', marginLeft: '0.5rem' }}>○ eyes idle</span>
            )}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <input
            type="text"
            value={canvas.documentId}
            onChange={e => canvas.setDocumentId(e.target.value)}
            style={{
              width: '280px',
              padding: '0.4rem 0.6rem',
              background: '#1a1a1a',
              color: '#e0e0e0',
              border: '1px solid #444',
              borderRadius: '4px',
              fontSize: '0.75rem',
              fontFamily: 'monospace',
            }}
            title="Document UUID for pressure/reckoning queries"
          />
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md,.doc,.docx"
            style={{ display: 'none' }}
            onChange={e => {
              const f = e.target.files?.[0];
              if (f) void handleUpload(f);
            }}
          />
          <button
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
            style={{
              padding: '0.5rem 1.5rem',
              background: uploading ? '#333' : '#4a9eff',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: uploading ? 'wait' : 'pointer',
            }}
          >
            {uploading ? 'Processing…' : 'Upload Document'}
          </button>
        </div>
      </div>

      {canvas.error && (
        <div style={{
          background: '#3a1a1a',
          border: '1px solid #ff5050',
          color: '#ffb0b0',
          padding: '0.75rem 1rem',
          borderRadius: '6px',
          marginBottom: '1rem',
          fontSize: '0.85rem',
        }}>
          {canvas.error}
        </div>
      )}

      {canvas.loading && canvas.pressureCells.length === 0 && (
        <p style={{ color: '#888', marginBottom: '1rem' }}>Loading live canvas data…</p>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', minHeight: '300px' }}>
          <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Terrain Survey</h3>
          <TerrainSurvey
            cells={canvas.pressureCells}
            onFaultZoneClick={(cell) => setSelectedFaultCell(cell)}
          />
          {selectedFaultCell && (
            <div style={{ marginTop: '1rem' }}>
              <FaultDrillDown
                cell={selectedFaultCell}
                onArtifactClick={(id) => void canvas.selectArtifact(id)}
                onClose={() => setSelectedFaultCell(null)}
              />
            </div>
          )}
        </div>

        <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', minHeight: '300px' }}>
          <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Reckoning Compass</h3>
          <ReckoningCompass
            records={canvas.reckoningRecords}
            anomalousCount={canvas.anomalousCount}
            novelCount={canvas.novelCount}
          />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px' }}>
          <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Elkedel Scene</h3>
          {canvas.sceneIdentities.length === 0 ? (
            <p style={{ color: '#666', fontSize: '0.85rem' }}>No identities in scene yet.</p>
          ) : (
            <ul style={{ margin: 0, padding: 0, listStyle: 'none', maxHeight: '180px', overflowY: 'auto' }}>
              {canvas.sceneIdentities.map(id => (
                <li
                  key={id.trace_id ?? id.identity_id ?? String(id.last_seen_ms)}
                  style={{
                    padding: '0.4rem 0',
                    borderBottom: '1px solid #333',
                    color: '#b0b0b0',
                    fontSize: '0.85rem',
                  }}
                >
                  <strong style={{ color: '#e0e0e0' }}>{id.label ?? 'object'}</strong>
                  {' '}· strength {(id.strength ?? 0).toFixed(2)}
                  {' '}· obs {id.n_observations ?? 0}
                </li>
              ))}
            </ul>
          )}
        </div>

        <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px' }}>
          <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Duel Arena</h3>
          <DuelArena duel={canvas.duel} loading={canvas.loading && !canvas.duel} />
        </div>
      </div>

      <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
        <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Voice Query</h3>
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
          <input
            type="text"
            value={voiceQuestion}
            onChange={e => setVoiceQuestion(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && void handleVoiceAsk()}
            placeholder="Ask a question about the document (citation-gated)…"
            style={{
              flex: 1,
              padding: '0.5rem',
              background: '#1a1a1a',
              color: '#e0e0e0',
              border: '1px solid #444',
              borderRadius: '4px',
            }}
          />
          <button
            disabled={voiceLoading}
            onClick={() => void handleVoiceAsk()}
            style={{
              padding: '0.5rem 1.5rem',
              background: voiceLoading ? '#333' : '#4a9eff',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: voiceLoading ? 'wait' : 'pointer',
            }}
          >
            {voiceLoading ? '…' : 'Ask'}
          </button>
        </div>
        {voiceError && (
          <p style={{ color: '#ff8080', fontSize: '0.85rem' }}>{voiceError}</p>
        )}
        <div style={{
          background: '#1a1a1a',
          padding: '1rem',
          borderRadius: '4px',
          border: '1px solid #444',
          minHeight: '80px',
        }}>
          {!voiceResult ? (
            <p style={{ color: '#666', margin: 0 }}>Cited answer or confession appears here.</p>
          ) : voiceResult.confessed ? (
            <div>
              <p style={{ color: '#ffaa00', margin: '0 0 0.5rem' }}>Confession — no witness span available</p>
              {(voiceResult.gaps ?? []).map((gap, i) => (
                <p key={i} style={{ color: '#888', fontSize: '0.85rem', margin: 0 }}>
                  {gap.reason}: {gap.claim_attempted}
                </p>
              ))}
            </div>
          ) : (
            voiceResult.spans.map(span => (
              <blockquote
                key={span.citation_id}
                style={{
                  margin: '0 0 0.5rem',
                  padding: '0.5rem 0.75rem',
                  borderLeft: '3px solid #4a9eff',
                  color: '#e0e0e0',
                  fontSize: '0.9rem',
                }}
              >
                "{span.quote}"
                <span style={{ display: 'block', color: '#666', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                  {span.citation_id} · chars {span.char_start}–{span.char_end}
                  {span.page != null ? ` · p.${span.page}` : ''}
                </span>
              </blockquote>
            ))
          )}
        </div>
      </div>

      <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
        <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Provenance River</h3>
        <ProvenanceRiver
          artifact={selectedArtifact}
          graphNodes={canvas.graphNodes}
          graphEdges={canvas.graphEdges}
          onNodeClick={(id) => void canvas.selectArtifact(id)}
        />
      </div>
    </div>
  );
};

export default ArtifactEngineCanvas;
