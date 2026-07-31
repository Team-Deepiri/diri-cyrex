// Click a fault zone on the Terrain Survey to show show which duel/reflect artifacts caused pressure.

import React, { useEffect, useState } from 'react';
import { PressureCell, ArtifactType } from '../../types/artifactEngine';
import { getArtifact } from '../../api/artifactEngine';

interface FaultDrillDownProps {
  cell: PressureCell | null;
  onArtifactClick?: (artifactId: string) => void;
  onClose?: () => void;
}

function badgeColor(type: ArtifactType | 'loading' | 'error'): string {
  switch (type) {
    case ArtifactType.EXTRACTION:
      return '#4a9eff';
    case ArtifactType.SYSTEM:
      return '#cc44ff';
    case ArtifactType.LEARNING:
      return '#50c878';
    case 'error':
      return '#ff5050';
    default:
      return '#888';
  }
}

export const FaultDrillDown: React.FC<FaultDrillDownProps> = ({
  cell,
  onArtifactClick,
  onClose,
}) => {
  const [artifactTypes, setArtifactTypes] = useState<Record<string, ArtifactType | 'loading' | 'error'>>({});

  useEffect(() => {
    if (!cell || cell.drill_down_artifact_ids.length === 0) return;

    // Fetch each artifact's type
    cell.drill_down_artifact_ids.forEach((artifactId) => {
      if (artifactTypes[artifactId]) return;
      setArtifactTypes((prev) => ({ ...prev, [artifactId]: 'loading' }));
      getArtifact(artifactId)
        .then((bundle) => {
          setArtifactTypes((prev) => ({ ...prev, [artifactId]: bundle.artifact_type }));
        })
        .catch(() => {
          setArtifactTypes((prev) => ({ ...prev, [artifactId]: 'error' }));
        });
    });
  }, [cell]);

  // Show nothing if no fault zone is selected
  if (!cell) {
    return (
      <div style={{
        background: '#1a1a1a',
        borderRadius: '8px',
        border: '1px solid #333',
        padding: '1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '120px',
      }}>
        <p style={{ color: '#666', fontSize: '0.85rem' }}>
          Click a fault zone on the Terrain Survey to drill down
        </p>
      </div>
    );
  }

  return (
    <div style={{
      background: '#1a1a1a',
      borderRadius: '8px',
      border: '1px solid rgba(255, 80, 80, 0.4)',
      padding: '1.5rem',
    }}>

      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: '1rem',
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ color: '#ff5050', fontSize: '1rem' }}>⚠</span>
            <h4 style={{ color: '#e0e0e0', margin: 0, fontSize: '1rem' }}>
              Fault Zone — {cell.section_id}
            </h4>
          </div>
          <p style={{ color: '#888', fontSize: '0.8rem', margin: '0.25rem 0 0 0' }}>
            Page {cell.page ?? 1} · Pressure score: {' '}
            <span style={{ color: '#ff5050', fontWeight: 600 }}>
              {cell.score.toFixed(2)}
            </span>
          </p>
        </div>

        {/* Close button */}
        {onClose && (
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: '1px solid #444',
              borderRadius: '4px',
              color: '#888',
              cursor: 'pointer',
              padding: '0.25rem 0.5rem',
              fontSize: '0.8rem',
            }}
          >
            ✕
          </button>
        )}
      </div>

      {/* Pressure breakdown */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '0.5rem',
        marginBottom: '1rem',
      }}>
        {[
          { label: 'Discrepancies', value: cell.discrepancy_count, color: '#ff5050' },
          { label: 'Reflect Failures', value: cell.reflect_failures, color: '#ffaa00' },
          { label: 'Low Confidence', value: cell.low_confidence_count, color: '#4a9eff' },
          { label: 'Duel Disagreements', value: cell.duel_disagreements, color: '#cc44ff' },
        ].map(({ label, value, color }) => (
          <div key={label} style={{
            background: '#2a2a2a',
            borderRadius: '6px',
            padding: '0.75rem',
            textAlign: 'center',
          }}>
            <div style={{ color, fontSize: '1.25rem', fontWeight: 700 }}>{value}</div>
            <div style={{ color: '#888', fontSize: '0.7rem', marginTop: '0.25rem' }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Artifact IDs that caused the pressure */}
      <div>
        <p style={{ color: '#b0b0b0', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
          Artifacts that caused this pressure:
        </p>

        {/* No artifacts found */}
        {cell.drill_down_artifact_ids.length === 0 ? (
          <p style={{ color: '#666', fontSize: '0.8rem' }}>
            No artifact IDs linked to this fault zone yet
          </p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            {cell.drill_down_artifact_ids.map(artifactId => {
              const artifactType = artifactTypes[artifactId];
              return (
                <button
                  key={artifactId}
                  onClick={() => onArtifactClick?.(artifactId)}
                  style={{
                    background: '#2a2a2a',
                    border: '1px solid #444',
                    borderRadius: '6px',
                    color: '#4a9eff',
                    cursor: 'pointer',
                    padding: '0.5rem 0.75rem',
                    textAlign: 'left',
                    fontSize: '0.85rem',
                    fontFamily: 'monospace',
                    transition: 'background 0.15s, border-color 0.15s',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                  }}
                  onMouseEnter={e => {
                    (e.target as HTMLButtonElement).style.background = '#333';
                    (e.target as HTMLButtonElement).style.borderColor = '#4a9eff';
                  }}
                  onMouseLeave={e => {
                    (e.target as HTMLButtonElement).style.background = '#2a2a2a';
                    (e.target as HTMLButtonElement).style.borderColor = '#444';
                  }}
                >
                  <span
                    style={{
                      fontSize: '0.65rem',
                      fontFamily: 'sans-serif',
                      fontWeight: 700,
                      padding: '0.1rem 0.4rem',
                      borderRadius: '4px',
                      color: badgeColor(artifactType ?? 'loading'),
                      background: `${badgeColor(artifactType ?? 'loading')}22`,
                      textTransform: 'uppercase',
                      letterSpacing: '0.03em',
                      flexShrink: 0,
                    }}
                  >
                    {!artifactType || artifactType === 'loading'
                      ? '···'
                      : artifactType === 'error'
                      ? 'unknown'
                      : artifactType}
                  </span>
                  <span>→ {artifactId}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default FaultDrillDown;
