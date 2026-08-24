import React from 'react';
import { PredictionRecord, PredictionStatus } from '../../types/artifactEngine';

interface ReckoningCompassProps {
  records: PredictionRecord[];
  anomalousCount: number;
  novelCount: number;
}

function statusColor(status: PredictionStatus | string): string {
  switch (status) {
    case PredictionStatus.ANOMALOUS:
    case 'anomalous':
      return '#ff5050';
    case PredictionStatus.NOVEL:
    case 'novel':
      return '#4a9eff';
    case PredictionStatus.CONFIRMED:
    case 'confirmed':
      return '#00cc88';
    default:
      return '#888';
  }
}

export const ReckoningCompass: React.FC<ReckoningCompassProps> = ({
  records,
  anomalousCount,
  novelCount,
}) => {
  const flagged = records.filter(
    r => r.status === PredictionStatus.ANOMALOUS || r.status === PredictionStatus.NOVEL,
  );

  return (
    <div>
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.75rem' }}>
        <span style={{ color: '#ff5050', fontSize: '0.85rem' }}>{anomalousCount} anomalous</span>
        <span style={{ color: '#4a9eff', fontSize: '0.85rem' }}>{novelCount} novel</span>
        <span style={{ color: '#888', fontSize: '0.85rem' }}>{records.length} total fields</span>
      </div>

      {flagged.length === 0 ? (
        <p style={{ color: '#666', fontSize: '0.85rem' }}>
          No reckoning flags yet — upload a document or wait for Elkedel eyes sync.
        </p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '220px', overflowY: 'auto' }}>
          {flagged.map(rec => (
            <div
              key={rec.field_name}
              style={{
                background: '#1a1a1a',
                border: `1px solid ${statusColor(rec.status)}55`,
                borderRadius: '6px',
                padding: '0.5rem 0.75rem',
                fontSize: '0.8rem',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <strong style={{ color: '#e0e0e0' }}>{rec.field_name}</strong>
                <span style={{ color: statusColor(rec.status), textTransform: 'uppercase', fontSize: '0.7rem' }}>
                  {rec.status}
                </span>
              </div>
              <div style={{ color: '#888', marginTop: '0.25rem' }}>
                prior {String(rec.predicted_mean ?? '—')} → actual {String(rec.actual_value ?? '—')}
                {rec.sigma_delta != null && (
                  <span style={{ marginLeft: '0.5rem', color: '#ffaa00' }}>
                    σΔ {rec.sigma_delta.toFixed(2)}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ReckoningCompass;
