// Reckoning Compass. Upload is a confirmation event — shows the
// predicted range for each field alongside the actual value, tagged
// confirmed / anomalous / novel.

import React, { useEffect, useState } from 'react';
import { PredictionRecord, PredictionStatus } from '../../types/artifactEngine';
import { getReckoning } from '../../api/artifactEngine';

interface ReckoningCompassProps {
  documentId: string;
}

function statusColor(status: PredictionStatus): string {
  switch (status) {
    case PredictionStatus.CONFIRMED:
      return '#50c878';
    case PredictionStatus.ANOMALOUS:
      return '#dc3232';
    case PredictionStatus.NOVEL:
      return '#4a9eff';
    case PredictionStatus.NO_PRIOR:
    default:
      return '#666';
  }
}

function statusLabel(status: PredictionStatus): string {
  switch (status) {
    case PredictionStatus.CONFIRMED:
      return 'CONFIRMED';
    case PredictionStatus.ANOMALOUS:
      return 'ANOMALOUS';
    case PredictionStatus.NOVEL:
      return 'NOVEL';
    case PredictionStatus.NO_PRIOR:
    default:
      return 'NO PRIOR';
  }
}

// Position of the actual value along the predicted range, 0-100%.
function pinPosition(record: PredictionRecord): number | null {
  if (record.predicted_range == null || record.actual_value == null) return null;
  const { min, max } = record.predicted_range;
  if (min == null || max == null || max === min) return null;
  const actual = Number(record.actual_value);
  if (Number.isNaN(actual)) return null;
  const pct = ((actual - min) / (max - min)) * 100;
  return Math.max(0, Math.min(100, pct));
}

export const ReckoningCompass: React.FC<ReckoningCompassProps> = ({ documentId }) => {
  const [records, setRecords] = useState<PredictionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    getReckoning(documentId)
      .then((res) => {
        if (!cancelled) setRecords(res.records);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load reckoning data');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [documentId]);

  if (loading) {
    return <p style={{ color: '#666', fontSize: '0.85rem' }}>Loading reckoning data...</p>;
  }

  if (error) {
    return <p style={{ color: '#ff5050', fontSize: '0.85rem' }}>Error: {error}</p>;
  }

  if (records.length === 0) {
    return <p style={{ color: '#666', fontSize: '0.85rem' }}>No prediction records for this document.</p>;
  }

  return (
    <div>
      {records.map((record) => {
        const pin = pinPosition(record);
        return (
          <div key={record.field_name} style={{ marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
              <span style={{ color: '#e0e0e0', fontSize: '0.8rem' }}>{record.field_name}</span>
              <span
                style={{
                  fontSize: '0.65rem',
                  padding: '0.1rem 0.5rem',
                  borderRadius: '999px',
                  color: statusColor(record.status),
                  background: `${statusColor(record.status)}22`,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                }}
              >
                {statusLabel(record.status)}
              </span>
            </div>

            {record.predicted_range && pin != null ? (
              <div
                style={{
                  position: 'relative',
                  height: '10px',
                  borderRadius: '5px',
                  background: 'linear-gradient(90deg, #333, #444)',
                }}
              >
                <div
                  title={`Actual: ${record.actual_value}`}
                  style={{
                    position: 'absolute',
                    left: `${pin}%`,
                    top: '-4px',
                    width: '18px',
                    height: '18px',
                    borderRadius: '50%',
                    background: statusColor(record.status),
                    border: '2px solid #1a1a1a',
                    transform: 'translateX(-50%)',
                    boxShadow: `0 0 8px ${statusColor(record.status)}`,
                  }}
                />
              </div>
            ) : (
              <div style={{ fontSize: '0.75rem', color: '#666' }}>No prior — first time seeing this field.</div>
            )}

            {record.predicted_range && (
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.15rem' }}>
                <span style={{ color: '#666', fontSize: '0.7rem' }}>{record.predicted_range.min}</span>
                <span style={{ color: '#888', fontSize: '0.7rem' }}>
                  predicted mean: {record.predicted_mean ?? '—'}
                </span>
                <span style={{ color: '#666', fontSize: '0.7rem' }}>{record.predicted_range.max}</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default ReckoningCompass;