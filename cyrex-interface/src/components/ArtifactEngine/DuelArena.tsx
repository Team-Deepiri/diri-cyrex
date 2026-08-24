import React from 'react';
import { DuelArenaResponse } from '../../api/duel';

interface DuelArenaProps {
  duel: DuelArenaResponse | null;
  loading?: boolean;
}

export const DuelArena: React.FC<DuelArenaProps> = ({ duel, loading }) => {
  if (loading) {
    return <p style={{ color: '#666', fontSize: '0.85rem' }}>Loading duel state…</p>;
  }

  if (!duel) {
    return (
      <p style={{ color: '#666', fontSize: '0.85rem' }}>
        No duel artifact yet — upload a document to run adversarial extract.
      </p>
    );
  }

  return (
    <div>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        marginBottom: '0.75rem',
        fontSize: '0.75rem',
        color: '#888',
      }}>
        <span>Agent A: <strong style={{ color: '#4a9eff' }}>{duel.agent_a_id}</strong></span>
        <span>Agent B: <strong style={{ color: '#cc44ff' }}>{duel.agent_b_id}</strong></span>
        <span>{duel.disagreements.length} disagreements · {duel.resolution_status}</span>
      </div>

      <div style={{
        maxHeight: '240px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.35rem',
      }}>
        {duel.fields.length === 0 ? (
          <p style={{ color: '#666', fontSize: '0.85rem' }}>No extracted fields in duel.</p>
        ) : (
          duel.fields.map(row => (
            <div
              key={row.field_name}
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr 1fr',
                gap: '0.5rem',
                padding: '0.5rem',
                borderRadius: '6px',
                background: row.is_disagreement ? '#3a2020' : '#1a1a1a',
                border: row.is_disagreement
                  ? '1px solid rgba(255, 80, 80, 0.5)'
                  : '1px solid #333',
                fontSize: '0.8rem',
              }}
            >
              <div style={{ color: '#e0e0e0', fontWeight: 600 }}>{row.field_name}</div>
              <div style={{ color: '#4a9eff' }}>
                {String(row.agent_a_value ?? '—')}
                {row.agent_a_confidence != null && (
                  <span style={{ color: '#666', marginLeft: '0.25rem' }}>
                    ({row.agent_a_confidence.toFixed(2)})
                  </span>
                )}
              </div>
              <div style={{ color: '#cc44ff' }}>
                {String(row.agent_b_value ?? '—')}
                {row.agent_b_confidence != null && (
                  <span style={{ color: '#666', marginLeft: '0.25rem' }}>
                    ({row.agent_b_confidence.toFixed(2)})
                  </span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default DuelArena;
