// Duel Arena & Disagreement Ribbon.
// TODO: swap MOCK_DUEL for real fetch

import React from 'react';
import { DuelState, FieldDiscrepancy } from '../../types/artifactEngine';

interface DuelArenaProps {
  duelState: DuelState;
}

export const MOCK_DUEL: DuelState = {
  document_id: 'lease_001',
  artifact_id: 'art_duel_001',
  agent_a_id: 'agent_a_llama',
  agent_b_id: 'agent_b_gpt',
  agent_a_fields: [
    {
      field_name: 'base_rent',
      value: 4500,
      value_type: 'currency',
      citations: [],
      confidence: 0.95,
      referenced_by: [],
      references: [],
    },
    {
      field_name: 'notice_period',
      value: 90,
      value_type: 'integer',
      citations: [],
      confidence: 0.88,
      referenced_by: [],
      references: [],
    },
  ],
  agent_b_fields: [
    {
      field_name: 'base_rent',
      value: 4500,
      value_type: 'currency',
      citations: [],
      confidence: 0.93,
      referenced_by: [],
      references: [],
    },
    {
      field_name: 'notice_period',
      value: 60,
      value_type: 'integer',
      citations: [],
      confidence: 0.80,
      referenced_by: [],
      references: [],
    },
  ],
  disagreements: [
    {
      field_name: 'notice_period',
      agent_a_value: 90,
      agent_b_value: 60,
      agent_a_confidence: 0.88,
      agent_b_confidence: 0.80,
      confidence_delta: 0.08,
      reason: "Agent A parsed 'ninety days'; Agent B parsed '60 days' from the same span",
    },
  ],
  resolution_status: 'unresolved',
};

// Amber → red based on how big the confidence gap is between agents
function ribbonColor(confidenceDelta?: number): string {
  const delta = confidenceDelta ?? 0;
  if (delta > 0.15) return 'rgba(220, 50, 50, 0.9)';
  if (delta > 0.05) return 'rgba(220, 130, 40, 0.9)';
  return 'rgba(220, 180, 40, 0.8)';
}

export const DuelArena: React.FC<DuelArenaProps> = ({ duelState }) => {
  const [expandedField, setExpandedField] = React.useState<string | null>(null);

  const disagreementFor = (fieldName: string): FieldDiscrepancy | undefined =>
    duelState.disagreements.find((d) => d.field_name === fieldName);

  const allFieldNames = Array.from(
    new Set([
      ...duelState.agent_a_fields.map((f) => f.field_name),
      ...duelState.agent_b_fields.map((f) => f.field_name),
    ])
  );

  const getValue = (fieldName: string, agent: 'a' | 'b') => {
    const fields = agent === 'a' ? duelState.agent_a_fields : duelState.agent_b_fields;
    return fields.find((f) => f.field_name === fieldName)?.value;
  };

  return (
    <div>
      {/* Resolution status badge */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', gap: '1.5rem' }}>
          <span style={{ color: '#888', fontSize: '0.75rem' }}>Agent A: {duelState.agent_a_id}</span>
          <span style={{ color: '#888', fontSize: '0.75rem' }}>Agent B: {duelState.agent_b_id}</span>
        </div>
        <span
          style={{
            fontSize: '0.7rem',
            padding: '0.2rem 0.6rem',
            borderRadius: '999px',
            background:
              duelState.resolution_status === 'unresolved'
                ? 'rgba(220, 130, 40, 0.2)'
                : duelState.resolution_status === 'resolved'
                ? 'rgba(80, 200, 120, 0.2)'
                : 'rgba(120, 120, 120, 0.2)',
            color:
              duelState.resolution_status === 'unresolved'
                ? '#dc8228'
                : duelState.resolution_status === 'resolved'
                ? '#50c878'
                : '#999',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}
        >
          {duelState.resolution_status}
        </span>
      </div>

      {/* Field rows */}
      {allFieldNames.map((fieldName) => {
        const disagreement = disagreementFor(fieldName);
        const isDisagreement = !!disagreement;
        const isExpanded = expandedField === fieldName;

        return (
          <div key={fieldName} style={{ marginBottom: '0.5rem' }}>
            <div
              onClick={() => isDisagreement && setExpandedField(isExpanded ? null : fieldName)}
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr auto 1fr',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.5rem 0.75rem',
                borderRadius: '4px',
                background: '#1a1a1a',
                border: isDisagreement ? `1px solid ${ribbonColor(disagreement?.confidence_delta)}` : '1px solid #333',
                opacity: isDisagreement ? 1 : 0.3,
                cursor: isDisagreement ? 'pointer' : 'default',
              }}
            >
              <span style={{ color: '#e0e0e0', fontSize: '0.85rem', textAlign: 'right' }}>
                {String(getValue(fieldName, 'a') ?? '—')}
              </span>

              <span
                style={{
                  fontSize: '0.65rem',
                  color: isDisagreement ? ribbonColor(disagreement?.confidence_delta) : '#666',
                  padding: '0 0.5rem',
                  fontWeight: isDisagreement ? 600 : 400,
                  whiteSpace: 'nowrap',
                }}
                title={fieldName}
              >
                {isDisagreement ? '◄──✕──►' : '◄────►'} {fieldName}
              </span>

              <span style={{ color: '#e0e0e0', fontSize: '0.85rem' }}>
                {String(getValue(fieldName, 'b') ?? '—')}
              </span>
            </div>

            {/* Disagreement reason expansion */}
            {isExpanded && disagreement?.reason && (
              <div
                style={{
                  marginTop: '0.25rem',
                  padding: '0.5rem 0.75rem',
                  fontSize: '0.75rem',
                  color: '#b0b0b0',
                  background: '#151515',
                  borderRadius: '4px',
                  borderLeft: `2px solid ${ribbonColor(disagreement.confidence_delta)}`,
                }}
              >
                {disagreement.reason}
                {disagreement.confidence_delta != null && (
                  <span style={{ color: '#888', marginLeft: '0.5rem' }}>
                    (Δ confidence: {disagreement.confidence_delta.toFixed(2)})
                  </span>
                )}
              </div>
            )}
          </div>
        );
      })}

      {allFieldNames.length === 0 && (
        <p style={{ color: '#666', fontSize: '0.85rem' }}>No fields to compare yet.</p>
      )}
    </div>
  );
};

export default DuelArena;