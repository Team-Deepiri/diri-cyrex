// Duel Arena & Disagreement Ribbon — Track-c UI wired to live duel API.

import React from 'react';
import { DuelArenaResponse } from '../../api/duel';
import { DuelState, FieldDiscrepancy } from '../../types/artifactEngine';

interface DuelArenaProps {
  /** Live API payload from GET /duel/{document_id}. */
  duel?: DuelArenaResponse | null;
  loading?: boolean;
  /** Track-c mock / storybook shape (optional fallback). */
  duelState?: DuelState;
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

function ribbonColor(confidenceDelta?: number): string {
  const delta = confidenceDelta ?? 0;
  if (delta > 0.15) return 'rgba(220, 50, 50, 0.9)';
  if (delta > 0.05) return 'rgba(220, 130, 40, 0.9)';
  return 'rgba(220, 180, 40, 0.8)';
}

type ArenaView = {
  agent_a_id: string;
  agent_b_id: string;
  resolution_status: string;
  disagreements: FieldDiscrepancy[];
  allFieldNames: string[];
  getValue: (fieldName: string, agent: 'a' | 'b') => unknown;
};

function fromDuelState(duelState: DuelState): ArenaView {
  return {
    agent_a_id: duelState.agent_a_id,
    agent_b_id: duelState.agent_b_id,
    resolution_status: duelState.resolution_status,
    disagreements: duelState.disagreements,
    allFieldNames: Array.from(
      new Set([
        ...duelState.agent_a_fields.map((f) => f.field_name),
        ...duelState.agent_b_fields.map((f) => f.field_name),
      ]),
    ),
    getValue: (fieldName, agent) => {
      const fields = agent === 'a' ? duelState.agent_a_fields : duelState.agent_b_fields;
      return fields.find((f) => f.field_name === fieldName)?.value;
    },
  };
}

function fromLiveDuel(duel: DuelArenaResponse): ArenaView {
  return {
    agent_a_id: duel.agent_a_id,
    agent_b_id: duel.agent_b_id,
    resolution_status: duel.resolution_status,
    disagreements: duel.disagreements,
    allFieldNames: duel.fields.map((f) => f.field_name),
    getValue: (fieldName, agent) => {
      const row = duel.fields.find((f) => f.field_name === fieldName);
      return agent === 'a' ? row?.agent_a_value : row?.agent_b_value;
    },
  };
}

export const DuelArena: React.FC<DuelArenaProps> = ({ duel, loading, duelState }) => {
  const [expandedField, setExpandedField] = React.useState<string | null>(null);

  if (loading) {
    return <p style={{ color: '#666', fontSize: '0.85rem' }}>Loading duel state…</p>;
  }

  const view: ArenaView | null = duel
    ? fromLiveDuel(duel)
    : duelState
      ? fromDuelState(duelState)
      : null;

  if (!view) {
    return (
      <p style={{ color: '#666', fontSize: '0.85rem' }}>
        No duel artifact yet — upload a document to run adversarial extract.
      </p>
    );
  }

  const disagreementFor = (fieldName: string): FieldDiscrepancy | undefined =>
    view.disagreements.find((d) => d.field_name === fieldName);

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', gap: '1.5rem' }}>
          <span style={{ color: '#888', fontSize: '0.75rem' }}>Agent A: {view.agent_a_id}</span>
          <span style={{ color: '#888', fontSize: '0.75rem' }}>Agent B: {view.agent_b_id}</span>
        </div>
        <span
          style={{
            fontSize: '0.7rem',
            padding: '0.2rem 0.6rem',
            borderRadius: '999px',
            background:
              view.resolution_status === 'unresolved'
                ? 'rgba(220, 130, 40, 0.2)'
                : view.resolution_status === 'resolved'
                  ? 'rgba(80, 200, 120, 0.2)'
                  : 'rgba(120, 120, 120, 0.2)',
            color:
              view.resolution_status === 'unresolved'
                ? '#dc8228'
                : view.resolution_status === 'resolved'
                  ? '#50c878'
                  : '#999',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}
        >
          {view.resolution_status}
        </span>
      </div>

      {view.allFieldNames.map((fieldName) => {
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
                border: isDisagreement
                  ? `1px solid ${ribbonColor(disagreement?.confidence_delta)}`
                  : '1px solid #333',
                opacity: isDisagreement ? 1 : 0.3,
                cursor: isDisagreement ? 'pointer' : 'default',
              }}
            >
              <span style={{ color: '#e0e0e0', fontSize: '0.85rem', textAlign: 'right' }}>
                {String(view.getValue(fieldName, 'a') ?? '—')}
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
                {String(view.getValue(fieldName, 'b') ?? '—')}
              </span>
            </div>

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

      {view.allFieldNames.length === 0 && (
        <p style={{ color: '#666', fontSize: '0.85rem' }}>No fields to compare yet.</p>
      )}
    </div>
  );
};

export default DuelArena;
