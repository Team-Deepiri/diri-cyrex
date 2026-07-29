// Voice answers are not generative. When a claim can't be grounded, it renders as a grey void.

import React, { useState } from 'react';
import { VoiceQueryRequest, PersonaScope } from '../../types/artifactEngine';
import { voiceQuery } from '../../api/artifactEngine';

interface WitnessStitchProps {
  documentId: string;
  personaScope?: PersonaScope;
}

const DEFAULT_PERSONA_SCOPE: PersonaScope = {
  witness_set_only: true,
  hard_citation_gate: true,
  corpus_filter: [],
};

export const WitnessStitch: React.FC<WitnessStitchProps> = ({
  documentId,
  personaScope = DEFAULT_PERSONA_SCOPE,
}) => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Awaited<ReturnType<typeof voiceQuery>> | null>(null);

  const handleAsk = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const request: VoiceQueryRequest = {
        document_id: documentId,
        question: question.trim(),
        persona_scope: personaScope,
      };
      const response = await voiceQuery(request);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Voice query failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Query bar */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
          placeholder="Ask a question about the document..."
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
          onClick={handleAsk}
          disabled={loading || !question.trim()}
          style={{
            padding: '0.5rem 1.5rem',
            background: loading ? '#333' : '#4a9eff',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'default' : 'pointer',
          }}
        >
          {loading ? 'Asking...' : 'Ask'}
        </button>
      </div>

      {/* Answer area */}
      <div
        style={{
          background: '#1a1a1a',
          padding: '1rem',
          borderRadius: '4px',
          border: '1px solid #444',
          minHeight: '80px',
        }}
      >
        {error && <p style={{ color: '#ff5050', margin: 0 }}>Error: {error}</p>}

        {!error && !result && (
          <p style={{ color: '#666', margin: 0 }}>Cited answer here</p>
        )}

        {result && (
          <div>
            {/* Document-voice styling */}
            {result.spans.map((span, i) => (
              <span
                key={span.citation_id + i}
                title={`Citation ${span.citation_id}${
                  span.char_start != null ? ` — chars ${span.char_start}-${span.char_end}` : ''
                }`}
                style={{
                  display: 'inline-block',
                  fontFamily: 'Georgia, "Times New Roman", serif',
                  color: '#e0e0e0',
                  background: 'rgba(74, 158, 255, 0.12)',
                  border: '1px solid rgba(74, 158, 255, 0.4)',
                  borderRadius: '4px',
                  padding: '0.4rem 0.6rem',
                  marginRight: '0.4rem',
                  marginBottom: '0.4rem',
                  cursor: 'pointer',
                }}
              >
                “{span.quote}”
              </span>
            ))}

            {/* Confusion Gap */}
            {result.confessed &&
              result.gaps.map((gap, i) => (
                <div
                  key={i}
                  style={{
                    marginTop: '0.5rem',
                    padding: '0.6rem 0.75rem',
                    borderRadius: '4px',
                    color: '#999',
                    fontStyle: 'italic',
                    background:
                      'repeating-linear-gradient(135deg, #262626, #262626 6px, #1a1a1a 6px, #1a1a1a 12px)',
                    border: '1px dashed #555',
                  }}
                >
                  {gap.reason}
                </div>
              ))}

            {result.spans.length === 0 && result.gaps.length === 0 && (
              <p style={{ color: '#666', margin: 0 }}>No answer found.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default WitnessStitch;