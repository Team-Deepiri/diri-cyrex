// Voice answers are not generative. When a claim can't be grounded, it renders as a grey void.
// Audio I/O is deepiri-speech (STT/TTS); grounding stays VoiceSynthesizer.

import React, { useEffect, useRef, useState } from 'react';
import { VoiceQueryRequest, PersonaScope } from '../../types/artifactEngine';
import { speechHealth, voiceQuery } from '../../api/artifactEngine';

interface WitnessStitchProps {
  documentId: string;
  personaScope?: PersonaScope;
}

const DEFAULT_PERSONA_SCOPE: PersonaScope = {
  witness_set_only: true,
  hard_citation_gate: true,
  corpus_filter: [],
};

function playBase64Audio(b64: string, mime: string) {
  const src = `data:${mime};base64,${b64}`;
  const audio = new Audio(src);
  void audio.play().catch(() => {
    /* autoplay may be blocked until user gesture — Speak button covers that */
  });
}

export const WitnessStitch: React.FC<WitnessStitchProps> = ({
  documentId,
  personaScope = DEFAULT_PERSONA_SCOPE,
}) => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Awaited<ReturnType<typeof voiceQuery>> | null>(null);
  const [speechOk, setSpeechOk] = useState<boolean | null>(null);
  const [recording, setRecording] = useState(false);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    speechHealth()
      .then((h) => setSpeechOk(Boolean(h.ok)))
      .catch(() => setSpeechOk(false));
  }, []);

  const handleAsk = async (opts?: { audio_b64?: string; audio_mime_type?: string }) => {
    if (!opts?.audio_b64 && !question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const request: VoiceQueryRequest & {
        audio_b64?: string;
        audio_mime_type?: string;
        synthesize_audio?: boolean;
      } = {
        document_id: documentId,
        question: question.trim() || undefined,
        persona_scope: personaScope,
        synthesize_audio: true,
        ...opts,
      };
      const response = await voiceQuery(request);
      setResult(response);
      if (response.question_used && !question.trim()) {
        setQuestion(response.question_used);
      }
      if (response.audio_b64 && response.audio_mime_type) {
        playBase64Audio(response.audio_b64, response.audio_mime_type);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Voice query failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const toggleMic = async () => {
    if (recording && mediaRef.current) {
      mediaRef.current.stop();
      setRecording(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size) chunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        const buf = await blob.arrayBuffer();
        const bytes = new Uint8Array(buf);
        let binary = '';
        const chunk = 0x8000;
        for (let i = 0; i < bytes.length; i += chunk) {
          binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
        }
        const b64 = btoa(binary);
        await handleAsk({ audio_b64: b64, audio_mime_type: blob.type || 'audio/webm' });
      };
      mediaRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Microphone unavailable');
    }
  };

  const replay = () => {
    if (result?.audio_b64 && result.audio_mime_type) {
      playBase64Audio(result.audio_b64, result.audio_mime_type);
    }
  };

  return (
    <div>
      <div
        style={{
          marginBottom: '0.75rem',
          fontSize: '0.85rem',
          color: speechOk === null ? '#888' : speechOk ? '#6d6' : '#c66',
        }}
      >
        Speech engine:{' '}
        {speechOk === null ? 'checking…' : speechOk ? 'deepiri-speech online' : 'unreachable (TTS/STT offline)'}
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
          placeholder="Ask a question about the document..."
          style={{
            flex: 1,
            minWidth: '200px',
            padding: '0.5rem',
            background: '#1a1a1a',
            color: '#e0e0e0',
            border: '1px solid #444',
            borderRadius: '4px',
          }}
        />
        <button
          onClick={() => handleAsk()}
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
        <button
          onClick={toggleMic}
          disabled={loading}
          title="Record question → STT via deepiri-speech"
          style={{
            padding: '0.5rem 1rem',
            background: recording ? '#a33' : '#333',
            color: '#fff',
            border: '1px solid #555',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          {recording ? 'Stop' : 'Mic'}
        </button>
        <button
          onClick={replay}
          disabled={!result?.audio_b64}
          style={{
            padding: '0.5rem 1rem',
            background: '#333',
            color: '#fff',
            border: '1px solid #555',
            borderRadius: '4px',
            cursor: result?.audio_b64 ? 'pointer' : 'default',
            opacity: result?.audio_b64 ? 1 : 0.4,
          }}
        >
          Speak
        </button>
      </div>

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
            {result.spoken_text && (
              <p style={{ color: '#aaa', marginTop: 0, marginBottom: '0.75rem', fontSize: '0.9rem' }}>
                Spoken: {result.spoken_text}
              </p>
            )}
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

            {result.confessed &&
              (result.gaps || []).map((gap, i) => (
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

            {result.spans.length === 0 && (!result.gaps || result.gaps.length === 0) && (
              <p style={{ color: '#666', margin: 0 }}>No answer found.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default WitnessStitch;
