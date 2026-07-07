import React, { useState } from 'react';

export const ArtifactEngineCanvas: React.FC = () => {
  const [activePanel, setActivePanel] = useState<'terrain' | 'duel' | 'voice' | 'provenance'>('terrain');

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
        
        {/*Terrain Survey Placeholder*/}
        <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', minHeight: '300px' }}>
          <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Terrain Survey.</h3>
          <div style={{ background: '#1a1a1a', height: '200px', borderRadius: '4px', border: '1px solid #444', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <p style={{ color: '#666' }}>Pressure heatmap here.</p>
          </div>
        </div>

        {/*Duel Arena Placeholder*/}
        <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', minHeight: '300px' }}>
          <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Duel Arena.</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', height: '200px' }}>
            <div style={{ background: '#1a1a1a', borderRadius: '4px', border: '1px solid #444', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <p style={{ color: '#666', fontSize: '0.85rem' }}>Agent A</p>
            </div>
            <div style={{ background: '#1a1a1a', borderRadius: '4px', border: '1px solid #444', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <p style={{ color: '#666', fontSize: '0.85rem' }}>Agent B</p>
            </div>
          </div>
        </div>
      </div>

      {/*Voice Query Placeholder*/}
      <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
        <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Voice Query.</h3>
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
          <input
            type="text"
            placeholder="Ask a question about the document..."
            style={{ flex: 1, padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
          />
          <button style={{ padding: '0.5rem 1.5rem', background: '#4a9eff', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
            Ask
          </button>
        </div>
        <div style={{ background: '#1a1a1a', padding: '1rem', borderRadius: '4px', border: '1px solid #444', minHeight: '80px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <p style={{ color: '#666' }}>Cited answer here</p>
        </div>
      </div>

      {/*Ghost Graph Placeholder*/}
      <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px' }}>
        <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Artifact Graph.</h3>
        <div style={{ background: '#1a1a1a', height: '150px', borderRadius: '4px', border: '1px solid #444', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <p style={{ color: '#666' }}>Ghost Graph here</p>
        </div>
      </div>
    </div>
  );
};

export default ArtifactEngineCanvas;