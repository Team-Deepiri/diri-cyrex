// Terrain Survey. Topographic heatmap over (section_id, page). Epistemic pressure as elevation,
// fault zones as red ridges.

import React, { useState } from 'react';
import { PressureCell } from '../../types/artifactEngine';

// Properties
interface TerrainSurveyProps {
  cells: PressureCell[];
  onFaultZoneClick?: (cell: PressureCell) => void;
}

// Change score (0-1) to a color between cool blue and hot red
function scoreToColor(score: number): string {
  // Deep ocean — lowest pressure, cool blue
  if (score < 0.2) {
    return `rgba(30, 100, 200, ${0.5 + score * 2})`;
  
  // Teal lowlands — mild pressure
  } else if (score < 0.4) {
    return `rgba(50, 180, 150, ${0.6 + score})`;
  
  // Yellow-green midlands — moderate pressure
  } else if (score < 0.6) {
    return `rgba(180, 200, 80, ${0.7 + score * 0.5})`;
  
  // Amber highlands — high pressure
  } else if (score < 0.8) {
    return `rgba(220, 120, 40, ${0.75 + score * 0.3})`;
  
  // Red fault peaks — critical pressure, fault zone territory
  } else {
    return `rgba(200, 50, 50, ${0.85 + score * 0.15})`;
  }
}

export const TerrainSurvey: React.FC<TerrainSurveyProps> = ({
  cells,
  onFaultZoneClick,
}) => {
  const [hoveredCell, setHoveredCell] = useState<PressureCell | null>(null);

  // Group cells by section_id
  const sections = Array.from(new Set(cells.map(c => c.section_id)));
  const pages = Array.from(new Set(cells.map(c => c.page ?? 1))).sort((a, b) => a - b);

  const getCell = (section_id: string, page: number) =>
    cells.find(c => c.section_id === section_id && (c.page ?? 1) === page);

  if (cells.length === 0) {
    return (
      <div style={{
        background: '#1a1a1a',
        borderRadius: '4px',
        border: '1px solid #444',
        height: '200px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <p style={{ color: '#666' }}>No pressure data. Upload a document first.</p>
      </div>
    );
  }

  return (
    <div>
      {/* Legend */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.75rem' }}>
        <span style={{ color: '#b0b0b0', fontSize: '0.8rem' }}>Elevation:</span>
        {[
          { color: 'rgba(30, 100, 200, 0.8)', label: 'Ocean' },
          { color: 'rgba(50, 180, 150, 0.8)', label: 'Lowlands' },
          { color: 'rgba(180, 200, 80, 0.9)', label: 'Midlands' },
          { color: 'rgba(220, 120, 40, 0.9)', label: 'Highlands' },
          { color: 'rgba(200, 50, 50, 1)', label: 'Fault Peak' },
        ].map(({ color, label }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <div style={{ width: '14px', height: '14px', borderRadius: '3px', background: color }} />
            <span style={{ color: '#b0b0b0', fontSize: '0.75rem' }}>{label}</span>
          </div>
        ))}
      </div>

      {/* Grid */}
      <div style={{ overflowX: 'auto' }}>
        {/* Section headers (X-axis) */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: `60px repeat(${sections.length}, 1fr)`,
          gap: '2px',
          marginBottom: '2px',
        }}>
          <div />
          {sections.map(section => (
            <div key={section} style={{
              color: '#888',
              fontSize: '0.7rem',
              textAlign: 'center',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              padding: '0 2px',
            }}>
              {section}
            </div>
          ))}
        </div>

        {/* Rows */}
        {pages.map(page => (
          <div key={page} style={{
            display: 'grid',
            gridTemplateColumns: `60px repeat(${sections.length}, 1fr)`,
            gap: '2px',
            marginBottom: '2px',
          }}>
            {/* Page label */}
            <div style={{
              color: '#888',
              fontSize: '0.7rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              paddingRight: '6px',
            }}>
              p.{page}
            </div>

            {/* Cells */}
            {sections.map(section => {
              const cell = getCell(section, page);
              const isFault = cell?.is_fault_zone ?? false;
              const score = cell?.score ?? 0;
              const isHovered = hoveredCell?.section_id === section && hoveredCell?.page === page;

              return (
                <div
                  key={section}
                  onClick={() => cell && isFault && onFaultZoneClick?.(cell)}
                  onMouseEnter={() => cell && setHoveredCell(cell)}
                  onMouseLeave={() => setHoveredCell(null)}
                  style={{
                    height: '48px',
                    borderRadius: '8px',
                    background: cell
                      ? `linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 50%, rgba(0,0,0,0.2) 100%), ${scoreToColor(score)}`
                      : '#1a1a1a',
                    border: isFault
                      ? '2px solid rgba(255, 80, 80, 0.9)'
                      : score > 0.6
                      ? '1px solid rgba(220, 120, 40, 0.4)'
                      : score > 0.3
                      ? '1px solid rgba(50, 180, 150, 0.3)'
                      : '1px solid rgba(30, 100, 200, 0.2)',
                    cursor: isFault ? 'pointer' : 'default',
                    transition: 'transform 0.2s, box-shadow 0.2s',
                    transform: isHovered ? 'scale(1.1) translateY(-2px)' : 'scale(1)',
                    // Layered shadows for 3D terrain depth effect
                    boxShadow: isHovered
                      ? `0 8px 20px ${scoreToColor(score)}, 0 4px 8px rgba(0,0,0,0.5), inset 0 1px 1px rgba(255,255,255,0.15)`
                      : isFault
                      ? '0 4px 12px rgba(255, 80, 80, 0.4), inset 0 1px 1px rgba(255,255,255,0.1)'
                      : score > 0.6
                      ? '0 3px 8px rgba(220, 120, 40, 0.3), inset 0 1px 1px rgba(255,255,255,0.08)'
                      : '0 2px 4px rgba(0,0,0,0.3), inset 0 1px 1px rgba(255,255,255,0.05)',
                    animation: isFault ? 'fault-pulse 1.5s infinite' : 'none',
                    position: 'relative' as const,
                  }}
                  title={cell
                    ? `${section} p.${page} — score: ${score.toFixed(2)}${isFault ? ' ⚠ FAULT ZONE' : ''}`
                    : 'No data'
                  }
                />
              );
            })}
          </div>
        ))}
      </div>

      {/* Hovered cell detail */}
      {hoveredCell && (
        <div style={{
          marginTop: '0.75rem',
          padding: '0.5rem 0.75rem',
          background: '#1a1a1a',
          borderRadius: '4px',
          border: '1px solid #444',
          fontSize: '0.8rem',
          color: '#b0b0b0',
        }}>
          <strong style={{ color: '#e0e0e0' }}>{hoveredCell.section_id}</strong>
          {' '}— page {hoveredCell.page ?? 1}
          {' '}— score: <span style={{ color: hoveredCell.score > 0.66 ? '#ff5050' : '#4a9eff' }}>
            {hoveredCell.score.toFixed(2)}
          </span>
          {hoveredCell.is_fault_zone && (
            <span style={{ color: '#ff5050', marginLeft: '0.5rem' }}>⚠ Fault Zone</span>
          )}
          {' '}— {hoveredCell.discrepancy_count} discrepancies
        </div>
      )}

      {/* Fault pulse animation */}
      <style>{`
        @keyframes fault-pulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(255, 80, 80, 0.4); }
          50% { box-shadow: 0 0 0 4px rgba(255, 80, 80, 0); }
        }
      `}</style>
    </div>
  );
};

export default TerrainSurvey;