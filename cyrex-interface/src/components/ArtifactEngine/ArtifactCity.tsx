// Artifact City VIZ-18 replaces VIZ-09 Ghost Graph.
// Artifact nodes grouped by ArtifactType around a
// central document hub, with a live-scrolling feed panel.

import React, { useEffect, useMemo, useState } from 'react';
import { ArtifactBundle, ArtifactType } from '../../types/artifactEngine';

interface ArtifactCityProps {
    documentId: string;
    artifacts: ArtifactBundle[];
}

// --- Event row shapes -------------------------------------------------

interface PipelineRunEvent {
    event_id: string;
    run_id: string;
    event_type: string;
    payload_json: Record<string, unknown>;
  // TODO: swap event_id for a real timestamp field
}

interface PressureEventRow {
    event_id: string;
    event_type: string;
    document_id: string;
    section_id: string;
    page: number;
    artifact_id: string;
    payload_json: Record<string, unknown>;
}

interface InvalidationEvent {
  // TODO: confirm with producer shape
    artifact_id: string;
    document_id: string;
    reason: string;
    superseded_by: string | null;
    invalidated_at: string;
}

type FeedEvent =
    | { kind: 'run'; data: PipelineRunEvent }
    | { kind: 'pressure'; data: PressureEventRow }
    | { kind: 'invalidation'; data: InvalidationEvent };

// --- Mock fixtures --------------------------

const MOCK_RUN_EVENTS: PipelineRunEvent[] = [
    { event_id: 'pre_001', run_id: 'run_001', event_type: 'parse.completed', payload_json: { document_id: 'lease_001' } },
    { event_id: 'pre_002', run_id: 'run_001', event_type: 'extract.pass_completed', payload_json: { pass: 2, fields_found: 14 } },
    { event_id: 'pre_003', run_id: 'run_001', event_type: 'duel.disagreement_found', payload_json: { field_name: 'notice_period' } },
];

const MOCK_PRESSURE_EVENTS: PressureEventRow[] = [
    { event_id: 'prs_001', event_type: 'reflect_failure', document_id: 'lease_001', section_id: 'financial_terms', page: 1, artifact_id: 'art_001', payload_json: { confidence: 0.42 } },
    { event_id: 'prs_002', event_type: 'low_confidence_field', document_id: 'lease_001', section_id: 'termination_clause', page: 2, artifact_id: 'art_002', payload_json: { field_name: 'notice_period' } },
];

const MOCK_INVALIDATION_EVENTS: InvalidationEvent[] = [
    { artifact_id: 'art_003', document_id: 'lease_001', reason: 'superseded by correction', superseded_by: 'art_005', invalidated_at: '2026-08-21T18:02:00Z' },
];

// --- Fixed radial clustering -----------------------------------

const CLUSTER_TYPES: ArtifactType[] = [
    ArtifactType.CANONICAL,
    ArtifactType.EXTRACTION,
    ArtifactType.REASONING,
    ArtifactType.ANSWER,
];

const CLUSTER_COLORS: Record<string, string> = {
    [ArtifactType.CANONICAL]: '#4a9eff',
    [ArtifactType.EXTRACTION]: '#50c878',
    [ArtifactType.REASONING]: '#cc44ff',
    [ArtifactType.ANSWER]: '#ffaa00',
    OTHER: '#888',
};

const VIEWBOX = 500;
const CENTER = VIEWBOX / 2;
const CLUSTER_RADIUS = 150; // distance of each cluster center from the hub
const NODE_SPREAD = 45; // radius of the small circle nodes sit on within a cluster

function clusterAngle(index: number, total: number): number {
    // Start at top, go clockwise
    return (index / total) * 2 * Math.PI - Math.PI / 2;
}

function nodePosition(clusterIndex: number, nodeIndex: number, nodeCount: number) {
    const cAngle = clusterAngle(clusterIndex, CLUSTER_TYPES.length);
    const cx = CENTER + CLUSTER_RADIUS * Math.cos(cAngle);
    const cy = CENTER + CLUSTER_RADIUS * Math.sin(cAngle);

    if (nodeCount === 1)
        return { x: cx, y: cy };

    const nAngle = (nodeIndex / nodeCount) * 2 * Math.PI;
    return {
        x: cx + NODE_SPREAD * Math.cos(nAngle),
        y: cy + NODE_SPREAD * Math.sin(nAngle),
    };
}

export const ArtifactCity: React.FC<ArtifactCityProps> = ({ documentId, artifacts }) => {
    const [highlightedArtifact, setHighlightedArtifact] = useState<string | null>(null);

  // Merge and sort all three event streams into one feed, newest first.

  // TODO: replace with a real sort-by-time once timestamp column exists.
    const feed: FeedEvent[] = useMemo(() => {
        const runEvents: FeedEvent[] = MOCK_RUN_EVENTS.map((data) => ({ kind: 'run', data }));
        const pressureEvents: FeedEvent[] = MOCK_PRESSURE_EVENTS.map((data) => ({ kind: 'pressure', data }));
        const invalidationEvents: FeedEvent[] = MOCK_INVALIDATION_EVENTS.map((data) => ({ kind: 'invalidation', data }));
        return [...invalidationEvents, ...pressureEvents, ...runEvents];
    }, []);

  // Pulse-highlight whichever artifact the most recent feed event touches.
    useEffect(() => {
        const latest = feed[0];
        if (!latest)
            return;
        const artifactId =
        latest.kind === 'pressure' || latest.kind === 'invalidation' ? latest.data.artifact_id : null;
        if (artifactId) {
            setHighlightedArtifact(artifactId);
            const timeout = setTimeout(() => setHighlightedArtifact(null), 2000);
            return () => clearTimeout(timeout);
        }
    }, [feed]);

    const clusters = CLUSTER_TYPES.map((type) => ({
        type,
        nodes: artifacts.filter((a) => a.artifact_type === type),
    }));
    const otherNodes = artifacts.filter((a) => !CLUSTER_TYPES.includes(a.artifact_type));

    return (
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem' }}>
        {/* Radial map */}
        <div style={{ background: '#1a1a1a', borderRadius: '8px', border: '1px solid #333', padding: '1rem' }}>
            <svg viewBox={`0 0 ${VIEWBOX} ${VIEWBOX}`} style={{ width: '100%', height: 'auto' }}>
            {/* Lines from hub to each cluster center */}
            {CLUSTER_TYPES.map((type, i) => {
                const angle = clusterAngle(i, CLUSTER_TYPES.length);
                const x = CENTER + CLUSTER_RADIUS * Math.cos(angle);
                const y = CENTER + CLUSTER_RADIUS * Math.sin(angle);
                return (
                <line
                    key={`spoke-${type}`}
                    x1={CENTER}
                    y1={CENTER}
                    x2={x}
                    y2={y}
                    stroke="#333"
                    strokeWidth={1}
                />
                );
            })}

            {/* Cluster labels */}
            {CLUSTER_TYPES.map((type, i) => {
                const angle = clusterAngle(i, CLUSTER_TYPES.length);
                const x = CENTER + (CLUSTER_RADIUS + 65) * Math.cos(angle);
                const y = CENTER + (CLUSTER_RADIUS + 65) * Math.sin(angle);
                return (
                <text
                    key={`label-${type}`}
                    x={x}
                    y={y}
                    fill={CLUSTER_COLORS[type]}
                    fontSize="11"
                    fontWeight={700}
                    textAnchor="middle"
                    style={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}
                >
                    {type}
                </text>
                );
            })}

            {/* Document hub */}
            <circle cx={CENTER} cy={CENTER} r={28} fill="#2a2a2a" stroke="#4a9eff" strokeWidth={2} />
            <text x={CENTER} y={CENTER + 4} fill="#e0e0e0" fontSize="10" textAnchor="middle">
                {documentId}
            </text>

            {/* Artifact nodes */}
            {clusters.map((cluster, ci) =>
                cluster.nodes.map((artifact, ni) => {
                const { x, y } = nodePosition(ci, ni, cluster.nodes.length);
                const isGhost = artifact.is_deleted;
                const isHighlighted = artifact.artifact_id === highlightedArtifact;
                const color = CLUSTER_COLORS[cluster.type];

                return (
                    <circle
                    key={artifact.artifact_id}
                    cx={x}
                    cy={y}
                    r={isHighlighted ? 10 : 7}
                    fill={isGhost ? 'transparent' : color}
                    stroke={color}
                    strokeWidth={isGhost ? 1.5 : 0}
                    strokeDasharray={isGhost ? '2,2' : undefined}
                    opacity={isGhost ? 0.4 : 1}
                    style={{
                        transition: 'r 0.3s ease, opacity 0.3s ease',
                        filter: isHighlighted ? `drop-shadow(0 0 6px ${color})` : undefined,
                    }}
                    >
                    <title>
                        {artifact.artifact_id} — {artifact.artifact_type}
                        {isGhost ? ' (ghost — superseded)' : ''}
                    </title>
                    </circle>
                );
                })
            )}

            {/* Anything not in the 4 named clusters is placed in a fifth informal ring. */}
            {otherNodes.map((artifact, i) => {
                const angle = (i / Math.max(otherNodes.length, 1)) * 2 * Math.PI;
                const r = CLUSTER_RADIUS + 90;
                const x = CENTER + r * Math.cos(angle);
                const y = CENTER + r * Math.sin(angle);
                return (
                <circle
                    key={artifact.artifact_id}
                    cx={x}
                    cy={y}
                    r={5}
                    fill={CLUSTER_COLORS.OTHER}
                    opacity={artifact.is_deleted ? 0.3 : 0.8}
                >
                    <title>{artifact.artifact_id} — {artifact.artifact_type}</title>
                </circle>
                );
            })}
            </svg>
        </div>

        {/* Live feed panel. */}
        <div
            style={{
            background: '#1a1a1a',
            borderRadius: '8px',
            border: '1px solid #333',
            padding: '0.75rem',
            maxHeight: '420px',
            overflowY: 'auto',
            }}
        >
            <p style={{ color: '#888', fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: 0 }}>
            Live Events
            </p>
            {feed.length === 0 && <p style={{ color: '#666', fontSize: '0.8rem' }}>No events yet.</p>}
            {feed.map((event, i) => (
            <FeedRow key={i} event={event} />
            ))}
        </div>
        </div>
    );
    };

    const FeedRow: React.FC<{ event: FeedEvent }> = ({ event }) => {
    let label: string;
    let color: string;

    if (event.kind === 'run') {
        label = event.data.event_type;
        color = '#4a9eff';
    } else if (event.kind === 'pressure') {
        label = `${event.data.event_type} · ${event.data.section_id}`;
        color = '#ffaa00';
    } else {
        label = `invalidated · ${event.data.reason}`;
        color = '#ff5050';
    }

    return (
        <div
        style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.35rem 0',
            borderBottom: '1px solid #262626',
            fontSize: '0.75rem',
        }}
        >
        <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: color, flexShrink: 0 }} />
        <span style={{ color: '#b0b0b0' }}>{label}</span>
        </div>
    );
};

export default ArtifactCity;