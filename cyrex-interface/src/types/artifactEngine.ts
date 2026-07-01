// Artifact Engine TypeScript Types. Based on app/pipeline/contracts/models.py

// Artifacts produced by the pipeline.
export enum ArtifactType {
  CANONICAL = "canonical",
  EXTRACTION = "extraction",
  REASONING = "reasoning",
  RETRIEVAL = "retrieval",
  ANSWER = "answer",
  TRANSFORMATION = "transformation",
  WORKFLOW = "workflow",
  LEARNING = "learning",
  SYSTEM = "system"
}

// Field extraction methods.
export enum ExtractionMethod {
  REGEX = "regex",
  LLM = "llm",
  CROSS_REF = "cross_ref",
  PATTERN = "pattern",
  VISION = "vision"
}

// Prediction status relative to actual data.
export enum PredictionStatus {
  NO_PRIOR = "no_prior",
  CONFIRMED = "confirmed",
  ANOMALOUS = "anomalous",
  NOVEL = "novel"
}

// Resolution status for a disagreement.
export enum DuelResolutionStatus {
  UNRESOLVED = "unresolved",
  RESOLVED = "resolved",
  IGNORED = "ignored"
}


// Location of a quote in a PDF. Can be a character range, page range, or element ID.
export interface CitationLocator {
  locator_type: "char_range" | "page_range" | "element_id";
  char_start?: number;
  char_end?: number;
  page_start?: number;
  page_end?: number;
  element_id?: string;
}

// Quote with its location, confidence score, and which document it came from.
export interface Citation {
  citation_id: string;
  document_id: string;
  source_doc_hash: string;
  locator: CitationLocator;
  quote: string;
  confidence: number;
  extraction_pass?: number;
}

// Rules for the voice query.
export interface PersonaScope {
  witness_set_only: boolean;
  hard_citation_gate: boolean;
  corpus_filter: string[];
}

// Section of the document with a pressure score and whether or not it's a fault zone.
export interface PressureCell {
  document_id: string;
  section_id: string;
  page?: number;
  discrepancy_count: number;
  reflect_failures: number;
  low_confidence_count: number;
  duel_disagreements: number;
  score: number;
  is_fault_zone: boolean;
  drill_down_artifact_ids: string[];
}

// Result of two agents analyzing the same document.
export interface DuelState {
  document_id: string;
  artifact_id?: string;
  agent_a_id: string;
  agent_b_id: string;
  agent_a_fields: CitedField[];
  agent_b_fields: CitedField[];
  disagreements: FieldDiscrepancy[];
  resolution_status: DuelResolutionStatus;
  resolution_artifact_id?: string;
}

// Single disagreement on one field.
export interface FieldDiscrepancy {
  field_name: string;
  pass_a_value?: any;
  pass_b_value?: any;
  agent_a_value?: any;
  agent_b_value?: any;
  agent_a_confidence?: number;
  agent_b_confidence?: number;
  confidence_delta?: number;
  reason?: string;
}

// Text in voice form.
export interface WitnessSpan {
  citation: Citation;
}

// When no factual answer is found.
export interface ConfusionGap {
  // TODO: fill in later once interface is created
}

// Voice request.
export interface VoiceQueryRequest {
  document_id: string;
  question: string;
  persona_scope: PersonaScope;
}

// Voice response.
export interface VoiceQueryResponse {
  confessed: boolean;
  spans: WitnessSpan[];
  gaps?: ConfusionGap[];
}

// Response object for artifact endpoints.
export interface ArtifactBundle {
  artifact_id: string;
  document_id: string;
  version: number;
  artifact_type: ArtifactType;
  source_doc_hash: string;
  confidence: number;
  payload: Record<string, any>;
  provenance: Provenance;
  citations: Citation[];
  created_at: string;
  is_deleted: boolean;
}

// Extracted field backed by citations.
export interface CitedField {
  field_name: string;
  value: any;
  value_type: string;
  citations: Citation[];
  confidence: number;
  referenced_by: string[];
  references: string[];
}

// Extraction pass with metadata.
export interface ProvenancePass {
  pass_number: number;
  method: ExtractionMethod;
  fields_extracted: string[];
  prompt_version?: string;
  extraction_time_ms?: number;
}

// Full provenance trail for an artifact. Shows where it came from and what it depends on.
export interface Provenance {
  source_doc_hash: string;
  document_id: string;
  version: number;
  model_id?: string;
  passes: ProvenancePass[];
  depends_on: string[];
  depended_on_by: string[];
  cross_references: string[];
  synthesized_from: string[];
}

// Human correction stored for Helox.
export interface LearningArtifact {
  artifact_id: string;
  document_id: string;
  field_name: string;
  original_value: any;
  corrected_value: any;
  corrected_citation: Citation;
  actor_id: string;
  timestamp: string;
}

// Result from multi-pass extraction synthesis.
export interface SynthesisResult {
  document_id: string;
  source_doc_hash: string;
  final_fields: CitedField[];
  all_citations: Citation[];
  confidence: number;
  passes: ProvenancePass[];
  provenance: Provenance;
  discrepancies: FieldDiscrepancy[];
}

// Issue found during reflection/validation.
export interface ReflectionIssue {
  code: string;
  severity: "info" | "warning" | "error";
  field_name?: string;
  message: string;
  citation_id?: string;
}

// Result from ReflectTool.
export interface ReflectionResult {
  passed: boolean;
  issues: ReflectionIssue[];
  low_confidence_fields: string[];
  missing_citation_fields: string[];
  unverifiable_citations: string[];
  confidence_floor: number;
}

// A prior prediction for a field, updated with the actual post-extraction.
export interface PredictionRecord {
  field_name: string;
  predicted_range?: Record<string, number>;
  predicted_mean?: number;
  actual_value?: any;
  sigma_delta?: number;
  status: PredictionStatus;
  corpus_doc_count: number;
  last_prior_update?: string;
}

