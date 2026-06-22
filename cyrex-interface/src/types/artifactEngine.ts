// Artifact Engine TypeScript Types. Based on app/pipeline/contracts/models.py

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
  section_id: string;
  score: number;
  is_fault_zone: boolean;
  drill_down_artifact_ids: string[];
}

// Result of two agents analyzing the same document.
export interface DuelState {
  agent_a_fields: Record<string, any>;
  agent_b_fields: Record<string, any>;
  disagreements: FieldDiscrepancy[];
  resolution_status: string;
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
export interface ConfessionGap {
  // TODO: ConfessionGap model not found
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
  gaps?: ConfessionGap[];
}

// Response object for artifact endpoints.
export interface ArtifactBundle {
  // TODO: confirm shape with Tyler
  artifact_id: string;
  document_id: string;
  artifact_type: string;
  is_deleted: boolean;
  citations: Citation[];
}