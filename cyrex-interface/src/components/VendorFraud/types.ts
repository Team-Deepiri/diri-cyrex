/**
 * Cyrex Vendor Fraud Detection Types
 */

export type Industry = 
  | 'property_management'
  | 'corporate_procurement'
  | 'insurance_pc'
  | 'general_contractors'
  | 'retail_ecommerce'
  | 'law_firms';

export interface LineItem {
  description: string;
  quantity: number;
  unit_price: number;
  total?: number;
}

export interface InvoiceData {
  vendor_name: string;
  vendor_id?: string;
  invoice_number?: string;
  invoice_date?: string;
  total_amount: number;
  line_items: LineItem[];
  service_category?: string;
  property_address?: string;
  work_description?: string;
}

export interface FraudIndicator {
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  evidence?: string | unknown;
}

export interface PricingAnalysis {
  invoice_total: number;
  service_category: string;
  industry: string;
  price_deviation_percent: number;
  overpriced_items: unknown[];
  line_item_analysis: unknown[];
}

export interface VendorProfile {
  vendor_id: string;
  vendor_name: string;
  known_industries: string[];
  fraud_flags_count: number;
  previous_flags?: boolean;
  rag_context?: string;
}

export interface RiskAssessment {
  risk_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  confidence_score: number;
  indicators_count: number;
  assessment_date: string;
  industry: string;
}

export interface InvoiceAnalysisResult {
  success: boolean;
  workflow_id: string;
  industry: string;
  fraud_detected: boolean;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  risk_score: number;
  confidence_score: number;
  fraud_indicators: FraudIndicator[];
  recommendations: string[];
  extracted_data?: unknown;
  vendor_profile?: VendorProfile;
  pricing_analysis?: PricingAnalysis;
  risk_assessment?: RiskAssessment;
  analyzed_at: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface IndustryInfo {
  id: Industry;
  name: string;
  description: string;
  vendor_types: string[];
}

export interface PricingBenchmark {
  service_type: string;
  industry: string;
  found: boolean;
  benchmark?: {
    min: number;
    avg: number;
    max: number;
    unit: string;
  };
  price_checked?: number;
  deviation_percent?: number;
  status?: 'above_market' | 'below_market' | 'within_range';
}




