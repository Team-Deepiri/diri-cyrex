/**
 * Cyrex Vendor Fraud Detection Panel
 * Main UI component for vendor fraud analysis
 */
import React, { useState, useCallback, useEffect } from 'react';
import { vendorFraudApi } from './api';
import type {
  Industry,
  InvoiceData,
  LineItem,
  InvoiceAnalysisResult,
  ChatMessage,
  IndustryInfo,
} from './types';
import {
  BuildingIcon,
  PackageIcon,
  ShieldIcon,
  ConstructionIcon,
  ShoppingCartIcon,
  ScalesIcon,
  SearchIcon,
  ChartIcon,
  WarningIcon,
  RefreshIcon,
  DocumentIcon,
  GlobeIcon,
  ChatIcon,
  BooksIcon,
  MoneyIcon,
  DownloadIcon,
  WaveIcon,
  CheckIcon,
  ArrowRightIcon,
} from './VendorFraudIcons';
import './VendorFraudPanel.css';

// Risk level colors
const RISK_COLORS: Record<string, string> = {
  low: '#4CAF50',
  medium: '#FF9800',
  high: '#f44336',
  critical: '#9C27B0',
};

// Industry icons
const INDUSTRY_ICONS: Record<Industry, React.ComponentType<{ size?: number; className?: string }>> = {
  property_management: BuildingIcon,
  corporate_procurement: PackageIcon,
  insurance_pc: ShieldIcon,
  general_contractors: ConstructionIcon,
  retail_ecommerce: ShoppingCartIcon,
  law_firms: ScalesIcon,
};

interface Props {
  baseUrl?: string;
  apiKey?: string;
}

export const VendorFraudPanel: React.FC<Props> = ({ baseUrl, apiKey }) => {
  // State
  const [activeTab, setActiveTab] = useState<'dashboard' | 'analyze' | 'vendors' | 'analytics' | 'chat' | 'documents' | 'benchmarks'>('dashboard');
  const [industry, setIndustry] = useState<Industry>('property_management');
  const [industries, setIndustries] = useState<IndustryInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Invoice analysis state
  const [invoiceData, setInvoiceData] = useState<InvoiceData>({
    vendor_name: '',
    vendor_id: '',
    invoice_number: '',
    invoice_date: new Date().toISOString().split('T')[0],
    total_amount: 0,
    line_items: [],
    service_category: '',
    work_description: '',
  });
  const [analysisResult, setAnalysisResult] = useState<InvoiceAnalysisResult | null>(null);

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [sessionId] = useState(() => `session_${Date.now()}`);

  // Document ingestion state
  const [docContent, setDocContent] = useState('');
  const [docTitle, setDocTitle] = useState('');
  const [ingestionResult, setIngestionResult] = useState<string | null>(null);

  // Benchmark state
  const [serviceType, setServiceType] = useState('hvac_repair');
  const [priceToCheck, setPriceToCheck] = useState<number>(0);
  const [benchmarkResult, setBenchmarkResult] = useState<unknown>(null);

  // Configure API
  useEffect(() => {
    if (baseUrl) vendorFraudApi.setBaseUrl(baseUrl);
    if (apiKey) vendorFraudApi.setApiKey(apiKey);
  }, [baseUrl, apiKey]);

  // Load industries on mount
  useEffect(() => {
    const loadIndustries = async () => {
      try {
        const result = await vendorFraudApi.listIndustries();
        setIndustries(result.industries);
      } catch (err) {
        console.error('Failed to load industries:', err);
      }
    };
    loadIndustries();
  }, []);

  // Add line item
  const addLineItem = useCallback(() => {
    setInvoiceData(prev => ({
      ...prev,
      line_items: [
        ...prev.line_items,
        { description: '', quantity: 1, unit_price: 0, total: 0 },
      ],
    }));
  }, []);

  // Update line item
  const updateLineItem = useCallback((index: number, field: keyof LineItem, value: string | number) => {
    setInvoiceData(prev => {
      const newItems = [...prev.line_items];
      newItems[index] = { ...newItems[index], [field]: value };
      // Recalculate total
      if (field === 'quantity' || field === 'unit_price') {
        newItems[index].total = Number(newItems[index].quantity) * Number(newItems[index].unit_price);
      }
      // Update invoice total
      const newTotal = newItems.reduce((sum, item) => sum + (item.total || 0), 0);
      return { ...prev, line_items: newItems, total_amount: newTotal };
    });
  }, []);

  // Remove line item
  const removeLineItem = useCallback((index: number) => {
    setInvoiceData(prev => {
      const newItems = prev.line_items.filter((_, i) => i !== index);
      const newTotal = newItems.reduce((sum, item) => sum + (item.total || 0), 0);
      return { ...prev, line_items: newItems, total_amount: newTotal };
    });
  }, []);

  // Analyze invoice
  const analyzeInvoice = useCallback(async () => {
    if (!invoiceData.vendor_name || invoiceData.total_amount <= 0) {
      setError('Please enter vendor name and at least one line item');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await vendorFraudApi.analyzeInvoice(invoiceData, industry, sessionId);
      setAnalysisResult(result.analysis);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [invoiceData, industry, sessionId]);

  // Send chat message
  const sendChatMessage = useCallback(async () => {
    if (!chatInput.trim()) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: chatInput,
      timestamp: new Date().toISOString(),
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setLoading(true);

    try {
      const result = await vendorFraudApi.chat(chatInput, industry, sessionId);
      
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: result.response,
        timestamp: new Date().toISOString(),
        metadata: { confidence: result.confidence, tool_calls: result.tool_calls },
      };

      setChatMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: ChatMessage = {
        role: 'system',
        content: `Error: ${err instanceof Error ? err.message : 'Failed to get response'}`,
        timestamp: new Date().toISOString(),
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  }, [chatInput, industry, sessionId]);

  // Ingest document
  const ingestDocument = useCallback(async () => {
    if (!docContent.trim() || !docTitle.trim()) {
      setError('Please enter document title and content');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await vendorFraudApi.ingestDocument(docContent, docTitle, industry);
      setIngestionResult(`Document "${docTitle}" indexed successfully (ID: ${result.document_id})`);
      setDocContent('');
      setDocTitle('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ingestion failed');
    } finally {
      setLoading(false);
    }
  }, [docContent, docTitle, industry]);

  // Check benchmark
  const checkBenchmark = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await vendorFraudApi.checkPricingBenchmark(
        serviceType,
        industry,
        priceToCheck > 0 ? priceToCheck : undefined
      );
      setBenchmarkResult(result.benchmark);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Benchmark check failed');
    } finally {
      setLoading(false);
    }
  }, [serviceType, industry, priceToCheck]);

  return (
    <div className="vendor-fraud-panel">
      {/* Header */}
      <div className="vf-header">
        <h2><SearchIcon size={24} /> <span>Cyrex Vendor Fraud Detection</span></h2>
        <p>AI-powered fraud analysis across six industries</p>
      </div>

      {/* Industry Selector */}
      <div className="vf-industry-selector">
        <label>Industry:</label>
        <select value={industry} onChange={(e) => setIndustry(e.target.value as Industry)}>
          {industries.map((ind) => {
            const IconComponent = INDUSTRY_ICONS[ind.id as Industry];
            return (
              <option key={ind.id} value={ind.id}>
                {ind.name}
              </option>
            );
          })}
          {industries.length === 0 && (
            <>
              <option value="property_management">Property Management</option>
              <option value="corporate_procurement">Corporate Procurement</option>
              <option value="insurance_pc">P&C Insurance</option>
              <option value="general_contractors">General Contractors</option>
              <option value="retail_ecommerce">Retail & E-Commerce</option>
              <option value="law_firms">Law Firms</option>
            </>
          )}
        </select>
      </div>

      {/* Tabs */}
      <div className="vf-tabs">
        <button
          className={activeTab === 'dashboard' ? 'active' : ''}
          onClick={() => setActiveTab('dashboard')}
        >
          <ChartIcon size={16} /> Dashboard
        </button>
        <button
          className={activeTab === 'analyze' ? 'active' : ''}
          onClick={() => setActiveTab('analyze')}
        >
          <SearchIcon size={16} /> Analyze Invoice
        </button>
        <button
          className={activeTab === 'vendors' ? 'active' : ''}
          onClick={() => setActiveTab('vendors')}
        >
          <BuildingIcon size={16} /> Vendor Intelligence
        </button>
        <button
          className={activeTab === 'analytics' ? 'active' : ''}
          onClick={() => setActiveTab('analytics')}
        >
          <ChartIcon size={16} /> Analytics
        </button>
        <button
          className={activeTab === 'chat' ? 'active' : ''}
          onClick={() => setActiveTab('chat')}
        >
          <ChatIcon size={16} /> Chat
        </button>
        <button
          className={activeTab === 'documents' ? 'active' : ''}
          onClick={() => setActiveTab('documents')}
        >
          <BooksIcon size={16} /> Documents
        </button>
        <button
          className={activeTab === 'benchmarks' ? 'active' : ''}
          onClick={() => setActiveTab('benchmarks')}
        >
          <MoneyIcon size={16} /> Benchmarks
        </button>
      </div>

      {/* Error display */}
      {error && (
        <div className="vf-error">
          <WarningIcon size={18} /> {error}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* Tab Content */}
      <div className="vf-content">
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="vf-dashboard-tab">
            <h3><ChartIcon size={24} /> Cyrex Intelligence Dashboard</h3>
            <p>Complete vendor intelligence and analytics overview</p>
            
            <div className="vf-dashboard-stats">
              <div className="vf-stat-card">
                <div className="vf-stat-icon"><DocumentIcon size={32} /></div>
                <div className="vf-stat-info">
                  <div className="vf-stat-value">Loading...</div>
                  <div className="vf-stat-label">Invoices Analyzed</div>
                </div>
              </div>
              <div className="vf-stat-card">
                <div className="vf-stat-icon"><WarningIcon size={32} /></div>
                <div className="vf-stat-info">
                  <div className="vf-stat-value">Loading...</div>
                  <div className="vf-stat-label">Fraud Detected</div>
                </div>
              </div>
              <div className="vf-stat-card">
                <div className="vf-stat-icon"><BuildingIcon size={32} /></div>
                <div className="vf-stat-info">
                  <div className="vf-stat-value">Loading...</div>
                  <div className="vf-stat-label">Vendors Tracked</div>
                </div>
              </div>
              <div className="vf-stat-card">
                <div className="vf-stat-icon"><GlobeIcon size={32} /></div>
                <div className="vf-stat-info">
                  <div className="vf-stat-value">Loading...</div>
                  <div className="vf-stat-label">Network Effects</div>
                </div>
              </div>
            </div>

            <div className="vf-dashboard-actions">
              <button onClick={() => setActiveTab('analyze')} className="vf-action-btn">
                <SearchIcon size={18} /> Analyze New Invoice
              </button>
              <button onClick={() => setActiveTab('vendors')} className="vf-action-btn">
                <BuildingIcon size={18} /> View Vendor Database
              </button>
              <button onClick={() => setActiveTab('analytics')} className="vf-action-btn">
                <ChartIcon size={18} /> View Analytics
              </button>
            </div>
          </div>
        )}

        {/* Vendor Intelligence Tab */}
        {activeTab === 'vendors' && (
          <div className="vf-vendors-tab">
            <h3><BuildingIcon size={24} /> Vendor Intelligence Database</h3>
            <p>Cross-industry vendor tracking and risk assessment</p>
            
            <div className="vf-vendors-search">
              <input
                type="text"
                placeholder="Search vendors by name..."
                style={{ width: '100%', padding: '12px', marginBottom: '16px' }}
              />
              <button onClick={() => {}} className="vf-search-btn">
                <SearchIcon size={18} /> Search
              </button>
            </div>

            <div className="vf-vendors-list">
              <p style={{ color: '#888', textAlign: 'center', padding: '40px' }}>
                Vendor intelligence database will display here.
                <br />
                Search for vendors to see cross-industry tracking, risk scores, and fraud history.
              </p>
            </div>
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div className="vf-analytics-tab">
            <h3><ChartIcon size={24} /> Comprehensive Analytics</h3>
            <p>Performance metrics, network effects, and insights</p>
            
            <div className="vf-analytics-grid">
              <div className="vf-analytics-card">
                <h4>Fraud Detection Rate</h4>
                <div className="vf-metric-value">Loading...</div>
                <div className="vf-metric-label">Detection accuracy</div>
              </div>
              <div className="vf-analytics-card">
                <h4>Network Effects</h4>
                <div className="vf-metric-value">Loading...</div>
                <div className="vf-metric-label">Cross-industry flags</div>
              </div>
              <div className="vf-analytics-card">
                <h4>Total Value Protected</h4>
                <div className="vf-metric-value">Loading...</div>
                <div className="vf-metric-label">Amount analyzed</div>
              </div>
            </div>
          </div>
        )}

        {/* Analyze Invoice Tab */}
        {activeTab === 'analyze' && (
          <div className="vf-analyze-tab">
            <div className="vf-form-section">
              <h3>Invoice Details</h3>
              
              <div className="vf-form-row">
                <label>Vendor Name *</label>
                <input
                  type="text"
                  value={invoiceData.vendor_name}
                  onChange={(e) => setInvoiceData(prev => ({ ...prev, vendor_name: e.target.value }))}
                  placeholder="e.g., ABC Plumbing Services"
                />
              </div>

              <div className="vf-form-row">
                <label>Invoice Number</label>
                <input
                  type="text"
                  value={invoiceData.invoice_number || ''}
                  onChange={(e) => setInvoiceData(prev => ({ ...prev, invoice_number: e.target.value }))}
                  placeholder="e.g., INV-2026-001"
                />
              </div>

              <div className="vf-form-row">
                <label>Service Category</label>
                <select
                  value={invoiceData.service_category || ''}
                  onChange={(e) => setInvoiceData(prev => ({ ...prev, service_category: e.target.value }))}
                >
                  <option value="">Select category...</option>
                  <option value="hvac_repair">HVAC Repair</option>
                  <option value="hvac_installation">HVAC Installation</option>
                  <option value="plumbing_repair">Plumbing Repair</option>
                  <option value="plumbing_emergency">Plumbing Emergency</option>
                  <option value="electrical_repair">Electrical Repair</option>
                  <option value="roof_repair">Roof Repair</option>
                  <option value="general_maintenance">General Maintenance</option>
                </select>
              </div>

              <div className="vf-form-row">
                <label>Work Description</label>
                <textarea
                  value={invoiceData.work_description || ''}
                  onChange={(e) => setInvoiceData(prev => ({ ...prev, work_description: e.target.value }))}
                  placeholder="Describe the work performed..."
                  rows={3}
                />
              </div>
            </div>

            <div className="vf-form-section">
              <h3>Line Items</h3>
              
              {invoiceData.line_items.map((item, index) => (
                <div key={index} className="vf-line-item">
                  <input
                    type="text"
                    value={item.description}
                    onChange={(e) => updateLineItem(index, 'description', e.target.value)}
                    placeholder="Description"
                    className="vf-line-desc"
                  />
                  <input
                    type="number"
                    value={item.quantity}
                    onChange={(e) => updateLineItem(index, 'quantity', parseFloat(e.target.value) || 0)}
                    placeholder="Qty"
                    className="vf-line-qty"
                  />
                  <input
                    type="number"
                    value={item.unit_price}
                    onChange={(e) => updateLineItem(index, 'unit_price', parseFloat(e.target.value) || 0)}
                    placeholder="Price"
                    className="vf-line-price"
                  />
                  <span className="vf-line-total">${(item.total || 0).toFixed(2)}</span>
                  <button onClick={() => removeLineItem(index)} className="vf-line-remove">×</button>
                </div>
              ))}

              <button onClick={addLineItem} className="vf-add-line">+ Add Line Item</button>

              <div className="vf-total">
                <strong>Total: ${invoiceData.total_amount.toFixed(2)}</strong>
              </div>
            </div>

            <button
              onClick={analyzeInvoice}
              disabled={loading}
              className="vf-analyze-btn"
            >
              {loading ? <><RefreshIcon size={18} className="spin" /> Analyzing...</> : <><SearchIcon size={18} /> Analyze Invoice</>}
            </button>

            {/* Analysis Result */}
            {analysisResult && (
              <div className="vf-result">
                <div
                  className="vf-result-header"
                  style={{ borderLeftColor: RISK_COLORS[analysisResult.risk_level] }}
                >
                  <span className="vf-risk-badge" style={{ backgroundColor: RISK_COLORS[analysisResult.risk_level] }}>
                    {analysisResult.risk_level.toUpperCase()} RISK
                  </span>
                  <span className="vf-risk-score">Score: {analysisResult.risk_score}/100</span>
                  <span className="vf-confidence">Confidence: {(analysisResult.confidence_score * 100).toFixed(0)}%</span>
                </div>

                {analysisResult.fraud_detected && (
                  <div className="vf-fraud-alert">
                    <WarningIcon size={20} /> FRAUD INDICATORS DETECTED
                  </div>
                )}

                {analysisResult.fraud_indicators.length > 0 && (
                  <div className="vf-indicators">
                    <h4>Fraud Indicators</h4>
                    {analysisResult.fraud_indicators.map((indicator, i) => (
                      <div key={i} className={`vf-indicator vf-severity-${indicator.severity}`}>
                        <span className="vf-indicator-type">{indicator.type}</span>
                        <span className="vf-indicator-desc">{indicator.description}</span>
                      </div>
                    ))}
                  </div>
                )}

                <div className="vf-recommendations">
                  <h4>Recommendations</h4>
                  <ul>
                    {analysisResult.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Chat Tab */}
        {activeTab === 'chat' && (
          <div className="vf-chat-tab">
            <div className="vf-chat-messages">
              {chatMessages.length === 0 && (
                <div className="vf-chat-welcome">
                  <p><WaveIcon size={20} /> Ask me anything about vendor fraud detection!</p>
                  <p>Examples:</p>
                  <ul>
                    <li>"What are common HVAC contractor fraud patterns?"</li>
                    <li>"How do I verify an auto body shop invoice?"</li>
                    <li>"What's the average price for plumbing repairs?"</li>
                  </ul>
                </div>
              )}
              {chatMessages.map((msg, i) => (
                <div key={i} className={`vf-chat-message vf-chat-${msg.role}`}>
                  <div className="vf-chat-content">{msg.content}</div>
                  <div className="vf-chat-time">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>

            <div className="vf-chat-input">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
                placeholder="Ask about vendor fraud..."
                disabled={loading}
              />
              <button onClick={sendChatMessage} disabled={loading}>
                {loading ? '...' : <ArrowRightIcon size={18} />}
              </button>
            </div>
          </div>
        )}

        {/* Documents Tab */}
        {activeTab === 'documents' && (
          <div className="vf-documents-tab">
            <h3><BooksIcon size={24} /> Ingest Documents into Knowledge Base</h3>
            <p>Add vendor invoices, pricing guides, or fraud patterns to the RAG knowledge base.</p>

            <div className="vf-form-row">
              <label>Document Title</label>
              <input
                type="text"
                value={docTitle}
                onChange={(e) => setDocTitle(e.target.value)}
                placeholder="e.g., HVAC Pricing Guide 2026"
              />
            </div>

            <div className="vf-form-row">
              <label>Document Content</label>
              <textarea
                value={docContent}
                onChange={(e) => setDocContent(e.target.value)}
                placeholder="Paste document content here..."
                rows={10}
              />
            </div>

            <button onClick={ingestDocument} disabled={loading} className="vf-ingest-btn">
              {loading ? <><RefreshIcon size={18} className="spin" /> Ingesting...</> : <><DownloadIcon size={18} /> Ingest Document</>}
            </button>

            {ingestionResult && (
              <div className="vf-success-message"><CheckIcon size={18} /> {ingestionResult}</div>
            )}
          </div>
        )}

        {/* Benchmarks Tab */}
        {activeTab === 'benchmarks' && (
          <div className="vf-benchmarks-tab">
            <h3><ChartIcon size={24} /> Pricing Benchmarks</h3>
            <p>Check market rates for common services.</p>

            <div className="vf-form-row">
              <label>Service Type</label>
              <select value={serviceType} onChange={(e) => setServiceType(e.target.value)}>
                <option value="hvac_repair">HVAC Repair</option>
                <option value="hvac_installation">HVAC Installation</option>
                <option value="plumbing_repair">Plumbing Repair</option>
                <option value="plumbing_emergency">Plumbing Emergency</option>
                <option value="electrical_repair">Electrical Repair</option>
                <option value="electrical_panel_upgrade">Electrical Panel Upgrade</option>
                <option value="roof_repair">Roof Repair</option>
                <option value="roof_replacement">Roof Replacement</option>
                <option value="appliance_repair">Appliance Repair</option>
                <option value="general_maintenance">General Maintenance</option>
              </select>
            </div>

            <div className="vf-form-row">
              <label>Price to Check (optional)</label>
              <input
                type="number"
                value={priceToCheck}
                onChange={(e) => setPriceToCheck(parseFloat(e.target.value) || 0)}
                placeholder="Enter price to compare"
              />
            </div>

            <button onClick={checkBenchmark} disabled={loading} className="vf-benchmark-btn">
              {loading ? <><RefreshIcon size={18} className="spin" /> Checking...</> : <><ChartIcon size={18} /> Check Benchmark</>}
            </button>

            {benchmarkResult && (
              <div className="vf-benchmark-result">
                <pre>{JSON.stringify(benchmarkResult, null, 2)}</pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default VendorFraudPanel;

