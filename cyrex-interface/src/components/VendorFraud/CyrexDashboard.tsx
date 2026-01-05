/**
 * Cyrex Comprehensive Vendor Intelligence Dashboard
 * Complete B2B analytics platform for vendor intelligence
 */
import React, { useState, useCallback, useEffect } from 'react';
import { vendorFraudApi } from './api';
import type {
  Industry,
  InvoiceAnalysisResult,
  VendorProfile,
  IndustryInfo,
} from './types';
import './VendorFraudPanel.css';

interface Analytics {
  total_invoices_analyzed: number;
  fraud_detected_count: number;
  fraud_detection_rate: number;
  total_amount_analyzed: number;
  high_risk_vendors: number;
  cross_industry_flags: number;
  network_effects_active: boolean;
}

interface VendorListItem {
  vendor_id: string;
  vendor_name: string;
  industries_served: string[];
  fraud_flags_count: number;
  current_risk_score: number;
  risk_level: string;
  status: string;
  cross_industry_flags: number;
}

export const CyrexDashboard: React.FC<{ baseUrl?: string; apiKey?: string }> = ({ baseUrl, apiKey }) => {
  const [activeView, setActiveView] = useState<'dashboard' | 'vendors' | 'analytics' | 'analyze'>('dashboard');
  const [industry, setIndustry] = useState<Industry>('property_management');
  const [industries, setIndustries] = useState<IndustryInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Dashboard state
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [vendors, setVendors] = useState<VendorListItem[]>([]);
  const [selectedVendor, setSelectedVendor] = useState<VendorProfile | null>(null);
  const [vendorSearch, setVendorSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState<string>('all');

  // Configure API
  useEffect(() => {
    if (baseUrl) vendorFraudApi.setBaseUrl(baseUrl);
    if (apiKey) vendorFraudApi.setApiKey(apiKey);
  }, [baseUrl, apiKey]);

  // Load industries
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

  // Load analytics
  const loadAnalytics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await vendorFraudApi.client.get('/vendor-fraud/analytics', {
        params: { industry: industry !== 'all' ? industry : undefined }
      });
      setAnalytics(result.data.analytics);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load analytics');
    } finally {
      setLoading(false);
    }
  }, [industry]);

  // Load vendors
  const loadVendors = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await vendorFraudApi.client.get('/vendor-fraud/vendors', {
        params: {
          query: vendorSearch || undefined,
          industry: industry !== 'all' ? industry : undefined,
          risk_level: riskFilter !== 'all' ? riskFilter : undefined,
          limit: 100
        }
      });
      setVendors(result.data.vendors);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load vendors');
    } finally {
      setLoading(false);
    }
  }, [industry, vendorSearch, riskFilter]);

  // Load vendor details
  const loadVendorDetails = useCallback(async (vendorId: string) => {
    setLoading(true);
    try {
      const result = await vendorFraudApi.client.get(`/vendor-fraud/vendors/${vendorId}`);
      setSelectedVendor(result.data.vendor);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load vendor details');
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-load on view change
  useEffect(() => {
    if (activeView === 'dashboard' || activeView === 'analytics') {
      loadAnalytics();
    }
    if (activeView === 'vendors') {
      loadVendors();
    }
  }, [activeView, industry, loadAnalytics, loadVendors]);

  const RISK_COLORS: Record<string, string> = {
    low: '#4CAF50',
    medium: '#FF9800',
    high: '#f44336',
    critical: '#9C27B0',
  };

  const INDUSTRY_ICONS: Record<Industry, string> = {
    property_management: '🏢',
    corporate_procurement: '📦',
    insurance_pc: '🛡️',
    general_contractors: '🏗️',
    retail_ecommerce: '🛒',
    law_firms: '⚖️',
  };

  return (
    <div className="cyrex-dashboard">
      {/* Header */}
      <div className="cyrex-header">
        <div className="cyrex-header-content">
          <h1>🔍 Cyrex Vendor Intelligence Platform</h1>
          <p>Complete B2B Analytics & Risk Management Across Six Industries</p>
        </div>
        <div className="cyrex-industry-selector">
          <label>Industry:</label>
          <select value={industry} onChange={(e) => setIndustry(e.target.value as Industry)}>
            <option value="all">All Industries</option>
            {industries.map((ind) => (
              <option key={ind.id} value={ind.id}>
                {INDUSTRY_ICONS[ind.id as Industry]} {ind.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Navigation */}
      <div className="cyrex-nav">
        <button
          className={activeView === 'dashboard' ? 'active' : ''}
          onClick={() => setActiveView('dashboard')}
        >
          📊 Dashboard
        </button>
        <button
          className={activeView === 'vendors' ? 'active' : ''}
          onClick={() => setActiveView('vendors')}
        >
          🏢 Vendor Intelligence
        </button>
        <button
          className={activeView === 'analytics' ? 'active' : ''}
          onClick={() => setActiveView('analytics')}
        >
          📈 Analytics
        </button>
        <button
          className={activeView === 'analyze' ? 'active' : ''}
          onClick={() => setActiveView('analyze')}
        >
          🔍 Analyze Invoice
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="cyrex-error">
          ⚠️ {error}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* Content */}
      <div className="cyrex-content">
        {/* Dashboard View */}
        {activeView === 'dashboard' && analytics && (
          <div className="cyrex-dashboard-view">
            <div className="cyrex-stats-grid">
              <div className="cyrex-stat-card">
                <div className="cyrex-stat-icon">📄</div>
                <div className="cyrex-stat-content">
                  <div className="cyrex-stat-value">{analytics.total_invoices_analyzed.toLocaleString()}</div>
                  <div className="cyrex-stat-label">Invoices Analyzed</div>
                </div>
              </div>

              <div className="cyrex-stat-card">
                <div className="cyrex-stat-icon">⚠️</div>
                <div className="cyrex-stat-content">
                  <div className="cyrex-stat-value">{analytics.fraud_detected_count.toLocaleString()}</div>
                  <div className="cyrex-stat-label">Fraud Detected</div>
                  <div className="cyrex-stat-sublabel">
                    {analytics.fraud_detection_rate.toFixed(1)}% detection rate
                  </div>
                </div>
              </div>

              <div className="cyrex-stat-card">
                <div className="cyrex-stat-icon">💰</div>
                <div className="cyrex-stat-content">
                  <div className="cyrex-stat-value">${(analytics.total_amount_analyzed / 1000000).toFixed(1)}M</div>
                  <div className="cyrex-stat-label">Total Amount Analyzed</div>
                </div>
              </div>

              <div className="cyrex-stat-card">
                <div className="cyrex-stat-icon">🔴</div>
                <div className="cyrex-stat-content">
                  <div className="cyrex-stat-value">{analytics.high_risk_vendors}</div>
                  <div className="cyrex-stat-label">High-Risk Vendors</div>
                </div>
              </div>

              <div className="cyrex-stat-card cyrex-network-card">
                <div className="cyrex-stat-icon">🌐</div>
                <div className="cyrex-stat-content">
                  <div className="cyrex-stat-value">{analytics.cross_industry_flags}</div>
                  <div className="cyrex-stat-label">Cross-Industry Flags</div>
                  <div className="cyrex-stat-sublabel">
                    {analytics.network_effects_active ? '✅ Network Effects Active' : 'Network effects building...'}
                  </div>
                </div>
              </div>
            </div>

            <div className="cyrex-quick-actions">
              <h3>Quick Actions</h3>
              <div className="cyrex-action-buttons">
                <button onClick={() => setActiveView('analyze')} className="cyrex-action-btn">
                  🔍 Analyze New Invoice
                </button>
                <button onClick={() => setActiveView('vendors')} className="cyrex-action-btn">
                  🏢 View Vendor Database
                </button>
                <button onClick={loadAnalytics} className="cyrex-action-btn" disabled={loading}>
                  🔄 Refresh Analytics
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Vendor Intelligence View */}
        {activeView === 'vendors' && (
          <div className="cyrex-vendors-view">
            <div className="cyrex-vendors-header">
              <h2>Vendor Intelligence Database</h2>
              <div className="cyrex-vendors-filters">
                <input
                  type="text"
                  placeholder="Search vendors..."
                  value={vendorSearch}
                  onChange={(e) => setVendorSearch(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && loadVendors()}
                />
                <select value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)}>
                  <option value="all">All Risk Levels</option>
                  <option value="low">Low Risk</option>
                  <option value="medium">Medium Risk</option>
                  <option value="high">High Risk</option>
                  <option value="critical">Critical Risk</option>
                </select>
                <button onClick={loadVendors} disabled={loading}>
                  🔍 Search
                </button>
              </div>
            </div>

            <div className="cyrex-vendors-list">
              {vendors.map((vendor) => (
                <div
                  key={vendor.vendor_id}
                  className="cyrex-vendor-card"
                  onClick={() => loadVendorDetails(vendor.vendor_id)}
                >
                  <div className="cyrex-vendor-header">
                    <h3>{vendor.vendor_name}</h3>
                    <span
                      className="cyrex-risk-badge"
                      style={{ backgroundColor: RISK_COLORS[vendor.risk_level] || '#666' }}
                    >
                      {vendor.risk_level.toUpperCase()}
                    </span>
                  </div>
                  <div className="cyrex-vendor-stats">
                    <div className="cyrex-vendor-stat">
                      <span className="cyrex-stat-label">Risk Score:</span>
                      <span className="cyrex-stat-value">{vendor.current_risk_score.toFixed(0)}/100</span>
                    </div>
                    <div className="cyrex-vendor-stat">
                      <span className="cyrex-stat-label">Fraud Flags:</span>
                      <span className="cyrex-stat-value">{vendor.fraud_flags_count}</span>
                    </div>
                    <div className="cyrex-vendor-stat">
                      <span className="cyrex-stat-label">Industries:</span>
                      <span className="cyrex-stat-value">{vendor.industries_served.length}</span>
                    </div>
                    {vendor.cross_industry_flags > 0 && (
                      <div className="cyrex-vendor-stat cyrex-network-badge">
                        <span className="cyrex-stat-label">🌐 Cross-Industry:</span>
                        <span className="cyrex-stat-value">{vendor.cross_industry_flags} flags</span>
                      </div>
                    )}
                  </div>
                  <div className="cyrex-vendor-industries">
                    {vendor.industries_served.map((ind) => (
                      <span key={ind} className="cyrex-industry-tag">
                        {INDUSTRY_ICONS[ind as Industry] || '📋'} {ind.replace('_', ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {selectedVendor && (
              <div className="cyrex-vendor-details-modal">
                <div className="cyrex-modal-content">
                  <button className="cyrex-modal-close" onClick={() => setSelectedVendor(null)}>×</button>
                  <h2>{selectedVendor.vendor_name}</h2>
                  <div className="cyrex-vendor-details">
                    <div className="cyrex-detail-section">
                      <h4>Risk Assessment</h4>
                      <div className="cyrex-detail-item">
                        <span>Risk Score:</span>
                        <span style={{ color: RISK_COLORS[selectedVendor.risk_level] }}>
                          {selectedVendor.current_risk_score.toFixed(0)}/100 ({selectedVendor.risk_level})
                        </span>
                      </div>
                      <div className="cyrex-detail-item">
                        <span>Status:</span>
                        <span>{selectedVendor.status}</span>
                      </div>
                    </div>
                    <div className="cyrex-detail-section">
                      <h4>Performance</h4>
                      <div className="cyrex-detail-item">
                        <span>Invoices Analyzed:</span>
                        <span>{selectedVendor.total_invoices_analyzed}</span>
                      </div>
                      <div className="cyrex-detail-item">
                        <span>Average Invoice:</span>
                        <span>${selectedVendor.average_invoice_amount.toFixed(2)}</span>
                      </div>
                      <div className="cyrex-detail-item">
                        <span>Price Deviation:</span>
                        <span>{selectedVendor.average_price_deviation.toFixed(1)}%</span>
                      </div>
                    </div>
                    <div className="cyrex-detail-section">
                      <h4>Cross-Industry Intelligence</h4>
                      <div className="cyrex-detail-item">
                        <span>Industries Served:</span>
                        <span>{selectedVendor.industries_served.join(', ')}</span>
                      </div>
                      <div className="cyrex-detail-item">
                        <span>Cross-Industry Flags:</span>
                        <span>{selectedVendor.cross_industry_flags}</span>
                      </div>
                      {selectedVendor.flagged_by_industries.length > 0 && (
                        <div className="cyrex-detail-item">
                          <span>Flagged By Industries:</span>
                          <span>{selectedVendor.flagged_by_industries.join(', ')}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Analytics View */}
        {activeView === 'analytics' && analytics && (
          <div className="cyrex-analytics-view">
            <h2>Comprehensive Analytics</h2>
            <div className="cyrex-analytics-grid">
              <div className="cyrex-analytics-card">
                <h3>Fraud Detection Performance</h3>
                <div className="cyrex-metric">
                  <div className="cyrex-metric-value">{analytics.fraud_detection_rate.toFixed(1)}%</div>
                  <div className="cyrex-metric-label">Detection Rate</div>
                  <div className="cyrex-metric-detail">
                    {analytics.fraud_detected_count} of {analytics.total_invoices_analyzed} invoices flagged
                  </div>
                </div>
              </div>

              <div className="cyrex-analytics-card">
                <h3>Network Effects</h3>
                <div className="cyrex-metric">
                  <div className="cyrex-metric-value">{analytics.cross_industry_flags}</div>
                  <div className="cyrex-metric-label">Cross-Industry Flags</div>
                  <div className="cyrex-metric-detail">
                    {analytics.network_effects_active
                      ? '✅ Vendors flagged in one industry help all industries'
                      : 'Building network effects...'}
                  </div>
                </div>
              </div>

              <div className="cyrex-analytics-card">
                <h3>Risk Distribution</h3>
                <div className="cyrex-metric">
                  <div className="cyrex-metric-value">{analytics.high_risk_vendors}</div>
                  <div className="cyrex-metric-label">High-Risk Vendors</div>
                  <div className="cyrex-metric-detail">Requiring immediate attention</div>
                </div>
              </div>

              <div className="cyrex-analytics-card">
                <h3>Total Value Protected</h3>
                <div className="cyrex-metric">
                  <div className="cyrex-metric-value">${(analytics.total_amount_analyzed / 1000000).toFixed(2)}M</div>
                  <div className="cyrex-metric-label">Amount Analyzed</div>
                  <div className="cyrex-metric-detail">Across all industries</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Analyze Invoice View - Use existing VendorFraudPanel component */}
        {activeView === 'analyze' && (
          <div className="cyrex-analyze-view">
            <p>Invoice analysis interface would go here - using VendorFraudPanel component</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default CyrexDashboard;


