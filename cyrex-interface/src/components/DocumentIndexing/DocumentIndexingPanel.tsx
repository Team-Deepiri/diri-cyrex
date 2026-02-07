import React, { useState, useEffect } from 'react';

interface Props {
  baseUrl?: string;
  apiKey?: string;
}

export const DocumentIndexingPanel: React.FC<Props> = ({ baseUrl, apiKey }) => {
  const [docType, setDocType] = useState('legal_document');
  const [metadata, setMetadata] = useState('{}');
  
  // File upload state
  const [file, setFile] = useState<File | null>(null);
  const [documentId, setDocumentId] = useState('');
  const [title, setTitle] = useState('');
  const [industry, setIndustry] = useState('legal');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Tab state
  const [activeTab, setActiveTab] = useState<'index' | 'search' | 'browse' | 'stats'>('index');

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchDocTypes, setSearchDocTypes] = useState<string[]>([]);
  const [searchIndustry, setSearchIndustry] = useState('');
  const [searchTopK, setSearchTopK] = useState(5);
  const [searchMetadataFilters, setSearchMetadataFilters] = useState('{}');
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Browse state
  const [browseDocType, setBrowseDocType] = useState('');
  const [browseIndustry, setBrowseIndustry] = useState('');
  const [browseLoading, setBrowseLoading] = useState(false);
  const [documentList, setDocumentList] = useState<any[]>([]);
  const [browseError, setBrowseError] = useState<string | null>(null);
  const [selectedDocumentId, setSelectedDocumentId] = useState('');
  const [documentDetails, setDocumentDetails] = useState<any>(null);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Stats state
  const [stats, setStats] = useState<any>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState<string | null>(null);

  // Common fields (all document types)
  const [version, setVersion] = useState('');
  const [versionDate, setVersionDate] = useState('');
  const [previousVersionId, setPreviousVersionId] = useState('');

  // Lease fields
  const [leaseId, setLeaseId] = useState('');
  const [tenantName, setTenantName] = useState('');
  const [landlordName, setLandlordName] = useState('');
  const [propertyAddress, setPropertyAddress] = useState('');
  const [leaseStartDate, setLeaseStartDate] = useState('');
  const [leaseEndDate, setLeaseEndDate] = useState('');
  const [regulatoryReferences, setRegulatoryReferences] = useState('');
  const [contractReferences, setContractReferences] = useState('');
  const [leaseObligations, setLeaseObligations] = useState('');
  const [leaseClauses, setLeaseClauses] = useState('');

  // Contract fields
  const [contractId, setContractId] = useState('');
  const [contractName, setContractName] = useState('');
  const [partyA, setPartyA] = useState('');
  const [partyB, setPartyB] = useState('');
  const [effectiveDate, setEffectiveDate] = useState('');
  const [expirationDate, setExpirationDate] = useState('');
  const [contractRegulatoryRefs, setContractRegulatoryRefs] = useState('');
  const [leaseReferences, setLeaseReferences] = useState('');
  const [contractObligations, setContractObligations] = useState('');
  const [contractClauses, setContractClauses] = useState('');

  // Regulation fields
  const [regulationId, setRegulationId] = useState('');
  const [regulationName, setRegulationName] = useState('');
  const [jurisdiction, setJurisdiction] = useState('');
  const [regulationEffectiveDate, setRegulationEffectiveDate] = useState('');
  const [impactedContracts, setImpactedContracts] = useState('');
  const [impactedLeases, setImpactedLeases] = useState('');
  const [languageChanges, setLanguageChanges] = useState('');

  // Amendment fields
  const [amendmentId, setAmendmentId] = useState('');
  const [amendsDocumentId, setAmendsDocumentId] = useState('');
  const [amendsDocumentType, setAmendsDocumentType] = useState('lease');
  const [amendmentDate, setAmendmentDate] = useState('');
  const [amendmentChanges, setAmendmentChanges] = useState('');

  // Compliance Report fields
  const [reportId, setReportId] = useState('');
  const [reportDate, setReportDate] = useState('');
  const [reportType, setReportType] = useState('quarterly');
  const [complianceStatus, setComplianceStatus] = useState('compliant');
  const [violations, setViolations] = useState('');
  const [patterns, setPatterns] = useState('');

  // Template fields
  const [templateId, setTemplateId] = useState('');

  // Reset form fields when document type changes
  useEffect(() => {
    // Reset all fields when docType changes
    setVersion('');
    setVersionDate('');
    setPreviousVersionId('');
    setLeaseId('');
    setTenantName('');
    setLandlordName('');
    setPropertyAddress('');
    setLeaseStartDate('');
    setLeaseEndDate('');
    setRegulatoryReferences('');
    setContractReferences('');
    setLeaseObligations('');
    setLeaseClauses('');
    setContractId('');
    setContractName('');
    setPartyA('');
    setPartyB('');
    setEffectiveDate('');
    setExpirationDate('');
    setContractRegulatoryRefs('');
    setLeaseReferences('');
    setContractObligations('');
    setContractClauses('');
    setRegulationId('');
    setRegulationName('');
    setJurisdiction('');
    setRegulationEffectiveDate('');
    setImpactedContracts('');
    setImpactedLeases('');
    setLanguageChanges('');
    setAmendmentId('');
    setAmendsDocumentId('');
    setAmendsDocumentType('lease');
    setAmendmentDate('');
    setAmendmentChanges('');
    setReportId('');
    setReportDate('');
    setReportType('quarterly');
    setComplianceStatus('compliant');
    setViolations('');
    setPatterns('');
    setTemplateId('');
  }, [docType]);

  // Build metadata JSON from form fields
  useEffect(() => {
    const buildMetadata = () => {
      const baseMetadata: any = {
        document_type: docType,
      };

      // Add common fields
      if (version) baseMetadata.version = version;
      if (versionDate) baseMetadata.version_date = versionDate;
      if (previousVersionId) baseMetadata.previous_version_id = previousVersionId;

      // Add document-type-specific fields
      if (docType === 'lease') {
        if (leaseId) baseMetadata.lease_id = leaseId;
        if (tenantName) baseMetadata.tenant_name = tenantName;
        if (landlordName) baseMetadata.landlord_name = landlordName;
        if (propertyAddress) baseMetadata.property_address = propertyAddress;
        if (leaseStartDate) baseMetadata.lease_start_date = leaseStartDate;
        if (leaseEndDate) baseMetadata.lease_end_date = leaseEndDate;
        if (regulatoryReferences) {
          baseMetadata.regulatory_references = regulatoryReferences.split(',').map((s: string) => s.trim()).filter(Boolean);
        }
        if (contractReferences) {
          baseMetadata.contract_references = contractReferences.split(',').map((s: string) => s.trim()).filter(Boolean);
        }
        if (leaseObligations) {
          try {
            baseMetadata.obligations = JSON.parse(leaseObligations);
          } catch (e) {
            // Invalid JSON, skip
          }
        }
        if (leaseClauses) {
          try {
            baseMetadata.clauses = JSON.parse(leaseClauses);
          } catch (e) {
            // Invalid JSON, skip
          }
        }
      } else if (docType === 'contract') {
        if (contractId) baseMetadata.contract_id = contractId;
        if (contractName) baseMetadata.contract_name = contractName;
        if (partyA) baseMetadata.party_a = partyA;
        if (partyB) baseMetadata.party_b = partyB;
        if (effectiveDate) baseMetadata.effective_date = effectiveDate;
        if (expirationDate) baseMetadata.expiration_date = expirationDate;
        if (contractRegulatoryRefs) {
          baseMetadata.regulatory_references = contractRegulatoryRefs.split(',').map((s: string) => s.trim()).filter(Boolean);
        }
        if (leaseReferences) {
          baseMetadata.lease_references = leaseReferences.split(',').map((s: string) => s.trim()).filter(Boolean);
        }
        if (contractObligations) {
          try {
            baseMetadata.obligations = JSON.parse(contractObligations);
          } catch (e) {
            // Invalid JSON, skip
          }
        }
        if (contractClauses) {
          try {
            baseMetadata.clauses = JSON.parse(contractClauses);
          } catch (e) {
            // Invalid JSON, skip
          }
        }
      } else if (docType === 'regulation') {
        if (regulationId) baseMetadata.regulation_id = regulationId;
        if (regulationName) baseMetadata.regulation_name = regulationName;
        if (jurisdiction) baseMetadata.jurisdiction = jurisdiction;
        if (regulationEffectiveDate) baseMetadata.effective_date = regulationEffectiveDate;
        if (impactedContracts) {
          baseMetadata.impacted_contracts = impactedContracts.split(',').map((s: string) => s.trim()).filter(Boolean);
        }
        if (impactedLeases) {
          baseMetadata.impacted_leases = impactedLeases.split(',').map((s: string) => s.trim()).filter(Boolean);
        }
        if (languageChanges) {
          try {
            baseMetadata.language_changes = JSON.parse(languageChanges);
          } catch (e) {
            // Invalid JSON, skip
          }
        }
      } else if (docType === 'amendment') {
        if (amendmentId) baseMetadata.amendment_id = amendmentId;
        if (amendsDocumentId) baseMetadata.amends_document_id = amendsDocumentId;
        if (amendsDocumentType) baseMetadata.amends_document_type = amendsDocumentType;
        if (amendmentDate) baseMetadata.amendment_date = amendmentDate;
        if (amendmentChanges) {
          try {
            baseMetadata.changes = JSON.parse(amendmentChanges);
          } catch (e) {
            // Invalid JSON, skip
          }
        }
      } else if (docType === 'compliance_report') {
        if (reportId) baseMetadata.report_id = reportId;
        if (reportDate) baseMetadata.report_date = reportDate;
        if (reportType) baseMetadata.report_type = reportType;
        if (complianceStatus) baseMetadata.compliance_status = complianceStatus;
        if (violations) {
          try {
            baseMetadata.violations = JSON.parse(violations);
          } catch (e) {
            // Invalid JSON, skip
          }
        }
        if (patterns) {
          try {
            baseMetadata.patterns = JSON.parse(patterns);
          } catch (e) {
            // Invalid JSON, skip
          }
        }
      } else if (docType === 'template') {
        baseMetadata.is_template = true;
        if (templateId) baseMetadata.template_id = templateId;
      }

      setMetadata(JSON.stringify(baseMetadata, null, 2));
    };

    buildMetadata();
  }, [
    docType, version, versionDate, previousVersionId,
    leaseId, tenantName, landlordName, propertyAddress, leaseStartDate, leaseEndDate, regulatoryReferences, contractReferences, leaseObligations, leaseClauses,
    contractId, contractName, partyA, partyB, effectiveDate, expirationDate, contractRegulatoryRefs, leaseReferences, contractObligations, contractClauses,
    regulationId, regulationName, jurisdiction, regulationEffectiveDate, impactedContracts, impactedLeases, languageChanges,
    amendmentId, amendsDocumentId, amendsDocumentType, amendmentDate, amendmentChanges,
    reportId, reportDate, reportType, complianceStatus, violations, patterns,
    templateId
  ]);

  const handleFileUpload = async () => {
    if (!file) {
      setError('Please select a file to upload');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
    const formData = new FormData();
    formData.append('file', file);
      
      // Optional parameters
      if (documentId) formData.append('document_id', documentId);
      if (title) formData.append('title', title);
      formData.append('doc_type', docType);
      formData.append('industry', industry);

      // Parse and append metadata JSON
    try {
        const metadataObj = JSON.parse(metadata);
        if (Object.keys(metadataObj).length > 0) {
          formData.append('metadata', JSON.stringify(metadataObj));
        }
      } catch (e) {
        // Invalid JSON, skip metadata
        console.warn('Invalid metadata JSON, skipping');
      }

      const apiUrl = `${baseUrl || 'http://localhost:8001'}/api/v1/documents/index/file`;
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
        headers: apiKey ? { 'x-api-key': apiKey } : {}
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error: any) {
      // Provide more helpful error messages
      let errorMessage = error.message || 'Failed to index file';
      
      if (error.message === 'Failed to fetch' || error.name === 'TypeError') {
        errorMessage = `Cannot connect to backend server. Please ensure:
- Backend server is running on ${baseUrl || 'http://localhost:8001'}
- CORS is properly configured
- Network connectivity is available`;
      }
      
      setError(errorMessage);
      console.error('Error:', error);
      console.error('API URL attempted:', `${baseUrl || 'http://localhost:8001'}/api/v1/documents/index/file`);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchError('Please enter a search query');
      return;
    }
    
    setSearchLoading(true);
    setSearchError(null);
    setSearchResults([]);

    try {
      // Parse metadata filters if provided
      let metadataFiltersObj = {};
      if (searchMetadataFilters && searchMetadataFilters.trim() !== '{}') {
        try {
          metadataFiltersObj = JSON.parse(searchMetadataFilters);
        } catch (e) {
          setSearchError('Invalid metadata filters JSON');
          return;
        }
      }

      const apiUrl = `${baseUrl || 'http://localhost:8001'}/api/v1/documents/search`;
      
      const requestBody = {
        query: searchQuery,
        top_k: searchTopK,
        ...(searchDocTypes.length > 0 && { doc_types: searchDocTypes }),
        ...(searchIndustry && { industry: searchIndustry }),
        ...(Object.keys(metadataFiltersObj).length > 0 && { metadata_filters: metadataFiltersObj })
      };

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { 'x-api-key': apiKey } : {})
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setSearchResults(data.results || []);
    } catch (error: any) {
      let errorMessage = error.message || 'Failed to search documents';
      
      if (error.message === 'Failed to fetch' || error.name === 'TypeError') {
        errorMessage = `Cannot connect to backend server. Please ensure:
- Backend server is running on ${baseUrl || 'http://localhost:8001'}
- CORS is properly configured
- Network connectivity is available`;
      }
      
      setSearchError(errorMessage);
      console.error('Search error:', error);
    } finally {
      setSearchLoading(false);
    }
  };

  const handleBrowseDocuments = async () => {
    setBrowseLoading(true);
    setBrowseError(null);
    setDocumentList([]);

    try {
      const params = new URLSearchParams();
      if (browseDocType) params.append('doc_type', browseDocType);
      if (browseIndustry) params.append('industry', browseIndustry);

      const apiUrl = `${baseUrl || 'http://localhost:8001'}/api/v1/documents/list${params.toString() ? '?' + params.toString() : ''}`;
      
      const response = await fetch(apiUrl, {
        method: 'GET',
        headers: {
          ...(apiKey ? { 'x-api-key': apiKey } : {})
        }
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setDocumentList(data.documents || []);
    } catch (error: any) {
      setBrowseError(error.message || 'Failed to list documents');
      console.error('Browse error:', error);
    } finally {
      setBrowseLoading(false);
    }
  };

  const handleGetDocumentDetails = async (docId: string) => {
    setDetailsLoading(true);
    setDocumentDetails(null);

    try {
      const apiUrl = `${baseUrl || 'http://localhost:8001'}/api/v1/documents/${docId}`;
      
      const response = await fetch(apiUrl, {
        method: 'GET',
        headers: {
          ...(apiKey ? { 'x-api-key': apiKey } : {})
        }
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setDocumentDetails(data.document);
    } catch (error: any) {
      setBrowseError(error.message || 'Failed to get document details');
      console.error('Get details error:', error);
    } finally {
      setDetailsLoading(false);
    }
  };

  const handleDeleteDocument = async (docId: string) => {
    if (!confirm(`Are you sure you want to delete document "${docId}"? This will remove all chunks and cannot be undone.`)) {
      return;
    }

    setDeletingDocId(docId);
    setDeleteError(null);

    try {
      const apiUrl = `${baseUrl || 'http://localhost:8001'}/api/v1/documents/delete`;
      
      const response = await fetch(apiUrl, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { 'x-api-key': apiKey } : {})
        },
        body: JSON.stringify({ document_id: docId })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        // Remove from document list
        setDocumentList(documentList.filter(doc => doc.document_id !== docId));
        
        // Clear details if this document was selected
        if (selectedDocumentId === docId) {
          setSelectedDocumentId('');
          setDocumentDetails(null);
        }
        
        alert(`Document "${docId}" deleted successfully`);
      } else {
        throw new Error('Delete operation returned unsuccessful');
      }
    } catch (error: any) {
      setDeleteError(error.message || 'Failed to delete document');
      console.error('Delete error:', error);
      alert(`Failed to delete document: ${error.message}`);
    } finally {
      setDeletingDocId(null);
    }
  };

  const handleGetStatistics = async () => {
    setStatsLoading(true);
    setStatsError(null);
    setStats(null);

    try {
      const apiUrl = `${baseUrl || 'http://localhost:8001'}/api/v1/documents/stats`;
      
      const response = await fetch(apiUrl, {
        method: 'GET',
        headers: {
          ...(apiKey ? { 'x-api-key': apiKey } : {})
        }
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setStats(data.statistics);
    } catch (error: any) {
      setStatsError(error.message || 'Failed to get statistics');
      console.error('Stats error:', error);
    } finally {
      setStatsLoading(false);
    }
  };

  // Load stats when stats tab is opened
  useEffect(() => {
    if (activeTab === 'stats' && !stats && !statsLoading) {
      handleGetStatistics();
    }
  }, [activeTab]);

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px' }}>
      <h2 style={{ color: '#e0e0e0', marginBottom: '1.5rem' }}>Document Indexing & Search</h2>
      
      {/* Tab Navigation */}
      <div style={{ 
        display: 'flex', 
        gap: '0.5rem', 
        marginBottom: '1.5rem',
        borderBottom: '2px solid #444'
      }}>
        <button
          onClick={() => setActiveTab('index')}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '0.95rem',
            background: activeTab === 'index' ? '#4a9eff' : 'transparent',
            color: activeTab === 'index' ? '#fff' : '#b0b0b0',
            border: 'none',
            borderBottom: activeTab === 'index' ? '2px solid #4a9eff' : '2px solid transparent',
            borderRadius: '4px 4px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'index' ? 'bold' : 'normal',
            transition: 'all 0.2s'
          }}
        >
          📄 Index
      </button>
        <button
          onClick={() => setActiveTab('search')}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '0.95rem',
            background: activeTab === 'search' ? '#4a9eff' : 'transparent',
            color: activeTab === 'search' ? '#fff' : '#b0b0b0',
            border: 'none',
            borderBottom: activeTab === 'search' ? '2px solid #4a9eff' : '2px solid transparent',
            borderRadius: '4px 4px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'search' ? 'bold' : 'normal',
            transition: 'all 0.2s'
          }}
        >
          🔍 Search
        </button>
        <button
          onClick={() => setActiveTab('browse')}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '0.95rem',
            background: activeTab === 'browse' ? '#4a9eff' : 'transparent',
            color: activeTab === 'browse' ? '#fff' : '#b0b0b0',
            border: 'none',
            borderBottom: activeTab === 'browse' ? '2px solid #4a9eff' : '2px solid transparent',
            borderRadius: '4px 4px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'browse' ? 'bold' : 'normal',
            transition: 'all 0.2s'
          }}
        >
          📚 Browse
        </button>
        <button
          onClick={() => setActiveTab('stats')}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '0.95rem',
            background: activeTab === 'stats' ? '#4a9eff' : 'transparent',
            color: activeTab === 'stats' ? '#fff' : '#b0b0b0',
            border: 'none',
            borderBottom: activeTab === 'stats' ? '2px solid #4a9eff' : '2px solid transparent',
            borderRadius: '4px 4px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'stats' ? 'bold' : 'normal',
            transition: 'all 0.2s'
          }}
        >
          📊 Statistics
        </button>
      </div>

      {/* Index Tab Content */}
      {activeTab === 'index' && (
        <div>
          {/* File Upload */}
      <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
        <h3 style={{ color: '#e0e0e0', marginBottom: '1rem' }}>File Upload</h3>
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
            File *
          </label>
          <input
            type="file"
            onChange={(e) => {
              const selectedFile = e.target.files?.[0] || null;
              setFile(selectedFile);
              if (selectedFile && !title) {
                setTitle(selectedFile.name.replace(/\.[^/.]+$/, ''));
              }
            }}
            accept=".pdf,.docx,.txt,.md,.csv"
            style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
          />
          {file && (
            <p style={{ color: '#b0b0b0', fontSize: '0.85rem', marginTop: '0.5rem' }}>
              Selected: {file.name} ({(file.size / 1024).toFixed(2)} KB)
            </p>
          )}
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
          <div>
            <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
              Document ID (optional)
            </label>
            <input
              type="text"
              value={documentId}
              onChange={(e) => setDocumentId(e.target.value)}
              placeholder="Auto-generated if empty"
              style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
            />
          </div>
          <div>
            <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
              Title (optional)
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder={file?.name || 'Document title'}
              style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
            />
          </div>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div>
            <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
              Document Type
            </label>
            <select 
              value={docType} 
              onChange={(e) => setDocType(e.target.value)}
              style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
            >
              <option value="legal_document">Legal Document</option>
              <option value="lease">Lease</option>
              <option value="contract">Contract</option>
              <option value="regulation">Regulation</option>
              <option value="amendment">Amendment</option>
              <option value="compliance_report">Compliance Report</option>
              <option value="template">Template</option>
            </select>
          </div>
          <div>
            <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
              Industry
            </label>
            <input
              type="text"
              value={industry}
              onChange={(e) => setIndustry(e.target.value)}
              placeholder="legal"
              style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
            />
          </div>
        </div>
      </div>

      {/* Common Fields (All Document Types) */}
      <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
        <h3 style={{ color: '#e0e0e0', marginBottom: '1rem' }}>Common Fields</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
          <div>
            <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Version</label>
            <input
              type="text"
              value={version}
              onChange={(e) => setVersion(e.target.value)}
              placeholder="1.0"
              style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
            />
          </div>
          <div>
            <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Version Date</label>
            <input
              type="date"
              value={versionDate}
              onChange={(e) => setVersionDate(e.target.value)}
              style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
            />
          </div>
        </div>
        <div>
          <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Previous Version ID (optional)</label>
          <input
            type="text"
            value={previousVersionId}
            onChange={(e) => setPreviousVersionId(e.target.value)}
            placeholder="Leave empty for first version"
            style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
          />
        </div>
      </div>

      {/* Document Type Specific Fields */}
      <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
        <h3 style={{ color: '#e0e0e0', marginBottom: '1rem' }}>
          {docType === 'lease' && 'Lease Fields'}
          {docType === 'contract' && 'Contract Fields'}
          {docType === 'regulation' && 'Regulation Fields'}
          {docType === 'amendment' && 'Amendment Fields'}
          {docType === 'compliance_report' && 'Compliance Report Fields'}
          {docType === 'template' && 'Template Fields'}
          {docType === 'legal_document' && 'Legal Document Fields'}
        </h3>

        {/* LEASE Fields */}
        {docType === 'lease' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Lease ID *</label>
                <input
                  type="text"
                  value={leaseId}
                  onChange={(e) => setLeaseId(e.target.value)}
                  placeholder="LEASE-001"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Property Address *</label>
                <input
                  type="text"
                  value={propertyAddress}
                  onChange={(e) => setPropertyAddress(e.target.value)}
                  placeholder="123 Main St"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Tenant Name *</label>
                <input
                  type="text"
                  value={tenantName}
                  onChange={(e) => setTenantName(e.target.value)}
                  placeholder="Acme Corp"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Landlord Name *</label>
                <input
                  type="text"
                  value={landlordName}
                  onChange={(e) => setLandlordName(e.target.value)}
                  placeholder="Property LLC"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Lease Start Date *</label>
                <input
                  type="date"
                  value={leaseStartDate}
                  onChange={(e) => setLeaseStartDate(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Lease End Date *</label>
                <input
                  type="date"
                  value={leaseEndDate}
                  onChange={(e) => setLeaseEndDate(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Regulatory References (comma-separated)</label>
                <input
                  type="text"
                  value={regulatoryReferences}
                  onChange={(e) => setRegulatoryReferences(e.target.value)}
                  placeholder="REG-2024-001, REG-2024-002"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Contract References (comma-separated)</label>
                <input
                  type="text"
                  value={contractReferences}
                  onChange={(e) => setContractReferences(e.target.value)}
                  placeholder="CONTRACT-001, CONTRACT-002"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                  Obligations (JSON array, optional)
                  <span style={{ color: '#4a9eff', fontSize: '0.75rem', marginLeft: '0.5rem' }}>
                    ⚡ Auto-extracted if empty
                  </span>
                </label>
                <textarea
                  value={leaseObligations}
                  onChange={(e) => setLeaseObligations(e.target.value)}
                  placeholder='[{"obligation_id": "OBL-001", "type": "rent_payment", "deadline": "monthly", "owner": "tenant", "amount": 5000.00}]\n\nLeave empty to auto-extract from document'
                  rows={4}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                  Clauses (JSON array, optional)
                  <span style={{ color: '#4a9eff', fontSize: '0.75rem', marginLeft: '0.5rem' }}>
                    ⚡ Auto-extracted if empty
                  </span>
                </label>
                <textarea
                  value={leaseClauses}
                  onChange={(e) => setLeaseClauses(e.target.value)}
                  placeholder='[{"clause_id": "CLAUSE-001", "type": "rent", "section": "Section 3.1", "text": "Tenant shall pay monthly rent..."}]\n\nLeave empty to auto-extract from document'
                  rows={4}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem' }}
                />
              </div>
            </div>
          </div>
        )}

        {/* CONTRACT Fields */}
        {docType === 'contract' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Contract ID *</label>
                <input
                  type="text"
                  value={contractId}
                  onChange={(e) => setContractId(e.target.value)}
                  placeholder="CONTRACT-001"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Contract Name *</label>
                <input
                  type="text"
                  value={contractName}
                  onChange={(e) => setContractName(e.target.value)}
                  placeholder="Service Agreement"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Party A *</label>
                <input
                  type="text"
                  value={partyA}
                  onChange={(e) => setPartyA(e.target.value)}
                  placeholder="Acme Corp"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Party B *</label>
                <input
                  type="text"
                  value={partyB}
                  onChange={(e) => setPartyB(e.target.value)}
                  placeholder="Vendor Inc"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Effective Date *</label>
                <input
                  type="date"
                  value={effectiveDate}
                  onChange={(e) => setEffectiveDate(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Expiration Date *</label>
                <input
                  type="date"
                  value={expirationDate}
                  onChange={(e) => setExpirationDate(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Regulatory References (comma-separated)</label>
                <input
                  type="text"
                  value={contractRegulatoryRefs}
                  onChange={(e) => setContractRegulatoryRefs(e.target.value)}
                  placeholder="REG-2024-001, REG-2024-002"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Lease References (comma-separated)</label>
                <input
                  type="text"
                  value={leaseReferences}
                  onChange={(e) => setLeaseReferences(e.target.value)}
                  placeholder="LEASE-001, LEASE-002"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                  Obligations (JSON array, optional)
                  <span style={{ color: '#4a9eff', fontSize: '0.75rem', marginLeft: '0.5rem' }}>
                    ⚡ Auto-extracted if empty
                  </span>
                </label>
                <textarea
                  value={contractObligations}
                  onChange={(e) => setContractObligations(e.target.value)}
                  placeholder='[{"obligation_id": "OBL-002", "type": "payment", "deadline": "net_30", "owner": "party_b", "amount": 10000.00}]\n\nLeave empty to auto-extract from document'
                  rows={4}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                  Clauses (JSON array, optional)
                  <span style={{ color: '#4a9eff', fontSize: '0.75rem', marginLeft: '0.5rem' }}>
                    ⚡ Auto-extracted if empty
                  </span>
                </label>
                <textarea
                  value={contractClauses}
                  onChange={(e) => setContractClauses(e.target.value)}
                  placeholder='[{"clause_id": "CLAUSE-002", "type": "payment_terms", "section": "Section 4.2", "text": "Payment shall be due within 30 days..."}]\n\nLeave empty to auto-extract from document'
                  rows={4}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem' }}
                />
              </div>
            </div>
          </div>
        )}

        {/* REGULATION Fields */}
        {docType === 'regulation' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Regulation ID *</label>
                <input
                  type="text"
                  value={regulationId}
                  onChange={(e) => setRegulationId(e.target.value)}
                  placeholder="REG-2024-001"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Regulation Name *</label>
                <input
                  type="text"
                  value={regulationName}
                  onChange={(e) => setRegulationName(e.target.value)}
                  placeholder="Commercial Lease Standards Act"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Jurisdiction *</label>
                <input
                  type="text"
                  value={jurisdiction}
                  onChange={(e) => setJurisdiction(e.target.value)}
                  placeholder="State of California"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Effective Date *</label>
                <input
                  type="date"
                  value={regulationEffectiveDate}
                  onChange={(e) => setRegulationEffectiveDate(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Impacted Contracts (comma-separated)</label>
                <input
                  type="text"
                  value={impactedContracts}
                  onChange={(e) => setImpactedContracts(e.target.value)}
                  placeholder="CONTRACT-001, CONTRACT-002"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Impacted Leases (comma-separated)</label>
                <input
                  type="text"
                  value={impactedLeases}
                  onChange={(e) => setImpactedLeases(e.target.value)}
                  placeholder="LEASE-001, LEASE-002"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div>
              <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Language Changes (JSON array, optional)</label>
              <textarea
                value={languageChanges}
                onChange={(e) => setLanguageChanges(e.target.value)}
                placeholder='[{"section": "Section 5.2", "old_text": "Landlords must provide...", "new_text": "Landlords shall provide...", "change_type": "mandatory_language", "impact": "high"}]'
                rows={4}
                style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem' }}
              />
            </div>
          </div>
        )}

        {/* AMENDMENT Fields */}
        {docType === 'amendment' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Amendment ID *</label>
                <input
                  type="text"
                  value={amendmentId}
                  onChange={(e) => setAmendmentId(e.target.value)}
                  placeholder="AMEND-001"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Amendment Date *</label>
                <input
                  type="date"
                  value={amendmentDate}
                  onChange={(e) => setAmendmentDate(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Amends Document ID *</label>
                <input
                  type="text"
                  value={amendsDocumentId}
                  onChange={(e) => setAmendsDocumentId(e.target.value)}
                  placeholder="LEASE-001"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Amends Document Type *</label>
                <select
                  value={amendsDocumentType}
                  onChange={(e) => setAmendsDocumentType(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                >
                  <option value="lease">Lease</option>
                  <option value="contract">Contract</option>
                </select>
              </div>
            </div>
            <div>
              <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Changes (JSON array, optional)</label>
              <textarea
                value={amendmentChanges}
                onChange={(e) => setAmendmentChanges(e.target.value)}
                placeholder='[{"clause_id": "CLAUSE-001", "change_type": "modified", "old_text": "Tenant shall pay $5,000...", "new_text": "Tenant shall pay $5,500..."}]'
                rows={4}
                style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem' }}
              />
            </div>
          </div>
        )}

        {/* COMPLIANCE_REPORT Fields */}
        {docType === 'compliance_report' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Report ID *</label>
                <input
                  type="text"
                  value={reportId}
                  onChange={(e) => setReportId(e.target.value)}
                  placeholder="COMP-001"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Report Date *</label>
                <input
                  type="date"
                  value={reportDate}
                  onChange={(e) => setReportDate(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Report Type *</label>
                <select
                  value={reportType}
                  onChange={(e) => setReportType(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                >
                  <option value="quarterly">Quarterly</option>
                  <option value="annual">Annual</option>
                  <option value="ad-hoc">Ad-hoc</option>
                </select>
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Compliance Status *</label>
                <select
                  value={complianceStatus}
                  onChange={(e) => setComplianceStatus(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                >
                  <option value="compliant">Compliant</option>
                  <option value="non_compliant">Non-Compliant</option>
                  <option value="partial">Partial</option>
                </select>
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Violations (JSON array, optional)</label>
                <textarea
                  value={violations}
                  onChange={(e) => setViolations(e.target.value)}
                  placeholder='[{"violation_id": "VIOL-001", "regulation_id": "REG-2024-001", "contract_id": "CONTRACT-001", "violation_type": "deadline_missed", "pattern": "recurring"}]'
                  rows={4}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Patterns (JSON array, optional)</label>
                <textarea
                  value={patterns}
                  onChange={(e) => setPatterns(e.target.value)}
                  placeholder='[{"pattern_id": "PATTERN-001", "pattern_type": "recurring_violation", "frequency": "monthly", "affected_documents": ["CONTRACT-001", "CONTRACT-002"]}]'
                  rows={4}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem' }}
                />
              </div>
            </div>
          </div>
        )}

        {/* TEMPLATE Fields */}
        {docType === 'template' && (
          <div>
            <div>
              <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Template ID (optional)</label>
              <input
                type="text"
                value={templateId}
                onChange={(e) => setTemplateId(e.target.value)}
                placeholder="TEMPLATE-001"
                style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
              />
            </div>
            <p style={{ color: '#b0b0b0', fontSize: '0.85rem', marginTop: '0.5rem' }}>
              Note: is_template will be automatically set to true
            </p>
          </div>
        )}

        {/* LEGAL_DOCUMENT - No specific fields */}
        {docType === 'legal_document' && (
          <div>
            <p style={{ color: '#b0b0b0', fontSize: '0.9rem' }}>
              No specific fields required. Use the metadata JSON field below to add custom metadata.
            </p>
          </div>
        )}
      </div>

      {/* Generated Metadata JSON (Read-only) */}
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem' }}>Generated Metadata JSON</label>
        <textarea 
          value={metadata}
          readOnly
          rows={10}
          style={{ fontFamily: 'monospace', width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
        />
      </div>

      {/* Submit Button */}
      <div style={{ marginBottom: '1rem' }}>
        <button
          onClick={handleFileUpload}
          disabled={!file || loading}
          style={{
            width: '100%',
            padding: '0.75rem 1.5rem',
            fontSize: '1rem',
            background: (!file || loading) ? '#444' : '#4a9eff',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: (!file || loading) ? 'not-allowed' : 'pointer',
            fontWeight: 'bold'
          }}
        >
          {loading ? 'Indexing Document...' : 'Index Document'}
      </button>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{
          marginTop: '1rem',
          padding: '1rem',
          background: '#3a1a1a',
          color: '#ff6b6b',
          border: '1px solid #ff6b6b',
          borderRadius: '4px'
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Success Display - JSON Format */}
      {result && (
        <div style={{
          marginTop: '1rem',
          padding: '1.5rem',
          background: '#1a1a1a',
          border: '1px solid #4a8a4a',
          borderRadius: '8px',
          color: '#e0e0e0'
        }}>
          <h3 style={{ color: '#6bff6b', marginTop: 0, marginBottom: '1rem' }}>
            ✅ {result.message || 'Document Indexed Successfully!'}
          </h3>
          <pre style={{
            margin: 0,
            padding: '1rem',
            background: '#0a0a0a',
            border: '1px solid #333',
            borderRadius: '4px',
            overflow: 'auto',
            fontFamily: 'monospace',
            fontSize: '0.9rem',
            lineHeight: '1.5',
            color: '#e0e0e0',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word'
          }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
        </div>
      )}

      {/* Search Tab Content */}
      {activeTab === 'search' && (
        <div>
          {/* Document Search Section */}
          <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
            <h3 style={{ color: '#e0e0e0', marginBottom: '1rem' }}>🔍 Search Documents</h3>
            
            {/* Search Query */}
            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                Search Query *
              </label>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="e.g., rent payment obligations, lease terms, contract clauses..."
                style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
              />
            </div>

            {/* Filters Row */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                  Document Types (optional)
                </label>
                <select
                  multiple
                  value={searchDocTypes}
                  onChange={(e) => {
                    const selected = Array.from(e.target.selectedOptions, option => option.value);
                    setSearchDocTypes(selected);
                  }}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px', minHeight: '100px' }}
                >
                  <option value="legal_document">Legal Document</option>
                  <option value="lease">Lease</option>
                  <option value="contract">Contract</option>
                  <option value="regulation">Regulation</option>
                  <option value="amendment">Amendment</option>
                  <option value="compliance_report">Compliance Report</option>
                  <option value="template">Template</option>
                </select>
                <p style={{ color: '#888', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                  Hold Ctrl/Cmd to select multiple
                </p>
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                  Industry Filter (optional)
                </label>
                <input
                  type="text"
                  value={searchIndustry}
                  onChange={(e) => setSearchIndustry(e.target.value)}
                  placeholder="legal"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>

            {/* Top K and Metadata Filters */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                  Number of Results
                </label>
                <input
                  type="number"
                  value={searchTopK}
                  onChange={(e) => setSearchTopK(parseInt(e.target.value) || 5)}
                  min={1}
                  max={50}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                  Metadata Filters (JSON, optional)
                </label>
                <textarea
                  value={searchMetadataFilters}
                  onChange={(e) => setSearchMetadataFilters(e.target.value)}
                  placeholder='{"lease_id": "LEASE-001"}'
                  rows={2}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem' }}
                />
              </div>
            </div>

            {/* Search Button */}
            <button
              onClick={handleSearch}
              disabled={!searchQuery.trim() || searchLoading}
              style={{
                width: '100%',
                padding: '0.75rem 1.5rem',
                fontSize: '1rem',
                background: (!searchQuery.trim() || searchLoading) ? '#444' : '#4a9eff',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: (!searchQuery.trim() || searchLoading) ? 'not-allowed' : 'pointer',
                fontWeight: 'bold'
              }}
            >
              {searchLoading ? 'Searching...' : 'Search Documents'}
            </button>

            {/* Search Error Display */}
            {searchError && (
              <div style={{
                marginTop: '1rem',
                padding: '1rem',
                background: '#3a1a1a',
                color: '#ff6b6b',
                border: '1px solid #ff6b6b',
                borderRadius: '4px'
              }}>
                <strong>Search Error:</strong> {searchError}
              </div>
            )}

            {/* Search Results */}
            {searchResults.length > 0 && (
              <div style={{ marginTop: '1.5rem' }}>
                <h4 style={{ color: '#e0e0e0', marginBottom: '1rem' }}>
                  Results ({searchResults.length})
                </h4>
                <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
                  {searchResults.map((result, index) => (
                    <div
                      key={index}
                      style={{
                        background: '#1a1a1a',
                        border: '1px solid #444',
                        borderRadius: '4px',
                        padding: '1rem',
                        marginBottom: '1rem'
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '0.5rem' }}>
                        <div>
                          <h5 style={{ color: '#4a9eff', margin: 0, fontSize: '1rem' }}>
                            {result.title || 'Untitled Document'}
                          </h5>
                          <p style={{ color: '#888', fontSize: '0.85rem', margin: '0.25rem 0' }}>
                            Document ID: {result.document_id} | Type: {result.doc_type} | Chunk {result.chunk_index + 1}/{result.total_chunks}
                          </p>
                        </div>
                        {result.score !== undefined && (
                          <span style={{
                            background: '#4a9eff',
                            color: '#fff',
                            padding: '0.25rem 0.5rem',
                            borderRadius: '4px',
                            fontSize: '0.85rem',
                            fontWeight: 'bold'
                          }}
                          title="Similarity score: how closely this result matches your search query"
                          >
                            Similarity: {(result.score * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                      <div style={{
                        background: '#0a0a0a',
                        padding: '0.75rem',
                        borderRadius: '4px',
                        marginTop: '0.5rem'
                      }}>
                        <p style={{ color: '#e0e0e0', margin: 0, fontSize: '0.9rem', lineHeight: '1.5' }}>
                          {result.content}
                        </p>
                      </div>
                      {result.metadata && Object.keys(result.metadata).length > 0 && (
                        <details style={{ marginTop: '0.5rem' }}>
                          <summary style={{ color: '#b0b0b0', cursor: 'pointer', fontSize: '0.85rem' }}>
                            View Metadata
                          </summary>
                          <pre style={{
                            marginTop: '0.5rem',
                            padding: '0.5rem',
                            background: '#0a0a0a',
                            border: '1px solid #333',
                            borderRadius: '4px',
                            overflow: 'auto',
                            fontSize: '0.75rem',
                            color: '#e0e0e0'
                          }}>
                            {JSON.stringify(result.metadata, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Browse Tab Content */}
      {activeTab === 'browse' && (
        <div>
          <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
            <h3 style={{ color: '#e0e0e0', marginBottom: '1rem' }}>📚 Browse Documents</h3>
            
            {/* Filters */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                  Document Type Filter (optional)
                </label>
                <select
                  value={browseDocType}
                  onChange={(e) => setBrowseDocType(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                >
                  <option value="">All Types</option>
                  <option value="legal_document">Legal Document</option>
                  <option value="lease">Lease</option>
                  <option value="contract">Contract</option>
                  <option value="regulation">Regulation</option>
                  <option value="amendment">Amendment</option>
                  <option value="compliance_report">Compliance Report</option>
                  <option value="template">Template</option>
                </select>
              </div>
              <div>
                <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                  Industry Filter (optional)
                </label>
                <input
                  type="text"
                  value={browseIndustry}
                  onChange={(e) => setBrowseIndustry(e.target.value)}
                  placeholder="legal"
                  style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
                />
              </div>
            </div>

            <button
              onClick={handleBrowseDocuments}
              disabled={browseLoading}
              style={{
                width: '100%',
                padding: '0.75rem 1.5rem',
                fontSize: '1rem',
                background: browseLoading ? '#444' : '#4a9eff',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: browseLoading ? 'not-allowed' : 'pointer',
                fontWeight: 'bold',
                marginBottom: '1rem'
              }}
            >
              {browseLoading ? 'Loading...' : 'Load Documents'}
            </button>

            {browseError && (
              <div style={{
                padding: '1rem',
                background: '#3a1a1a',
                color: '#ff6b6b',
                border: '1px solid #ff6b6b',
                borderRadius: '4px',
                marginBottom: '1rem'
              }}>
                <strong>Error:</strong> {browseError}
              </div>
            )}

            {/* Document List */}
            {documentList.length > 0 && (
              <div>
                <h4 style={{ color: '#e0e0e0', marginBottom: '1rem' }}>
                  Documents ({documentList.length})
                </h4>
                <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ background: '#1a1a1a', borderBottom: '2px solid #444' }}>
                        <th style={{ padding: '0.75rem', textAlign: 'left', color: '#e0e0e0', fontSize: '0.9rem' }}>Document ID</th>
                        <th style={{ padding: '0.75rem', textAlign: 'left', color: '#e0e0e0', fontSize: '0.9rem' }}>Title</th>
                        <th style={{ padding: '0.75rem', textAlign: 'left', color: '#e0e0e0', fontSize: '0.9rem' }}>Type</th>
                        <th style={{ padding: '0.75rem', textAlign: 'left', color: '#e0e0e0', fontSize: '0.9rem' }}>Chunks</th>
                        <th style={{ padding: '0.75rem', textAlign: 'left', color: '#e0e0e0', fontSize: '0.9rem' }}>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {documentList.map((doc, index) => (
                        <tr key={index} style={{ borderBottom: '1px solid #333' }}>
                          <td style={{ padding: '0.75rem', color: '#b0b0b0', fontSize: '0.85rem' }}>{doc.document_id}</td>
                          <td style={{ padding: '0.75rem', color: '#e0e0e0', fontSize: '0.9rem' }}>{doc.title}</td>
                          <td style={{ padding: '0.75rem', color: '#b0b0b0', fontSize: '0.85rem' }}>{doc.doc_type}</td>
                          <td style={{ padding: '0.75rem', color: '#b0b0b0', fontSize: '0.85rem' }}>{doc.chunk_count}</td>
                          <td style={{ padding: '0.75rem' }}>
                            <div style={{ display: 'flex', gap: '0.5rem' }}>
                              <button
                                onClick={() => {
                                  setSelectedDocumentId(doc.document_id);
                                  handleGetDocumentDetails(doc.document_id);
                                }}
                                style={{
                                  padding: '0.25rem 0.75rem',
                                  background: '#4a9eff',
                                  color: '#fff',
                                  border: 'none',
                                  borderRadius: '4px',
                                  cursor: 'pointer',
                                  fontSize: '0.85rem'
                                }}
                              >
                                View Details
                              </button>
                              <button
                                onClick={() => handleDeleteDocument(doc.document_id)}
                                disabled={deletingDocId === doc.document_id}
                                style={{
                                  padding: '0.25rem 0.75rem',
                                  background: deletingDocId === doc.document_id ? '#444' : '#ff4444',
                                  color: '#fff',
                                  border: 'none',
                                  borderRadius: '4px',
                                  cursor: deletingDocId === doc.document_id ? 'not-allowed' : 'pointer',
                                  fontSize: '0.85rem'
                                }}
                              >
                                {deletingDocId === doc.document_id ? 'Deleting...' : '🗑️ Delete'}
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Document Details */}
            {selectedDocumentId && (
              <div style={{ marginTop: '2rem', background: '#1a1a1a', padding: '1.5rem', borderRadius: '8px', border: '1px solid #444' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h4 style={{ color: '#e0e0e0', margin: 0 }}>Document Details: {selectedDocumentId}</h4>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      onClick={() => handleDeleteDocument(selectedDocumentId)}
                      disabled={deletingDocId === selectedDocumentId}
                      style={{
                        padding: '0.5rem 1rem',
                        background: deletingDocId === selectedDocumentId ? '#444' : '#ff4444',
                        color: '#fff',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: deletingDocId === selectedDocumentId ? 'not-allowed' : 'pointer',
                        fontSize: '0.9rem'
                      }}
                    >
                      {deletingDocId === selectedDocumentId ? 'Deleting...' : '🗑️ Delete Document'}
                    </button>
                    <button
                      onClick={() => {
                        setSelectedDocumentId('');
                        setDocumentDetails(null);
                      }}
                      style={{
                        padding: '0.5rem 1rem',
                        background: '#666',
                        color: '#fff',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      Close
                    </button>
                  </div>
                </div>
                
                {deleteError && (
                  <div style={{
                    padding: '0.75rem',
                    background: '#3a1a1a',
                    color: '#ff6b6b',
                    border: '1px solid #ff6b6b',
                    borderRadius: '4px',
                    marginBottom: '1rem'
                  }}>
                    <strong>Delete Error:</strong> {deleteError}
                  </div>
                )}
                
                {detailsLoading ? (
                  <p style={{ color: '#b0b0b0' }}>Loading details...</p>
                ) : documentDetails ? (
                  <div>
                    <pre style={{
                      background: '#0a0a0a',
                      padding: '1rem',
                      borderRadius: '4px',
                      overflow: 'auto',
                      fontSize: '0.85rem',
                      color: '#e0e0e0',
                      maxHeight: '500px'
                    }}>
                      {JSON.stringify(documentDetails, null, 2)}
                    </pre>
                  </div>
                ) : (
                  <p style={{ color: '#b0b0b0' }}>No details available</p>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Statistics Tab Content */}
      {activeTab === 'stats' && (
        <div>
          <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ color: '#e0e0e0', margin: 0 }}>📊 Statistics</h3>
              <button
                onClick={handleGetStatistics}
                disabled={statsLoading}
                style={{
                  padding: '0.5rem 1rem',
                  background: statsLoading ? '#444' : '#4a9eff',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: statsLoading ? 'not-allowed' : 'pointer',
                  fontSize: '0.9rem'
                }}
              >
                {statsLoading ? 'Loading...' : '🔄 Refresh'}
              </button>
            </div>

            {statsError && (
              <div style={{
                padding: '1rem',
                background: '#3a1a1a',
                color: '#ff6b6b',
                border: '1px solid #ff6b6b',
                borderRadius: '4px',
                marginBottom: '1rem'
              }}>
                <strong>Error:</strong> {statsError}
              </div>
            )}

            {stats ? (
              <div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                  <div style={{ background: '#1a1a1a', padding: '1rem', borderRadius: '4px', border: '1px solid #444' }}>
                    <div style={{ color: '#b0b0b0', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Collection Name</div>
                    <div style={{ color: '#4a9eff', fontSize: '1.25rem', fontWeight: 'bold' }}>
                      {stats.collection_name || 'N/A'}
                    </div>
                  </div>
                  <div style={{ background: '#1a1a1a', padding: '1rem', borderRadius: '4px', border: '1px solid #444' }}>
                    <div style={{ color: '#b0b0b0', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Total Chunks</div>
                    <div style={{ color: '#6bff6b', fontSize: '1.25rem', fontWeight: 'bold' }}>
                      {stats.num_entities || 0}
                    </div>
                  </div>
                  <div style={{ background: '#1a1a1a', padding: '1rem', borderRadius: '4px', border: '1px solid #444' }}>
                    <div style={{ color: '#b0b0b0', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Milvus Status</div>
                    <div style={{ color: stats.milvus_available ? '#6bff6b' : '#ff6b6b', fontSize: '1.25rem', fontWeight: 'bold' }}>
                      {stats.milvus_available ? '✓ Available' : '✗ Unavailable'}
                    </div>
                  </div>
                  <div style={{ background: '#1a1a1a', padding: '1rem', borderRadius: '4px', border: '1px solid #444' }}>
                    <div style={{ color: '#b0b0b0', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Health Status</div>
                    <div style={{ color: stats.healthy ? '#6bff6b' : '#ff6b6b', fontSize: '1.25rem', fontWeight: 'bold' }}>
                      {stats.healthy ? '✓ Healthy' : '✗ Unhealthy'}
                    </div>
                  </div>
                </div>

                <div style={{ background: '#1a1a1a', padding: '1rem', borderRadius: '4px', border: '1px solid #444' }}>
                  <h4 style={{ color: '#e0e0e0', marginTop: 0, marginBottom: '1rem' }}>Full Statistics</h4>
                  <pre style={{
                    background: '#0a0a0a',
                    padding: '1rem',
                    borderRadius: '4px',
                    overflow: 'auto',
                    fontSize: '0.85rem',
                    color: '#e0e0e0',
                    margin: 0
                  }}>
                    {JSON.stringify(stats, null, 2)}
                  </pre>
                </div>
              </div>
            ) : statsLoading ? (
              <p style={{ color: '#b0b0b0' }}>Loading statistics...</p>
            ) : (
              <p style={{ color: '#b0b0b0' }}>Click "Refresh" to load statistics</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
