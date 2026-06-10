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
  const [activeTab, setActiveTab] = useState<'index' | 'search' | 'browse' | 'stats' | 'batch' | 'versions'>('index');

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

  // Batch indexing state
  const [batchFiles, setBatchFiles] = useState<Array<{file: File, docType: string}>>([]);
  const [batchIndustry, setBatchIndustry] = useState('legal');
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchResult, setBatchResult] = useState<any>(null);
  const [batchError, setBatchError] = useState<string | null>(null);

  // Versions & Obligations state
  const [versionsDocId, setVersionsDocId] = useState('');
  const [versionsLoading, setVersionsLoading] = useState(false);
  const [versionsData, setVersionsData] = useState<any>(null);
  const [versionsError, setVersionsError] = useState<string | null>(null);
  const [obligationsDocId, setObligationsDocId] = useState('');
  const [obligationsLoading, setObligationsLoading] = useState(false);
  const [obligationsData, setObligationsData] = useState<any>(null);
  const [obligationsError, setObligationsError] = useState<string | null>(null);

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
      
      // Refresh statistics if stats tab is active or stats have been loaded
      if (activeTab === 'stats' || stats !== null) {
        handleGetStatistics();
      }
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
        const errorMessage = errorData.detail || errorData.message || `HTTP ${response.status}`;
        throw new Error(errorMessage);
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
        
        // Refresh statistics if stats tab is active or stats have been loaded
        // Add delay to allow Milvus to flush the deletion (backend already waits 1.5s, but we add extra buffer)
        // Milvus deletions are asynchronous and may need time to update the entity count
        if (activeTab === 'stats' || stats !== null) {
          setTimeout(() => {
            handleGetStatistics();
          }, 2000); // 2 second delay to ensure Milvus has flushed the deletion
        }
        
        alert(`Document "${docId}" deleted successfully`);
      } else {
        // Backend returned success: false with a message
        const errorMessage = data.message || data.detail || 'Delete operation returned unsuccessful';
        throw new Error(errorMessage);
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
    // Clear stats to show loading state
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
      // Always update stats, even if num_entities is 0 - this ensures refresh works
      setStats(data.statistics || data || { num_entities: 0 });
    } catch (error: any) {
      setStatsError(error.message || 'Failed to get statistics');
      setStats(null); // Clear stats on error
      console.error('Stats error:', error);
    } finally {
      setStatsLoading(false);
    }
  };

  // Load stats when stats tab is opened
  useEffect(() => {
    if (activeTab === 'stats' && !statsLoading) {
      handleGetStatistics();
    }
  }, [activeTab]);

  const handleBatchIndex = async () => {
    if (batchFiles.length === 0) {
      setBatchError('Please select at least one file to upload');
      return;
    }
    
    setBatchLoading(true);
    setBatchError(null);
    setBatchResult(null);

    try {
      // Upload and index files sequentially
      const results = {
        successful: [] as any[],
        failed: [] as any[]
      };

      for (const fileItem of batchFiles) {
        try {
          const formData = new FormData();
          formData.append('file', fileItem.file);
          formData.append('doc_type', fileItem.docType); // Use each file's individual docType
          formData.append('industry', batchIndustry);
          
          // Build metadata based on the file's docType
          // For batch, we'll use minimal metadata (just document_type)
          // Users can add more metadata via the single file indexing if needed
          const fileMetadata = {
            document_type: fileItem.docType
          };
          formData.append('metadata', JSON.stringify(fileMetadata));

          const uploadUrl = `${baseUrl || 'http://localhost:8001'}/api/v1/documents/index/file`;
          const uploadResponse = await fetch(uploadUrl, {
            method: 'POST',
            body: formData,
            headers: apiKey ? { 'x-api-key': apiKey } : {}
          });

          if (!uploadResponse.ok) {
            const errorData = await uploadResponse.json().catch(() => ({ detail: `HTTP ${uploadResponse.status}` }));
            throw new Error(errorData.detail || `HTTP ${uploadResponse.status}`);
          }

          const uploadData = await uploadResponse.json();
          results.successful.push({
            document_id: uploadData.document_id,
            title: uploadData.title,
            chunk_count: uploadData.chunks,
            file_name: fileItem.file.name,
            doc_type: fileItem.docType
          });
        } catch (error: any) {
          results.failed.push({
            file_name: fileItem.file.name,
            error: error.message || 'Failed to index file'
          });
        }
      }

      // Format result similar to batch endpoint response
      const total = batchFiles.length;
      const successCount = results.successful.length;
      const failedCount = results.failed.length;
      const successRate = total > 0 ? successCount / total : 0;

      setBatchResult({
        total,
        success_count: successCount,
        failed_count: failedCount,
        success_rate: successRate,
        successful: results.successful,
        failed: results.failed
      });
      
      // Refresh statistics if stats tab is active or stats have been loaded
      if (activeTab === 'stats' || stats !== null) {
        handleGetStatistics();
      }
    } catch (error: any) {
      setBatchError(error.message || 'Failed to batch index files');
      console.error('Batch index error:', error);
    } finally {
      setBatchLoading(false);
    }
  };

  const handleGetVersions = async () => {
    if (!versionsDocId.trim()) {
      setVersionsError('Please enter a document ID');
      return;
    }

    setVersionsLoading(true);
    setVersionsError(null);
    setVersionsData(null);

    try {
      const apiUrl = `${baseUrl || 'http://localhost:8001'}/api/v1/documents/${encodeURIComponent(versionsDocId.trim())}/versions`;
      
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
      setVersionsData(data);
    } catch (error: any) {
      setVersionsError(error.message || 'Failed to get document versions');
      console.error('Versions error:', error);
    } finally {
      setVersionsLoading(false);
    }
  };

  const handleGetObligations = async () => {
    if (!obligationsDocId.trim()) {
      setObligationsError('Please enter a document ID');
      return;
    }

    setObligationsLoading(true);
    setObligationsError(null);
    setObligationsData(null);

    try {
      const apiUrl = `${baseUrl || 'http://localhost:8001'}/api/v1/documents/${encodeURIComponent(obligationsDocId.trim())}/obligations`;
      
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
      setObligationsData(data);
    } catch (error: any) {
      setObligationsError(error.message || 'Failed to get document obligations');
      console.error('Obligations error:', error);
    } finally {
      setObligationsLoading(false);
    }
  };

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
        <button
          onClick={() => setActiveTab('batch')}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '0.95rem',
            background: activeTab === 'batch' ? '#4a9eff' : 'transparent',
            color: activeTab === 'batch' ? '#fff' : '#b0b0b0',
            border: 'none',
            borderBottom: activeTab === 'batch' ? '2px solid #4a9eff' : '2px solid transparent',
            borderRadius: '4px 4px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'batch' ? 'bold' : 'normal',
            transition: 'all 0.2s'
          }}
        >
          📦 Batch Index
        </button>
        <button
          onClick={() => setActiveTab('versions')}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '0.95rem',
            background: activeTab === 'versions' ? '#4a9eff' : 'transparent',
            color: activeTab === 'versions' ? '#fff' : '#b0b0b0',
            border: 'none',
            borderBottom: activeTab === 'versions' ? '2px solid #4a9eff' : '2px solid transparent',
            borderRadius: '4px 4px 0 0',
            cursor: 'pointer',
            fontWeight: activeTab === 'versions' ? 'bold' : 'normal',
            transition: 'all 0.2s'
          }}
        >
          🔄 Version Control
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
            {browseLoading ? (
              <p style={{ color: '#b0b0b0', textAlign: 'center', padding: '2rem' }}>Loading documents...</p>
            ) : documentList.length > 0 ? (
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
            ) : (
              <div style={{
                padding: '2rem',
                textAlign: 'center',
                background: '#1a1a1a',
                borderRadius: '4px',
                border: '1px solid #444'
              }}>
                <p style={{ color: '#b0b0b0', fontSize: '1.1rem', marginBottom: '0.5rem' }}>
                  📭 No documents found
                </p>
                <p style={{ color: '#888', fontSize: '0.9rem' }}>
                  {browseDocType || browseIndustry 
                    ? 'No documents match the selected filters. Try adjusting your filters or index documents first.'
                    : 'No documents have been indexed yet. Start by indexing documents in the "Index" tab.'}
                </p>
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

            {statsLoading ? (
              <p style={{ color: '#b0b0b0' }}>Loading statistics...</p>
            ) : stats ? (
              <div>
                {/* Stats cards */}
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

                {/* Full stats JSON */}
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
            ) : (
              <p style={{ color: '#b0b0b0' }}>Click "Refresh" to load statistics</p>
            )}
          </div>
        </div>
      )}

      {/* Batch Index Tab Content */}
      {activeTab === 'batch' && (
        <div>
          <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
            <h3 style={{ color: '#e0e0e0', marginBottom: '1rem' }}>📦 Batch Index Multiple Files</h3>
            
            {/* File Selection */}
            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                Select Files (multiple) *
              </label>
              <input
                key={`batch-file-input-${Date.now()}`}
                type="file"
                multiple={true}
                onChange={(e) => {
                  const selectedFiles = Array.from(e.target.files || []);
                  // Append new files to existing ones, avoiding duplicates by name
                  setBatchFiles(prevFiles => {
                    const existingNames = new Set(prevFiles.map(f => f.file.name));
                    const newFiles = selectedFiles
                      .filter(f => !existingNames.has(f.name))
                      .map(file => ({ file, docType: 'legal_document' })); // Default docType
                    return [...prevFiles, ...newFiles];
                  });
                  // Reset the input value so the same file can be selected again if needed
                  e.target.value = '';
                }}
                accept=".pdf,.docx,.txt,.md,.csv"
                style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
              />
              <p style={{ color: '#888', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                Hold Ctrl (Windows/Linux) or Cmd (Mac) to select multiple files
              </p>
              {batchFiles.length > 0 && (
                <div style={{ marginTop: '0.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <p style={{ color: '#b0b0b0', fontSize: '0.85rem', margin: 0 }}>
                      Selected {batchFiles.length} file(s):
                    </p>
                    <button
                      onClick={() => setBatchFiles([])}
                      style={{
                        padding: '0.25rem 0.75rem',
                        background: '#ff4444',
                        color: '#fff',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.75rem'
                      }}
                    >
                      Clear All
                    </button>
                  </div>
                  <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                    {batchFiles.map((fileItem, index) => (
                      <div key={index} style={{ 
                        marginBottom: '0.75rem', 
                        padding: '0.75rem', 
                        background: '#1a1a1a', 
                        borderRadius: '4px',
                        border: '1px solid #444'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                          <div style={{ flex: 1 }}>
                            <div style={{ color: '#e0e0e0', fontSize: '0.9rem', fontWeight: 'bold' }}>
                              {fileItem.file.name}
                            </div>
                            <div style={{ color: '#888', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                              {(fileItem.file.size / 1024).toFixed(2)} KB
                            </div>
                          </div>
                          <button
                            onClick={() => {
                              const newFiles = batchFiles.filter((_, i) => i !== index);
                              setBatchFiles(newFiles);
                            }}
                            style={{
                              padding: '0.25rem 0.75rem',
                              background: '#ff4444',
                              color: '#fff',
                              border: 'none',
                              borderRadius: '3px',
                              cursor: 'pointer',
                              fontSize: '0.75rem',
                              marginLeft: '0.5rem'
                            }}
                          >
                            Remove
                          </button>
                        </div>
                        <div>
                          <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.25rem', fontSize: '0.8rem' }}>
                            Document Type
                          </label>
                          <select
                            value={fileItem.docType}
                            onChange={(e) => {
                              const newFiles = [...batchFiles];
                              newFiles[index] = { ...newFiles[index], docType: e.target.value };
                              setBatchFiles(newFiles);
                            }}
                            style={{
                              width: '100%',
                              padding: '0.4rem',
                              background: '#0a0a0a',
                              color: '#e0e0e0',
                              border: '1px solid #555',
                              borderRadius: '4px',
                              fontSize: '0.85rem'
                            }}
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
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Industry (applies to all files) */}
            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', color: '#b0b0b0', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                Industry (applies to all files)
              </label>
              <input
                type="text"
                value={batchIndustry}
                onChange={(e) => setBatchIndustry(e.target.value)}
                placeholder="legal"
                style={{ width: '100%', padding: '0.5rem', background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #444', borderRadius: '4px' }}
              />
              <p style={{ color: '#888', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                Note: Each file can have its own document type (set above). Industry applies to all files in the batch.
              </p>
            </div>

            {/* Submit Button */}
            <button
              onClick={handleBatchIndex}
              disabled={batchFiles.length === 0 || batchLoading}
              style={{
                width: '100%',
                padding: '0.75rem 1.5rem',
                fontSize: '1rem',
                background: (batchFiles.length === 0 || batchLoading) ? '#444' : '#4a9eff',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: (batchFiles.length === 0 || batchLoading) ? 'not-allowed' : 'pointer',
                fontWeight: 'bold'
              }}
            >
              {batchLoading ? `Indexing ${batchFiles.length} file(s)...` : `Index ${batchFiles.length} File(s)`}
            </button>

            {/* Error Display */}
            {batchError && (
              <div style={{
                marginTop: '1rem',
                padding: '1rem',
                background: '#3a1a1a',
                color: '#ff6b6b',
                border: '1px solid #ff6b6b',
                borderRadius: '4px'
              }}>
                <strong>Error:</strong> {batchError}
              </div>
            )}

            {/* Results Display */}
            {batchResult && (
              <div style={{
                marginTop: '1rem',
                padding: '1.5rem',
                background: '#1a1a1a',
                border: '1px solid #4a8a4a',
                borderRadius: '8px',
                color: '#e0e0e0'
              }}>
                <h4 style={{ color: '#6bff6b', marginTop: 0, marginBottom: '1rem' }}>
                  Batch Indexing Results
                </h4>
                
                {/* Summary Stats */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                  <div style={{ background: '#0a0a0a', padding: '1rem', borderRadius: '4px', border: '1px solid #444' }}>
                    <div style={{ color: '#b0b0b0', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Total</div>
                    <div style={{ color: '#e0e0e0', fontSize: '1.25rem', fontWeight: 'bold' }}>
                      {batchResult.total || 0}
                    </div>
                  </div>
                  <div style={{ background: '#0a0a0a', padding: '1rem', borderRadius: '4px', border: '1px solid #444' }}>
                    <div style={{ color: '#b0b0b0', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Successful</div>
                    <div style={{ color: '#6bff6b', fontSize: '1.25rem', fontWeight: 'bold' }}>
                      {batchResult.success_count || 0}
                    </div>
                  </div>
                  <div style={{ background: '#0a0a0a', padding: '1rem', borderRadius: '4px', border: '1px solid #444' }}>
                    <div style={{ color: '#b0b0b0', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Failed</div>
                    <div style={{ color: '#ff6b6b', fontSize: '1.25rem', fontWeight: 'bold' }}>
                      {batchResult.failed_count || 0}
                    </div>
                  </div>
                  <div style={{ background: '#0a0a0a', padding: '1rem', borderRadius: '4px', border: '1px solid #444' }}>
                    <div style={{ color: '#b0b0b0', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Success Rate</div>
                    <div style={{ color: '#4a9eff', fontSize: '1.25rem', fontWeight: 'bold' }}>
                      {((batchResult.success_rate || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Successful Files */}
                {batchResult.successful && batchResult.successful.length > 0 && (
                  <div style={{ marginBottom: '1.5rem' }}>
                    <h5 style={{ color: '#6bff6b', marginBottom: '0.5rem' }}>✅ Successful ({batchResult.successful.length})</h5>
                    <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                      {batchResult.successful.map((item: any, index: number) => (
                        <div key={index} style={{
                          background: '#0a0a0a',
                          padding: '0.75rem',
                          borderRadius: '4px',
                          marginBottom: '0.5rem',
                          border: '1px solid #333'
                        }}>
                          <div style={{ color: '#e0e0e0', fontSize: '0.9rem' }}>
                            <strong>{item.title || item.file_name}</strong>
                          </div>
                          <div style={{ color: '#b0b0b0', fontSize: '0.85rem', marginTop: '0.25rem' }}>
                            ID: {item.document_id} | Type: {item.doc_type || 'N/A'} | Chunks: {item.chunk_count || 'N/A'}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Failed Files */}
                {batchResult.failed && batchResult.failed.length > 0 && (
                  <div>
                    <h5 style={{ color: '#ff6b6b', marginBottom: '0.5rem' }}>❌ Failed ({batchResult.failed.length})</h5>
                    <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                      {batchResult.failed.map((item: any, index: number) => (
                        <div key={index} style={{
                          background: '#0a0a0a',
                          padding: '0.75rem',
                          borderRadius: '4px',
                          marginBottom: '0.5rem',
                          border: '1px solid #ff4444'
                        }}>
                          <div style={{ color: '#e0e0e0', fontSize: '0.9rem' }}>
                            <strong>{item.file_name || item.file_path}</strong>
                          </div>
                          <div style={{ color: '#ff6b6b', fontSize: '0.85rem', marginTop: '0.25rem' }}>
                            {item.error || 'Unknown error'}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Full JSON Response */}
                <details style={{ marginTop: '1rem' }}>
                  <summary style={{ color: '#b0b0b0', cursor: 'pointer', fontSize: '0.9rem' }}>
                    View Full Response JSON
                  </summary>
                  <pre style={{
                    marginTop: '0.5rem',
                    padding: '1rem',
                    background: '#0a0a0a',
                    border: '1px solid #333',
                    borderRadius: '4px',
                    overflow: 'auto',
                    fontSize: '0.85rem',
                    color: '#e0e0e0',
                    maxHeight: '300px'
                  }}>
                    {JSON.stringify(batchResult, null, 2)}
                  </pre>
                </details>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Version Control Tab Content */}
      {activeTab === 'versions' && (
        <div>
          <div style={{ background: '#2a2a2a', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
            <h3 style={{ color: '#e0e0e0', marginBottom: '1.5rem' }}>🔄 Version Control</h3>
            
            {/* Document Versions Section */}
            <div style={{ marginBottom: '2rem', padding: '1rem', background: '#1a1a1a', borderRadius: '4px', border: '1px solid #444' }}>
              <h4 style={{ color: '#e0e0e0', marginTop: 0, marginBottom: '1rem' }}>📋 Get Document Versions</h4>
              <p style={{ color: '#b0b0b0', fontSize: '0.85rem', marginBottom: '1rem' }}>
                Enter any version's document ID to retrieve all versions of that document. 
                Versions are linked via contract_id, lease_id, or regulation_id in metadata.
              </p>
              
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                <input
                  type="text"
                  value={versionsDocId}
                  onChange={(e) => setVersionsDocId(e.target.value)}
                  placeholder="Enter document ID (e.g., CONTRACT-001-v2.0)"
                  style={{ 
                    flex: 1, 
                    padding: '0.5rem', 
                    background: '#0a0a0a', 
                    color: '#e0e0e0', 
                    border: '1px solid #444', 
                    borderRadius: '4px',
                    fontSize: '0.9rem'
                  }}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleGetVersions();
                    }
                  }}
                />
                <button
                  onClick={handleGetVersions}
                  disabled={versionsLoading || !versionsDocId.trim()}
                  style={{
                    padding: '0.5rem 1rem',
                    background: versionsLoading || !versionsDocId.trim() ? '#444' : '#4a9eff',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: versionsLoading || !versionsDocId.trim() ? 'not-allowed' : 'pointer',
                    fontSize: '0.9rem'
                  }}
                >
                  {versionsLoading ? 'Loading...' : '🔍 Get Versions'}
                </button>
              </div>

              {versionsError && (
                <div style={{
                  padding: '0.75rem',
                  background: '#3a1a1a',
                  color: '#ff6b6b',
                  border: '1px solid #ff6b6b',
                  borderRadius: '4px',
                  marginBottom: '1rem',
                  fontSize: '0.9rem'
                }}>
                  <strong>Error:</strong> {versionsError}
                </div>
              )}

              {versionsData && (
                <div style={{ marginTop: '1rem' }}>
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                    gap: '0.5rem', 
                    marginBottom: '1rem' 
                  }}>
                    <div style={{ background: '#0a0a0a', padding: '0.75rem', borderRadius: '4px', border: '1px solid #333' }}>
                      <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Base ID</div>
                      <div style={{ color: '#4a9eff', fontSize: '0.9rem', fontWeight: 'bold' }}>
                        {versionsData.base_document_id || 'N/A'}
                      </div>
                    </div>
                    <div style={{ background: '#0a0a0a', padding: '0.75rem', borderRadius: '4px', border: '1px solid #333' }}>
                      <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Total Versions</div>
                      <div style={{ color: '#6bff6b', fontSize: '0.9rem', fontWeight: 'bold' }}>
                        {versionsData.total_versions || 0}
                      </div>
                    </div>
                  </div>

                  {versionsData.versions && versionsData.versions.length > 0 ? (
                    <div style={{ marginTop: '1rem' }}>
                      <h5 style={{ color: '#e0e0e0', marginBottom: '0.75rem', fontSize: '0.95rem' }}>Version History</h5>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {versionsData.versions.map((version: any, idx: number) => (
                          <div 
                            key={idx}
                            style={{ 
                              background: '#0a0a0a', 
                              padding: '1rem', 
                              borderRadius: '4px', 
                              border: '1px solid #444',
                              display: 'grid',
                              gridTemplateColumns: '2fr 1fr 1fr 1fr',
                              gap: '1rem',
                              alignItems: 'center'
                            }}
                          >
                            <div>
                              <div style={{ color: '#4a9eff', fontWeight: 'bold', marginBottom: '0.25rem' }}>
                                {version.title || version.document_id}
                              </div>
                              <div style={{ color: '#888', fontSize: '0.75rem' }}>
                                ID: {version.document_id}
                              </div>
                            </div>
                            <div>
                              <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Version</div>
                              <div style={{ color: '#e0e0e0', fontSize: '0.9rem' }}>
                                {version.version || 'N/A'}
                              </div>
                            </div>
                            <div>
                              <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Date</div>
                              <div style={{ color: '#e0e0e0', fontSize: '0.9rem' }}>
                                {version.version_date || 'N/A'}
                              </div>
                            </div>
                            <div>
                              <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Chunks</div>
                              <div style={{ color: '#6bff6b', fontSize: '0.9rem', fontWeight: 'bold' }}>
                                {version.chunk_count || 0}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <p style={{ color: '#b0b0b0', fontSize: '0.9rem', marginTop: '1rem' }}>
                      No versions found. Make sure the document has contract_id, lease_id, or regulation_id in metadata.
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Document Obligations Section */}
            <div style={{ padding: '1rem', background: '#1a1a1a', borderRadius: '4px', border: '1px solid #444' }}>
              <h4 style={{ color: '#e0e0e0', marginTop: 0, marginBottom: '1rem' }}>📝 Get Document Obligations</h4>
              <p style={{ color: '#b0b0b0', fontSize: '0.85rem', marginBottom: '1rem' }}>
                Enter a document ID to retrieve all obligations extracted from that document.
                Obligations can be manually provided or auto-extracted during indexing.
              </p>
              
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                <input
                  type="text"
                  value={obligationsDocId}
                  onChange={(e) => setObligationsDocId(e.target.value)}
                  placeholder="Enter document ID (e.g., LEASE-001)"
                  style={{ 
                    flex: 1, 
                    padding: '0.5rem', 
                    background: '#0a0a0a', 
                    color: '#e0e0e0', 
                    border: '1px solid #444', 
                    borderRadius: '4px',
                    fontSize: '0.9rem'
                  }}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleGetObligations();
                    }
                  }}
                />
                <button
                  onClick={handleGetObligations}
                  disabled={obligationsLoading || !obligationsDocId.trim()}
                  style={{
                    padding: '0.5rem 1rem',
                    background: obligationsLoading || !obligationsDocId.trim() ? '#444' : '#4a9eff',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: obligationsLoading || !obligationsDocId.trim() ? 'not-allowed' : 'pointer',
                    fontSize: '0.9rem'
                  }}
                >
                  {obligationsLoading ? 'Loading...' : '🔍 Get Obligations'}
                </button>
              </div>

              {obligationsError && (
                <div style={{
                  padding: '0.75rem',
                  background: '#3a1a1a',
                  color: '#ff6b6b',
                  border: '1px solid #ff6b6b',
                  borderRadius: '4px',
                  marginBottom: '1rem',
                  fontSize: '0.9rem'
                }}>
                  <strong>Error:</strong> {obligationsError}
                </div>
              )}

              {obligationsData && (
                <div style={{ marginTop: '1rem' }}>
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                    gap: '0.5rem', 
                    marginBottom: '1rem' 
                  }}>
                    <div style={{ background: '#0a0a0a', padding: '0.75rem', borderRadius: '4px', border: '1px solid #333' }}>
                      <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Document</div>
                      <div style={{ color: '#4a9eff', fontSize: '0.9rem', fontWeight: 'bold' }}>
                        {obligationsData.title || obligationsData.document_id || 'N/A'}
                      </div>
                    </div>
                    <div style={{ background: '#0a0a0a', padding: '0.75rem', borderRadius: '4px', border: '1px solid #333' }}>
                      <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Total Obligations</div>
                      <div style={{ color: '#6bff6b', fontSize: '0.9rem', fontWeight: 'bold' }}>
                        {obligationsData.total_obligations || 0}
                      </div>
                    </div>
                  </div>

                  {obligationsData.obligations && obligationsData.obligations.length > 0 ? (
                    <div style={{ marginTop: '1rem' }}>
                      <h5 style={{ color: '#e0e0e0', marginBottom: '0.75rem', fontSize: '0.95rem' }}>Obligations</h5>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {obligationsData.obligations.map((obligation: any, idx: number) => (
                          <div 
                            key={idx}
                            style={{ 
                              background: '#0a0a0a', 
                              padding: '1rem', 
                              borderRadius: '4px', 
                              border: '1px solid #444'
                            }}
                          >
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
                              <div>
                                <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Obligation ID</div>
                                <div style={{ color: '#4a9eff', fontSize: '0.9rem', fontWeight: 'bold' }}>
                                  {obligation.obligation_id || `OBL-${idx + 1}`}
                                </div>
                              </div>
                              <div>
                                <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Type</div>
                                <div style={{ color: '#e0e0e0', fontSize: '0.9rem' }}>
                                  {obligation.type || 'N/A'}
                                </div>
                              </div>
                              <div>
                                <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Owner</div>
                                <div style={{ color: '#e0e0e0', fontSize: '0.9rem' }}>
                                  {obligation.owner || 'N/A'}
                                </div>
                              </div>
                              <div>
                                <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Deadline</div>
                                <div style={{ color: obligation.deadline ? '#ffaa00' : '#888', fontSize: '0.9rem' }}>
                                  {obligation.deadline || 'N/A'}
                                </div>
                              </div>
                              {obligation.amount && (
                                <div>
                                  <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Amount</div>
                                  <div style={{ color: '#6bff6b', fontSize: '0.9rem', fontWeight: 'bold' }}>
                                    ${typeof obligation.amount === 'number' ? obligation.amount.toLocaleString() : obligation.amount}
                                  </div>
                                </div>
                              )}
                              {obligation.frequency && (
                                <div>
                                  <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Frequency</div>
                                  <div style={{ color: '#e0e0e0', fontSize: '0.9rem' }}>
                                    {obligation.frequency}
                                  </div>
                                </div>
                              )}
                            </div>
                            {obligation.description && (
                              <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid #333' }}>
                                <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Description</div>
                                <div style={{ color: '#b0b0b0', fontSize: '0.85rem' }}>
                                  {obligation.description}
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <p style={{ color: '#b0b0b0', fontSize: '0.9rem', marginTop: '1rem' }}>
                      No obligations found. Obligations must be provided in metadata during indexing or auto-extracted.
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
