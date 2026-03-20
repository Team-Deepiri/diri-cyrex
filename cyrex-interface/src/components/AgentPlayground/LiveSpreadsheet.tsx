/**
 * Live Spreadsheet Component
 * 
 * Features:
 * - Real-time cell editing
 * - Number calculations
 * - Visual effects for updates
 * - Multi-agent support
 */

import React, { useState, useEffect, useRef, useCallback, useImperativeHandle, forwardRef } from 'react';
import { FaTable, FaPlus, FaTrash, FaCalculator, FaSync, FaFileImport, FaTimes, FaChevronDown } from 'react-icons/fa';
import './LiveSpreadsheet.css';

const API_BASE = import.meta.env.VITE_CYREX_BASE_URL || 'http://localhost:8000';

interface Cell {
  id: string;
  value: string;
  formula?: string;
  computedValue?: number;
  lastUpdatedBy?: string;
  lastUpdatedAt?: string;
}

interface SpreadsheetData {
  [key: string]: Cell;
}

interface LiveSpreadsheetProps {
  instanceId?: string;
  agentName?: string;
  onCellUpdate?: (cellId: string, value: string, agentName?: string) => void;
  onAgentMessage?: (message: string) => void;
  userId?: string; // User ID for PostgreSQL storage (defaults to 'admin')
}

export interface LiveSpreadsheetRef {
  processMessage: (message: string) => void;
  setCell: (cellId: string, value: string, agentName?: string) => void;
  setCellFromTool: (toolName: string, params: Record<string, unknown>, result?: unknown) => void;
}

// Generate A-Z columns (26 columns)
const INITIAL_COLUMNS = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));
const INITIAL_ROWS = 1000;

// Generate column letter from index (A, B, ..., Z, AA, AB, etc.)
const getColumnLetter = (index: number): string => {
  let result = '';
  let num = index;
  while (num >= 0) {
    result = String.fromCharCode(65 + (num % 26)) + result;
    num = Math.floor(num / 26) - 1;
  }
  return result;
};

interface Document {
  document_id: string;
  title: string;
  doc_type?: string;
  created_at?: string;
}

export const LiveSpreadsheet = forwardRef<LiveSpreadsheetRef, LiveSpreadsheetProps>(
  (props, ref) => {
    const { instanceId, agentName, onCellUpdate, onAgentMessage, userId = 'admin' } = props;
    const [columns, setColumns] = useState<string[]>(INITIAL_COLUMNS);
    const [rowCount, setRowCount] = useState(INITIAL_ROWS);
    const [data, setData] = useState<SpreadsheetData>({});
    const [selectedCell, setSelectedCell] = useState<string | null>(null);
    const [editingCell, setEditingCell] = useState<string | null>(null);
    const [editValue, setEditValue] = useState('');
    const [updateEffects, setUpdateEffects] = useState<Set<string>>(new Set());
    const [agentActivity, setAgentActivity] = useState<Array<{ cellId: string; agentName: string; timestamp: string }>>([]);
    const [showDocumentPicker, setShowDocumentPicker] = useState(false);
    const [documents, setDocuments] = useState<Document[]>([]);
    const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
    const [isImporting, setIsImporting] = useState(false);
    const [isLoadingSpreadsheet, setIsLoadingSpreadsheet] = useState(false);
    const [isSavingSpreadsheet, setIsSavingSpreadsheet] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [parsingStatus, setParsingStatus] = useState<string>('');
    const [parsedPreview, setParsedPreview] = useState<any>(null);
    const [showPreview, setShowPreview] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    
    const cellRefs = useRef<{ [key: string]: HTMLInputElement | null }>({});
    const spreadsheetRef = useRef<HTMLDivElement>(null);
    const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    // Initialize empty cells for new rows/columns while preserving existing data
    useEffect(() => {
      setData(prev => {
        const newData = { ...prev };
        let hasChanges = false;
        
        // Ensure all cells exist for current grid size
        for (let row = 1; row <= rowCount; row++) {
          for (const col of columns) {
            const cellId = `${col}${row}`;
            if (!newData[cellId]) {
              newData[cellId] = {
                id: cellId,
                value: '',
              };
              hasChanges = true;
            }
          }
        }
        
        // Only update if we added new cells (preserve existing data)
        return hasChanges ? newData : prev;
      });
    }, [columns, rowCount]);

    // Save spreadsheet data to PostgreSQL
    const saveSpreadsheetData = useCallback(async () => {
      if (isSavingSpreadsheet) return;
      
      setIsSavingSpreadsheet(true);
      try {
        const response = await fetch(`${API_BASE}/api/agent/spreadsheet/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            instance_id: instanceId,
            agent_name: agentName,
            columns: columns,
            row_count: rowCount,
            data: data,
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to save spreadsheet data');
        }

        const result = await response.json();
        if (result.success) {
          console.log('Spreadsheet data saved successfully');
        }
      } catch (error) {
        console.error('Error saving spreadsheet data:', error);
      } finally {
        setIsSavingSpreadsheet(false);
      }
    }, [userId, instanceId, agentName, columns, rowCount, data, isSavingSpreadsheet]);

    // Load spreadsheet data from PostgreSQL
    const loadSpreadsheetData = useCallback(async () => {
      setIsLoadingSpreadsheet(true);
      try {
        const response = await fetch(`${API_BASE}/api/agent/spreadsheet/load`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            instance_id: instanceId,
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to load spreadsheet data');
        }

        const result = await response.json();
        if (result.success && result.found) {
          setColumns(result.columns || INITIAL_COLUMNS);
          setRowCount(result.row_count || INITIAL_ROWS);
          setData(result.data || {});
          console.log('Spreadsheet data loaded successfully');
        }
      } catch (error) {
        console.error('Error loading spreadsheet data:', error);
      } finally {
        setIsLoadingSpreadsheet(false);
      }
    }, [userId, instanceId]);

    // Load spreadsheet data on mount
    useEffect(() => {
      if (instanceId) {
        loadSpreadsheetData();
      }
    }, [instanceId]); // Only load when instanceId changes

    // Poll for spreadsheet updates (so agent tool changes are visible)
    useEffect(() => {
      if (!instanceId) return;

      const pollInterval = setInterval(async () => {
        try {
          const response = await fetch(`${API_BASE}/api/agent/spreadsheet/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              user_id: userId,
              instance_id: instanceId,
            }),
          });

          if (response.ok) {
            const result = await response.json();
            if (result.success && result.found) {
              // Only update if data has actually changed (avoid unnecessary re-renders)
              const currentDataStr = JSON.stringify(data);
              const newDataStr = JSON.stringify(result.data || {});
              
              if (currentDataStr !== newDataStr) {
                setData(result.data || {});
                // Also update columns/row_count if they changed
                if (result.columns && JSON.stringify(result.columns) !== JSON.stringify(columns)) {
                  setColumns(result.columns);
                }
                if (result.row_count && result.row_count !== rowCount) {
                  setRowCount(result.row_count);
                }
              }
            }
          }
        } catch (error) {
          // Silently fail - don't spam console with polling errors
          console.debug('Polling for spreadsheet updates:', error);
        }
      }, 2000); // Poll every 2 seconds

      return () => clearInterval(pollInterval);
    }, [instanceId, userId, data, columns, rowCount]);

    // Auto-save spreadsheet data when it changes (debounced)
    useEffect(() => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }

      // Only save if we have data and instanceId
      if (instanceId && Object.keys(data).length > 0) {
        saveTimeoutRef.current = setTimeout(() => {
          saveSpreadsheetData();
        }, 2000); // Debounce: save 2 seconds after last change
      }

      return () => {
        if (saveTimeoutRef.current) {
          clearTimeout(saveTimeoutRef.current);
        }
      };
    }, [data, columns, rowCount, instanceId, saveSpreadsheetData]);

    // Calculate cell value (simple formula support)
    const calculateCell = useCallback((cell: Cell, allData: SpreadsheetData): number | null => {
      if (!cell.formula) {
        const num = parseFloat(cell.value);
        return isNaN(num) ? null : num;
      }

      try {
        // Simple formula parsing: SUM(A1:A5), AVG(B1:B3), or direct math
        let formula = cell.formula.trim();
        
        // Handle SUM(range)
        if (formula.startsWith('SUM(') && formula.endsWith(')')) {
          const range = formula.slice(4, -1);
          const [start, end] = range.split(':');
          if (start && end) {
            const values = getRangeValues(start, end, allData);
            return values.reduce((sum: number, val) => (sum || 0) + (val || 0), 0);
          }
        }
        
        // Handle AVG(range)
        if (formula.startsWith('AVG(') && formula.endsWith(')')) {
          const range = formula.slice(4, -1);
          const [start, end] = range.split(':');
          if (start && end) {
            const values = getRangeValues(start, end, allData);
            const validValues = values.filter(v => v !== null) as number[];
            if (validValues.length === 0) return null;
            return validValues.reduce((sum, val) => sum + val, 0) / validValues.length;
          }
        }

        // Handle direct math expressions (simple)
        // Replace cell references like A1, B2, AA1, etc. with their values
        const cellRefRegex = /([A-Z]+\d+)/g;
        let processedFormula = formula;
        processedFormula = processedFormula.replace(cellRefRegex, (match) => {
          const cell = allData[match];
          if (cell) {
            const computed = calculateCell(cell, allData);
            return computed !== null ? computed.toString() : '0';
          }
          return '0';
        });

        // Evaluate simple math (only safe operations)
        // eslint-disable-next-line no-eval
        const result = eval(processedFormula);
        return typeof result === 'number' ? result : null;
      } catch {
        return null;
      }
    }, []);

    // Get values from a range
    const getRangeValues = (start: string, end: string, allData: SpreadsheetData): (number | null)[] => {
      const values: (number | null)[] = [];
      const startCol = start.match(/^([A-Z]+)/)?.[1] || '';
      const startRow = parseInt(start.match(/\d+$/)?.[0] || '0');
      const endCol = end.match(/^([A-Z]+)/)?.[1] || '';
      const endRow = parseInt(end.match(/\d+$/)?.[0] || '0');

      const startColIdx = columns.indexOf(startCol);
      const endColIdx = columns.indexOf(endCol);

      if (startColIdx === -1 || endColIdx === -1) return values;

      for (let row = startRow; row <= endRow; row++) {
        for (let colIdx = startColIdx; colIdx <= endColIdx; colIdx++) {
          const cellId = `${columns[colIdx]}${row}`;
          const cell = allData[cellId];
          if (cell) {
            const computed = calculateCell(cell, allData);
            values.push(computed);
          }
        }
      }
      return values;
    };

    // Add column
    const addColumn = () => {
      const newCol = getColumnLetter(columns.length);
      setColumns(prev => [...prev, newCol]);
      
      // Initialize cells for new column
      setData(prev => {
        const newData = { ...prev };
        for (let row = 1; row <= rowCount; row++) {
          const cellId = `${newCol}${row}`;
          if (!newData[cellId]) {
            newData[cellId] = {
              id: cellId,
              value: '',
            };
          }
        }
        return newData;
      });
    };

    // Remove column
    const removeColumn = () => {
      if (columns.length <= 1) return;
      
      const lastCol = columns[columns.length - 1];
      setColumns(prev => prev.slice(0, -1));
      
      // Remove cells for deleted column
      setData(prev => {
        const newData = { ...prev };
        for (let row = 1; row <= rowCount; row++) {
          const cellId = `${lastCol}${row}`;
          delete newData[cellId];
        }
        return newData;
      });
    };

    // Add row
    const addRow = () => {
      setRowCount(prev => prev + 1);
      
      // Initialize cells for new row
      setData(prev => {
        const newData = { ...prev };
        const newRow = rowCount + 1;
        for (const col of columns) {
          const cellId = `${col}${newRow}`;
          if (!newData[cellId]) {
            newData[cellId] = {
              id: cellId,
              value: '',
            };
          }
        }
        return newData;
      });
    };

    // Remove row
    const removeRow = () => {
      if (rowCount <= 1) return;
      
      const lastRow = rowCount;
      setRowCount(prev => prev - 1);
      
      // Remove cells for deleted row
      setData(prev => {
        const newData = { ...prev };
        for (const col of columns) {
          const cellId = `${col}${lastRow}`;
          delete newData[cellId];
        }
        return newData;
      });
    };

    // Fetch documents from cyrex
    const fetchDocuments = async () => {
      setIsLoadingDocuments(true);
      try {
        const response = await fetch(`${API_BASE}/api/v1/documents`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });
        
        if (response.ok) {
          const data = await response.json();
          setDocuments(data.documents || []);
        } else {
          console.error('Failed to fetch documents');
        }
      } catch (error) {
        console.error('Error fetching documents:', error);
      } finally {
        setIsLoadingDocuments(false);
      }
    };

    // Handle file upload and parsing (supports single or batch)
    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files;
      if (!files || files.length === 0) return;
      
      // If multiple files, use batch upload
      if (files.length > 1) {
        await handleBatchUpload(files);
        return;
      }
      
      // Single file upload
      const file = files[0];

      setIsImporting(true);
      setUploadProgress(0);
      setParsingStatus('Uploading file...');

      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('instance_id', instanceId || '');
        formData.append('use_ocr', 'true');
        formData.append('extract_tables', 'true');
        formData.append('start_cell', 'A1');

        setParsingStatus('Parsing document...');
        setUploadProgress(50);

        const response = await fetch(`${API_BASE}/api/agent/spreadsheet/parse-document`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Failed to parse document' }));
          throw new Error(errorData.detail || 'Failed to parse document');
        }

        setUploadProgress(80);
        setParsingStatus('Analyzing document...');

        const result = await response.json();

        if (!result.success) {
          throw new Error(result.error || 'Failed to parse document');
        }

        // Show preview instead of importing directly
        setParsedPreview(result);
        setShowPreview(true);
        setUploadProgress(100);
        setParsingStatus('Preview ready');
      } catch (error) {
        console.error('Error uploading and parsing document:', error);
        alert(`Failed to import document: ${error instanceof Error ? error.message : 'Unknown error'}`);
        setParsingStatus('');
        setUploadProgress(0);
      } finally {
        setIsImporting(false);
        // Reset file input
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    };

    // Handle batch file upload
    const handleBatchUpload = async (files: FileList) => {
      setIsImporting(true);
      setUploadProgress(0);
      setParsingStatus(`Processing ${files.length} files...`);

      try {
        const formData = new FormData();
        Array.from(files).forEach(file => {
          formData.append('files', file);
        });
        formData.append('instance_id', instanceId || '');
        formData.append('user_id', userId);
        formData.append('use_ocr', 'true');
        formData.append('extract_tables', 'true');
        formData.append('detect_layout', 'true');
        formData.append('start_cell', 'A1');

        setParsingStatus('Parsing documents...');
        setUploadProgress(30);

        const response = await fetch(`${API_BASE}/api/agent/spreadsheet/parse-document-batch`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Failed to parse documents' }));
          throw new Error(errorData.detail || 'Failed to parse documents');
        }

        setUploadProgress(80);
        setParsingStatus('Importing results...');

        const result = await response.json();

        if (!result.success) {
          throw new Error(result.error || 'Batch processing failed');
        }

        // Show batch results summary
        const successful = result.successful || 0;
        const failed = result.failed || 0;
        const message = `Batch processing complete!\n\nSuccessful: ${successful}\nFailed: ${failed}`;
        
        if (failed > 0) {
          const failedFiles = result.results
            .filter((r: any) => !r.success)
            .map((r: any) => `- ${r.filename}: ${r.error || 'Unknown error'}`)
            .join('\n');
          alert(`${message}\n\nFailed files:\n${failedFiles}`);
        } else {
          alert(message);
        }

        // Import first successful result (or could import all)
        const firstSuccess = result.results.find((r: any) => r.success);
        if (firstSuccess) {
          setParsedPreview({
            ...firstSuccess,
            success: true,
            filename: firstSuccess.filename,
          });
          setShowPreview(true);
        }

        setUploadProgress(100);
        setParsingStatus('Batch processing complete!');
        setTimeout(() => {
          setParsingStatus('');
          setUploadProgress(0);
        }, 2000);

      } catch (error) {
        console.error('Error in batch upload:', error);
        alert(`Failed to process batch: ${error instanceof Error ? error.message : 'Unknown error'}`);
        setParsingStatus('');
        setUploadProgress(0);
      } finally {
        setIsImporting(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    };

    // Confirm and import previewed document
    const confirmImport = () => {
      if (!parsedPreview) return;

      const mapping = parsedPreview.spreadsheet_mapping;
      const data = mapping.data || [];

      if (data.length > 0) {
        // Ensure we have enough columns and rows
        const maxCols = Math.max(...data.map((row: string[]) => row.length), columns.length);
        const maxRows = Math.max(data.length, rowCount);

        // Add columns if needed
        if (maxCols > columns.length) {
          const newCols: string[] = [];
          for (let i = columns.length; i < maxCols; i++) {
            newCols.push(getColumnLetter(i));
          }
          setColumns(prev => [...prev, ...newCols]);
        }

        // Add rows if needed
        if (maxRows > rowCount) {
          setRowCount(maxRows);
        }

        // Populate cells
        setData(prev => {
          const newData = { ...prev };
          data.forEach((row: string[], rowIdx: number) => {
            row.forEach((cellValue: string, colIdx: number) => {
              if (colIdx < columns.length + (maxCols - columns.length)) {
                const col = colIdx < columns.length 
                  ? columns[colIdx] 
                  : getColumnLetter(colIdx);
                const cellId = `${col}${rowIdx + 1}`;
                newData[cellId] = {
                  ...newData[cellId],
                  id: cellId,
                  value: cellValue || '',
                };
                setUpdateEffects(prevEffects => new Set(prevEffects).add(cellId));
              }
            });
          });
          return newData;
        });

        // Show warnings if any
        if (parsedPreview.warnings && parsedPreview.warnings.length > 0) {
          alert(`Document imported with warnings:\n${parsedPreview.warnings.join('\n')}`);
        }

        // Close preview and reset
        setShowPreview(false);
        setParsedPreview(null);
        setShowDocumentPicker(false);
        setParsingStatus('');
        setUploadProgress(0);
      }
    };

    // Cancel preview
    const cancelPreview = () => {
      setShowPreview(false);
      setParsedPreview(null);
      setParsingStatus('');
      setUploadProgress(0);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    };

    // Import document data into spreadsheet
    const importDocument = async (documentId: string) => {
      setIsImporting(true);
      try {
        // First get document metadata
        const docResponse = await fetch(`${API_BASE}/api/v1/documents/${documentId}`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!docResponse.ok) {
          throw new Error('Failed to fetch document');
        }

        const docData = await docResponse.json();
        const document = docData.document;

      // Try to extract text from document
      // For now, we'll use the document indexing service to get chunks
      // In a real implementation, you might want to extract structured data
      
        // Try to get document content via extraction API or search
        // First try to extract text directly if we have a document URL
        let chunks: any[] = [];
        
        if (document.document_url || document.file_path) {
          // Try document extraction API
          try {
            const extractResponse = await fetch(`${API_BASE}/document-extraction/extract-text`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                documentUrl: document.document_url || document.file_path,
                documentType: document.doc_type || 'pdf',
              }),
            });

            if (extractResponse.ok) {
              const extractData = await extractResponse.json();
              if (extractData.text) {
                // Split text into chunks for parsing
                const text = extractData.text;
                const lines = text.split('\n').filter((l: string) => l.trim());
                chunks = lines.map((line: string, idx: number) => ({
                  content: line,
                  text: line,
                  index: idx,
                }));
              }
            }
          } catch (extractError) {
            console.warn('Document extraction failed, trying search API:', extractError);
          }

          // Fallback to search API if extraction didn't work
          if (chunks.length === 0) {
            try {
              const searchResponse = await fetch(`${API_BASE}/api/v1/documents/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  query: '*',
                  document_ids: [documentId],
                  limit: 1000,
                }),
              });

              if (searchResponse.ok) {
                const searchData = await searchResponse.json();
                chunks = searchData.results || searchData.chunks || [];
              }
            } catch (searchError) {
              console.warn('Document search failed:', searchError);
            }
          }

          if (chunks.length > 0) {
            const firstChunk = chunks[0].content || chunks[0].text || '';
            
            // Try CSV parsing
            if (firstChunk.includes(',') || firstChunk.includes('\t')) {
              parseCSVData(chunks, firstChunk);
            } else {
              // Parse as structured text
              parseStructuredData(chunks);
            }
          } else {
            throw new Error('No content found in document');
          }
        } else {
          throw new Error('Document has no URL or file path');
        }
      } catch (error) {
        console.error('Error importing document:', error);
        alert(`Failed to import document: ${error instanceof Error ? error.message : 'Unknown error'}`);
      } finally {
        setIsImporting(false);
        setShowDocumentPicker(false);
      }
    };

    // Parse CSV-like data
    const parseCSVData = (chunks: any[], sample: string) => {
      const allLines: string[] = [];
      
      // Collect all text from chunks
      chunks.forEach(chunk => {
        const content = chunk.content || chunk.text || '';
        const lines = content.split('\n');
        allLines.push(...lines);
      });

      // Parse CSV
      const rows: string[][] = [];
      allLines.forEach(line => {
        if (line.trim()) {
          // Try comma first, then tab
          const delimiter = line.includes('\t') ? '\t' : ',';
          const cells = line.split(delimiter).map(c => c.trim());
          if (cells.length > 0 && cells.some(c => c)) {
            rows.push(cells);
          }
        }
      });

      // Populate spreadsheet starting from A1
      setData(prev => {
        const newData = { ...prev };
        rows.forEach((row, rowIdx) => {
          row.forEach((cellValue, colIdx) => {
            if (colIdx < columns.length && rowIdx < rowCount) {
              const cellId = `${columns[colIdx]}${rowIdx + 1}`;
              newData[cellId] = {
                ...newData[cellId],
                value: cellValue,
              };
              // Trigger update effect
              setUpdateEffects(prevEffects => new Set(prevEffects).add(cellId));
            }
          });
        });
        return newData;
      });

      // Add more columns/rows if needed
      const maxCols = Math.max(...rows.map(r => r.length), columns.length);
      const maxRows = Math.max(rows.length, rowCount);
      
      if (maxCols > columns.length) {
        const newCols: string[] = [];
        for (let i = columns.length; i < maxCols; i++) {
          newCols.push(getColumnLetter(i));
        }
        setColumns(prev => [...prev, ...newCols]);
      }
      
      if (maxRows > rowCount) {
        setRowCount(maxRows);
      }
    };

    // Parse structured text data
    const parseStructuredData = (chunks: any[]) => {
      let row = 1;
      let col = 0;

      chunks.forEach((chunk, idx) => {
        const content = chunk.content || chunk.text || '';
        const lines = content.split('\n').filter((l: string) => l.trim());

        lines.forEach((line: string) => {
          if (col >= columns.length) {
            addColumn();
          }
          if (row > rowCount) {
            addRow();
          }

          const cellId = `${columns[col]}${row}`;
          updateCell(cellId, line.trim(), 'Document Import');
          
          col++;
          if (col >= 10) { // Wrap to next row after 10 columns
            col = 0;
            row++;
          }
        });
      });
    };

    // Update cell value
    const updateCell = useCallback((cellId: string, value: string, updatingAgent?: string) => {
      setData(prev => {
        const newData = { ...prev };
        const cell = { ...newData[cellId] };
        
        // Check if it's a formula (starts with =)
        if (value.startsWith('=')) {
          cell.formula = value.slice(1);
          cell.value = '';
        } else {
          cell.value = value;
          cell.formula = undefined;
        }

        // Calculate computed value
        const computed = calculateCell(cell, newData);
        cell.computedValue = computed !== null ? computed : undefined;

        // Track who updated it
        if (updatingAgent) {
          cell.lastUpdatedBy = updatingAgent;
          cell.lastUpdatedAt = new Date().toISOString();
          
          // Add to activity log
          setAgentActivity(prev => [
            { cellId, agentName: updatingAgent, timestamp: new Date().toISOString() },
            ...prev.slice(0, 49) // Keep last 50
          ]);
        }

        newData[cellId] = cell;
        return newData;
      });

      // Add visual effect
      setUpdateEffects(prev => new Set(prev).add(cellId));
      setTimeout(() => {
        setUpdateEffects(prev => {
          const next = new Set(prev);
          next.delete(cellId);
          return next;
        });
      }, 1000);

      // Notify parent
      if (onCellUpdate) {
        onCellUpdate(cellId, value, updatingAgent);
      }
    }, [calculateCell, onCellUpdate]);

    // Handle cell click
    const handleCellClick = (cellId: string) => {
      setSelectedCell(cellId);
      setEditingCell(cellId);
      const cell = data[cellId];
      setEditValue(cell?.formula ? `=${cell.formula}` : cell?.value || '');
    };

    // Handle cell edit
    const handleCellEdit = (cellId: string, value: string) => {
      updateCell(cellId, value, agentName);
      setEditingCell(null);
      setSelectedCell(null);
    };

    // Handle key press in cell
    const handleCellKeyPress = (e: React.KeyboardEvent, cellId: string) => {
      if (e.key === 'Enter') {
        handleCellEdit(cellId, editValue);
      } else if (e.key === 'Escape') {
        setEditingCell(null);
        setSelectedCell(null);
      }
    };

    // Direct method to set cell value (used by tool results)
    const setCellDirect = useCallback((cellId: string, value: string, updatingAgent?: string) => {
      updateCell(cellId.toUpperCase(), value, updatingAgent || agentName || 'Agent');
    }, [updateCell, agentName]);

    // Process tool result to update spreadsheet
    const processToolResult = useCallback((toolName: string, params: Record<string, unknown>, result?: unknown) => {
      if (toolName === 'spreadsheet_set_cell') {
        // Handle both direct params and result object
        let cellId = String(params.cell_id || '');
        let value = String(params.value || '');
        
        // If result is provided and contains the data, use it
        if (result) {
          let resultObj: Record<string, unknown> = {};
          if (typeof result === 'string') {
            try {
              resultObj = JSON.parse(result);
            } catch {
              // Not JSON, treat as plain string
            }
          } else if (typeof result === 'object' && result !== null) {
            resultObj = result as Record<string, unknown>;
          }
          
          // Extract from result if params are missing
          if (!cellId && resultObj.cell_id) {
            cellId = String(resultObj.cell_id);
          }
          if (!value && resultObj.value) {
            value = String(resultObj.value);
          }
        }
        
        if (cellId && value) {
          setCellDirect(cellId, value, agentName || 'Agent');
        }
      } else if (toolName === 'spreadsheet_sum_range') {
        const startCell = String(params.start_cell || '');
        const endCell = String(params.end_cell || '');
        const targetCell = params.target_cell ? String(params.target_cell) : null;
        
        if (startCell && endCell) {
          // If we have a result with the sum, use it
          if (result && typeof result === 'object' && result !== null) {
            const resultObj = result as Record<string, unknown>;
            const sum = resultObj.sum;
            if (targetCell && sum !== undefined) {
              setCellDirect(targetCell, String(sum), agentName || 'Agent');
            } else if (targetCell) {
              // Set formula if no direct sum value
              setCellDirect(targetCell, `=SUM(${startCell}:${endCell})`, agentName || 'Agent');
            }
          } else if (targetCell) {
            // Set formula as fallback
            setCellDirect(targetCell, `=SUM(${startCell}:${endCell})`, agentName || 'Agent');
          }
        }
      } else if (toolName === 'spreadsheet_avg_range') {
        const startCell = String(params.start_cell || '');
        const endCell = String(params.end_cell || '');
        const targetCell = params.target_cell ? String(params.target_cell) : null;
        
        if (startCell && endCell) {
          // If we have a result with the average, use it
          if (result && typeof result === 'object' && result !== null) {
            const resultObj = result as Record<string, unknown>;
            const avg = resultObj.average || resultObj.avg;
            if (targetCell && avg !== undefined) {
              setCellDirect(targetCell, String(avg), agentName || 'Agent');
            } else if (targetCell) {
              // Set formula if no direct average value
              setCellDirect(targetCell, `=AVG(${startCell}:${endCell})`, agentName || 'Agent');
            }
          } else if (targetCell) {
            // Set formula as fallback
            setCellDirect(targetCell, `=AVG(${startCell}:${endCell})`, agentName || 'Agent');
          }
        }
      }
    }, [setCellDirect, agentName]);

    // Process agent message to update spreadsheet
    const processAgentMessage = useCallback((message: string) => {
      // Pattern: "set [cell] to [value]" or "put [value] in [cell]"
      // Updated to match all column letters (A-Z, AA-ZZ, etc.)
      const setPatterns = [
        /set\s+([A-Z]+\d+)\s+to\s+([\d.]+)/i,
        /put\s+([\d.]+)\s+in\s+([A-Z]+\d+)/i,
        /([A-Z]+\d+)\s*=\s*([\d.]+)/i
      ];
      
      for (const pattern of setPatterns) {
        const match = message.match(pattern);
        if (match) {
          const cellId = (match[1] || match[2]).toUpperCase();
          const value = match[2] || match[1];
          if (cellId && value) {
            updateCell(cellId, value, agentName || 'Agent');
            return;
          }
        }
      }

      // Pattern: "add [value] to [cell]"
      const addPattern = /add\s+([\d.]+)\s+to\s+([A-Z]+\d+)/i;
      const addMatch = message.match(addPattern);
      if (addMatch) {
        const [, value, cellId] = addMatch;
        setData(currentData => {
          const cell = currentData[cellId.toUpperCase()];
          const currentValue = cell?.computedValue !== undefined 
            ? cell.computedValue 
            : (parseFloat(cell?.value || '0') || 0);
          const newValue = currentValue + parseFloat(value);
          updateCell(cellId.toUpperCase(), newValue.toString(), agentName || 'Agent');
          return currentData; // Return unchanged, updateCell will update
        });
        return;
      }

      // Pattern: "calculate sum of [range] in [cell]" or "sum [range] in [cell]"
      const sumPatterns = [
        /(?:calculate\s+)?sum\s+([A-Z]+\d+:[A-Z]+\d+)\s+in\s+([A-Z]+\d+)/i,
        /sum\s+([A-Z]+\d+:[A-Z]+\d+)\s+to\s+([A-Z]+\d+)/i
      ];
      
      for (const pattern of sumPatterns) {
        const sumMatch = message.match(pattern);
        if (sumMatch) {
          const range = (sumMatch[1] || sumMatch[2]).toUpperCase();
          const targetCell = (sumMatch[2] || sumMatch[3]).toUpperCase();
          if (range && targetCell) {
            const [start, end] = range.split(':');
            if (start && end) {
              updateCell(targetCell, `=SUM(${range})`, agentName || 'Agent');
              return;
            }
          }
        }
      }
    }, [agentName, updateCell]);

    // Expose methods to parent via ref
    useImperativeHandle(ref, () => ({
      processMessage: (message: string) => {
        processAgentMessage(message);
      },
      setCell: (cellId: string, value: string, agentName?: string) => {
        setCellDirect(cellId, value, agentName);
      },
      setCellFromTool: (toolName: string, params: Record<string, unknown>, result?: unknown) => {
        processToolResult(toolName, params, result);
      }
    }), [processAgentMessage, setCellDirect, processToolResult]);

    // Listen for agent messages from parent
    useEffect(() => {
      if (onAgentMessage) {
        // Parent can trigger this via the ref
      }
    }, [onAgentMessage]);

    // Get display value for cell
    const getCellDisplayValue = (cell: Cell): string => {
      if (cell.computedValue !== undefined) {
        return cell.computedValue.toFixed(2);
      }
      return cell.value || '';
    };

    // Clear spreadsheet
    const clearSpreadsheet = () => {
      const clearedData: SpreadsheetData = {};
      for (let row = 1; row <= rowCount; row++) {
        for (const col of columns) {
          const cellId = `${col}${row}`;
          clearedData[cellId] = {
            id: cellId,
            value: '',
          };
        }
      }
      setData(clearedData);
      setAgentActivity([]);
    };

    return (
      <div className="live-spreadsheet" ref={spreadsheetRef}>
      <div className="spreadsheet-header">
        <div className="spreadsheet-title">
          <FaTable /> Live Spreadsheet
          {instanceId && <span className="instance-id">Instance: {instanceId.slice(0, 8)}</span>}
        </div>
        <div className="spreadsheet-actions">
          <button 
            className="btn-spreadsheet" 
            onClick={() => {
              setShowDocumentPicker(true);
              fetchDocuments();
            }} 
            title="Import document"
          >
            <FaFileImport /> Import Document
          </button>
          <button className="btn-spreadsheet" onClick={addColumn} title="Add column">
            <FaPlus /> Add Column
          </button>
          <button className="btn-spreadsheet" onClick={removeColumn} title="Remove column" disabled={columns.length <= 1}>
            <FaTimes /> Remove Column
          </button>
          <button className="btn-spreadsheet" onClick={addRow} title="Add row">
            <FaPlus /> Add Row
          </button>
          <button className="btn-spreadsheet" onClick={removeRow} title="Remove row" disabled={rowCount <= 1}>
            <FaTimes /> Remove Row
          </button>
          <button className="btn-spreadsheet" onClick={clearSpreadsheet} title="Clear all">
            <FaTrash /> Clear
          </button>
        </div>
      </div>

      {showDocumentPicker && (
        <div className="document-picker-overlay" onClick={() => setShowDocumentPicker(false)}>
          <div className="document-picker" onClick={e => e.stopPropagation()}>
            <div className="document-picker-header">
              <h3>Import Document</h3>
              <button className="btn-close" onClick={() => setShowDocumentPicker(false)}>
                <FaTimes />
              </button>
            </div>
            <div className="document-picker-content">
              {/* File Upload Section */}
              {!showPreview && (
              <div className="file-upload-section">
                <h4>Upload New Document</h4>
                <p className="upload-hint">
                  Supported formats: PDF, Word (DOCX), Excel (XLSX), CSV, Text, Images (PNG, JPG, TIFF)
                </p>
                <div className="file-upload-area">
                  <input
                    ref={fileInputRef}
                    type="file"
                    id="document-upload"
                    accept=".pdf,.docx,.xlsx,.csv,.txt,.md,.png,.jpg,.jpeg,.tiff,.html"
                    onChange={handleFileUpload}
                    multiple={true}
                    style={{ display: 'none' }}
                    disabled={isImporting}
                  />
                  <label 
                    htmlFor="document-upload" 
                    className={`file-upload-label ${isImporting ? 'disabled' : ''}`}
                  >
                    <FaFileImport size={32} />
                    <span>Click to upload or drag and drop</span>
                    <span className="file-size-hint">Max file size: 50MB per file</span>
                    <span className="file-size-hint" style={{ fontSize: '0.7rem', marginTop: '0.25rem' }}>
                      Multiple files supported (batch processing)
                    </span>
                  </label>
                </div>
                {isImporting && (
                  <div className="upload-progress">
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <div className="progress-text">{parsingStatus}</div>
                  </div>
                )}
              </div>
              )}

              {/* Preview Section */}
              {showPreview && parsedPreview && (
                <div className="preview-section">
                  <h4>Document Preview</h4>
                  <div className="preview-content">
                    <div className="preview-info">
                      <div className="info-row">
                        <span className="info-label">Document Type:</span>
                        <span className="info-value">{parsedPreview.document_type}</span>
                      </div>
                      {parsedPreview.document_category && (
                        <div className="info-row">
                          <span className="info-label">Category:</span>
                          <span className="info-value">{parsedPreview.document_category}</span>
                        </div>
                      )}
                      {parsedPreview.confidence_scores && (
                        <div className="info-row">
                          <span className="info-label">Confidence:</span>
                          <span className="info-value">
                            {(parsedPreview.confidence_scores.overall * 100).toFixed(0)}%
                          </span>
                        </div>
                      )}
                      {parsedPreview.spreadsheet_mapping && (
                        <div className="info-row">
                          <span className="info-label">Rows:</span>
                          <span className="info-value">{parsedPreview.spreadsheet_mapping.row_count}</span>
                          <span className="info-label" style={{ marginLeft: '1rem' }}>Columns:</span>
                          <span className="info-value">{parsedPreview.spreadsheet_mapping.column_count}</span>
                        </div>
                      )}
                    </div>

                    {parsedPreview.parsed_data?.key_value_pairs && 
                     Object.keys(parsedPreview.parsed_data.key_value_pairs).length > 0 && (
                      <div className="preview-kvp">
                        <h5>Extracted Information:</h5>
                        <div className="kvp-grid">
                          {Object.entries(parsedPreview.parsed_data.key_value_pairs).map(([key, value]) => (
                            <div key={key} className="kvp-item">
                              <span className="kvp-key">{key.replace(/_/g, ' ')}:</span>
                              <span className="kvp-value">{String(value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {parsedPreview.parsed_data?.tables && parsedPreview.parsed_data.tables.length > 0 && (
                      <div className="preview-tables">
                        <h5>Tables Found: {parsedPreview.parsed_data.tables.length}</h5>
                        {parsedPreview.parsed_data.tables.slice(0, 2).map((table: string[][], idx: number) => (
                          <div key={idx} className="preview-table">
                            <table>
                              <tbody>
                                {table.slice(0, 5).map((row: string[], rowIdx: number) => (
                                  <tr key={rowIdx}>
                                    {row.map((cell: string, cellIdx: number) => (
                                      <td key={cellIdx}>{cell}</td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            {table.length > 5 && <div className="table-more">... and {table.length - 5} more rows</div>}
                          </div>
                        ))}
                      </div>
                    )}

                    {parsedPreview.warnings && parsedPreview.warnings.length > 0 && (
                      <div className="preview-warnings">
                        <h5>⚠️ Warnings:</h5>
                        <ul>
                          {parsedPreview.warnings.map((warning: string, idx: number) => (
                            <li key={idx}>{warning}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="preview-actions">
                      <button className="btn-primary" onClick={confirmImport}>
                        Import to Spreadsheet
                      </button>
                      <button className="btn-secondary" onClick={cancelPreview}>
                        Cancel
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Existing Documents Section */}
              {!showPreview && (
                <div className="existing-documents-section">
                  <h4>Or Select Existing Document</h4>
                {isLoadingDocuments ? (
                  <div className="loading-documents">
                    <FaSync className="spin" /> Loading documents...
                  </div>
                ) : documents.length === 0 ? (
                  <div className="no-documents">
                    <p>No indexed documents found. Upload a new document above.</p>
                  </div>
                ) : (
                  <div className="document-list">
                    {documents.map(doc => (
                      <div 
                        key={doc.document_id} 
                        className="document-item"
                        onClick={() => !isImporting && importDocument(doc.document_id)}
                        style={{ opacity: isImporting ? 0.5 : 1, cursor: isImporting ? 'not-allowed' : 'pointer' }}
                      >
                        <div className="document-title">{doc.title || 'Untitled'}</div>
                        <div className="document-meta">
                          <span className="document-type">{doc.doc_type || 'other'}</span>
                          {doc.created_at && (
                            <span className="document-date">
                              {new Date(doc.created_at).toLocaleDateString()}
                            </span>
                          )}
                        </div>
                        {isImporting && <div className="importing-overlay">Importing...</div>}
                      </div>
                    ))}
                  </div>
                )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {agentActivity.length > 0 && (
        <div className="agent-activity">
          <div className="activity-header">Recent Agent Activity</div>
          <div className="activity-list">
            {agentActivity.slice(0, 5).map((activity, idx) => (
              <div key={idx} className="activity-item">
                <span className="activity-agent">{activity.agentName}</span>
                <span className="activity-cell">{activity.cellId}</span>
                <span className="activity-time">
                  {new Date(activity.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="spreadsheet-container">
        <div className="spreadsheet-grid">
          {/* Header row */}
          <div className="spreadsheet-row header-row">
            <div className="spreadsheet-cell header-cell"></div>
            {columns.map(col => (
              <div key={col} className="spreadsheet-cell header-cell">
                {col}
              </div>
            ))}
          </div>

          {/* Data rows */}
          {Array.from({ length: rowCount }, (_, rowIdx) => {
            const rowNum = rowIdx + 1;
            return (
              <div key={rowNum} className="spreadsheet-row">
                <div className="spreadsheet-cell row-header">{rowNum}</div>
                {columns.map(col => {
                  const cellId = `${col}${rowNum}`;
                  const cell = data[cellId];
                  const isSelected = selectedCell === cellId;
                  const isEditing = editingCell === cellId;
                  const hasEffect = updateEffects.has(cellId);
                  const displayValue = cell ? getCellDisplayValue(cell) : '';

                  return (
                    <div
                      key={cellId}
                      className={`spreadsheet-cell data-cell ${isSelected ? 'selected' : ''} ${hasEffect ? 'updated' : ''}`}
                      onClick={() => handleCellClick(cellId)}
                    >
                      {isEditing ? (
                        <input
                          ref={el => cellRefs.current[cellId] = el}
                          type="text"
                          value={editValue}
                          onChange={e => setEditValue(e.target.value)}
                          onBlur={() => handleCellEdit(cellId, editValue)}
                          onKeyDown={e => handleCellKeyPress(e, cellId)}
                          className="cell-input"
                          autoFocus
                        />
                      ) : (
                        <>
                          <span className="cell-value">{displayValue}</span>
                          {cell?.lastUpdatedBy && (
                            <span className="cell-agent-badge" title={`Updated by ${cell.lastUpdatedBy}`}>
                              {cell.lastUpdatedBy.slice(0, 1)}
                            </span>
                          )}
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      <div className="spreadsheet-footer">
        <div className="footer-info">
          <FaCalculator /> Click any cell to edit. Use formulas like =SUM(A1:A5) or =AVG(B1:B3)
        </div>
        <div className="footer-stats">
          {Object.values(data).filter(c => c.value || c.formula).length} cells filled
        </div>
      </div>
      </div>
    );
  }
);

LiveSpreadsheet.displayName = 'LiveSpreadsheet';

