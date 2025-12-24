import React, { useCallback, useMemo, useState, useEffect } from 'react';
import './App.css';
import { Sidebar } from './components/layout/Sidebar';
import { useUI } from './context/UIContext';

type ChatMessage = {
  role: 'user' | 'assistant' | 'system';
  content: string;
  meta?: Record<string, unknown>;
  timestamp?: string;
};

type TestResult = {
  endpoint: string;
  request: unknown;
  response?: React.ReactNode | null;
  duration: number;
  timestamp: string;
  success: boolean;
  error?: string;
};

type WorkflowState = {
  workflow_id: string;
  status: string;
  current_step?: string;
  state_data: Record<string, unknown>;
  checkpoints: unknown[];
};

// Auto-detect if we're in browser and convert Docker hostnames to localhost
const getDefaultBaseUrl = () => {
  const envUrl = (import.meta.env?.VITE_CYREX_BASE_URL as string) || 'http://localhost:8000';
  // If running in browser (not in Docker) and URL uses Docker hostname, convert to localhost
  if (typeof window !== 'undefined' && envUrl.includes('cyrex:')) {
    return envUrl.replace('cyrex:', 'localhost:');
  }
  return envUrl;
};

const defaultBaseUrl = getDefaultBaseUrl();

// Switch Toggle Component
const Switch = ({ checked, onChange, label }: { checked: boolean; onChange: (checked: boolean) => void; label: string }) => {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer', userSelect: 'none' }}>
      <div
        onClick={() => onChange(!checked)}
        style={{
          position: 'relative',
          width: '44px',
          height: '24px',
          borderRadius: '12px',
          background: checked ? '#4CAF50' : '#555',
          transition: 'background 0.3s ease',
          cursor: 'pointer',
          flexShrink: 0
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: '2px',
            left: checked ? '22px' : '2px',
            width: '20px',
            height: '20px',
            borderRadius: '50%',
            background: '#fff',
            transition: 'left 0.3s ease',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)'
          }}
        />
      </div>
      <span style={{ fontSize: '0.9rem', color: '#e0e0e0' }}>{label}</span>
    </label>
  );
};

export default function App() {
  const { state: uiState } = useUI();
  const activeTab = uiState.activeTab;
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl);
  const [apiKey, setApiKey] = useState('');
  
  // Orchestration state
  const [orchestrationInput, setOrchestrationInput] = useState('What are my tasks for today?');
  const [orchestrationResult, setOrchestrationResult] = useState('');
  const [useRAG, setUseRAG] = useState(true);
  const [useTools, setUseTools] = useState(true);
  const [streamingResponse, setStreamingResponse] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  
  // Workflow state
  const [workflowId, setWorkflowId] = useState('');
  const [workflowSteps, setWorkflowSteps] = useState(JSON.stringify([
    { name: 'step1', tool: 'knowledge_retrieval', input: { query: 'user tasks' } },
    { name: 'step2', input: { action: 'generate_summary' } }
  ], null, 2));
  const [workflowResult, setWorkflowResult] = useState('');
  const [workflowStates, setWorkflowStates] = useState<WorkflowState[]>([]);
  
  // Local LLM state
  const [llmPrompt, setLlmPrompt] = useState('Explain how RAG works in simple terms.');
  const [llmResult, setLlmResult] = useState('');
  const [llmModel, setLlmModel] = useState('llama3:8b');
  const [llmBackend, setLlmBackend] = useState('ollama');
  
  // RAG/Vector Store state
  const [ragQuery, setRagQuery] = useState('productivity tips');
  const [ragResult, setRagResult] = useState('');
  const [ragCollection, setRagCollection] = useState('deepiri_knowledge');
  const [ragTopK, setRagTopK] = useState(5);
  const [documentContent, setDocumentContent] = useState('');
  const [documentMetadata, setDocumentMetadata] = useState('{"source": "test", "type": "document"}');
  
  // Tools state
  const [toolsList, setToolsList] = useState<unknown[]>([]);
  const [toolName, setToolName] = useState('');
  const [toolInput, setToolInput] = useState('{}');
  const [toolResult, setToolResult] = useState('');
  
  // State Management
  const [stateWorkflowId, setStateWorkflowId] = useState('');
  const [stateData, setStateData] = useState('{"key": "value"}');
  const [checkpointName, setCheckpointName] = useState('');
  const [stateResult, setStateResult] = useState('');
  
  // Monitoring state
  const [systemStatus, setSystemStatus] = useState<Record<string, unknown> | null>(null);
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);
  const [testHistory, setTestHistory] = useState<TestResult[]>([]);
  
  // Health state
  const [healthStatus, setHealthStatus] = useState<Record<string, any> | null>(null);
  
  // Safety/Guardrails state
  const [safetyInput, setSafetyInput] = useState('Generate a summary of my tasks');
  const [safetyResult, setSafetyResult] = useState('');
  
  // Chat state
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatProvider, setChatProvider] = useState<'api' | 'local'>('api');
  const [showProviderDropdown, setShowProviderDropdown] = useState(false);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (showProviderDropdown && !target.closest('[data-provider-dropdown]')) {
        setShowProviderDropdown(false);
      }
    };

    if (showProviderDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showProviderDropdown]);

  const [chatLocalLLMService, setChatLocalLLMService] = useState<string>('');
  const [chatLocalLLMModel, setChatLocalLLMModel] = useState<string>('llama3:8b');
  const [chatLocalLLMBackend, setChatLocalLLMBackend] = useState<string>('ollama');
  const [chatMaxTokens, setChatMaxTokens] = useState<number>(200);
  const [availableLLMServices, setAvailableLLMServices] = useState<Array<{name: string; type: string; base_url: string; models: string[]}>>([]);
  const [scanningLLMServices, setScanningLLMServices] = useState(false);

  // Testing state
  const [testCategories, setTestCategories] = useState<Record<string, any>>({});
  const [testFiles, setTestFiles] = useState<Record<string, string>>({});
  const [selectedTestCategory, setSelectedTestCategory] = useState<string>('');
  const [selectedTestFile, setSelectedTestFile] = useState<string>('');
  const [selectedTestPath, setSelectedTestPath] = useState<string>('');
  const [testVerbose, setTestVerbose] = useState(true);
  const [testCoverage, setTestCoverage] = useState(false);
  const [testSkipSlow, setTestSkipSlow] = useState(false);
  const [testTimeout, setTestTimeout] = useState(15);
  const [testOutput, setTestOutput] = useState<string[]>([]);
  const [testRunning, setTestRunning] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; return_code: number } | null>(null);
  const [testStatus, setTestStatus] = useState<Record<string, any> | null>(null);
  const [testSummary, setTestSummary] = useState<{
    passed: number;
    failed: number;
    warnings: number;
    duration: string;
    failures: Array<{ test: string; error: string }>;
  } | null>(null);

  // Debug state for each tab
  type DebugInfo = {
    request?: unknown;
    response?: unknown;
    logs?: string[];
    timestamp?: string;
    duration?: number;
    error?: string;
  };
  const [debugInfo, setDebugInfo] = useState<Record<string, DebugInfo>>({});
  const [showDebug, setShowDebug] = useState<Record<string, boolean>>({});
  
  // Abort controller for stopping tests
  const [testAbortController, setTestAbortController] = useState<AbortController | null>(null);

  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const headers = useMemo(() => {
    const result: Record<string, string> = {
      'Content-Type': 'application/json'
    };
    if (apiKey) {
      result['x-api-key'] = apiKey;
    }
    return result;
  }, [apiKey]);

  const callEndpoint = useCallback(
    async (path: string, payload: unknown, method = 'POST', tabId?: string) => {
      const startTime = Date.now();
      setLoading(path);
      setError(null);
      
      // Determine timeout based on endpoint type
      let timeoutMs = 5 * 60 * 1000; // Default 5 minutes
      
      // Discovery/scanning endpoints need more time
      if (path.includes('/llm-services') || path.includes('/discover') || path.includes('/scan')) {
        timeoutMs = 10 * 60 * 1000; // 10 minutes for discovery
      }
      // Chat/LLM generation endpoints
      else if (path.includes('/process') || path.includes('/generate') || path.includes('/chat')) {
        timeoutMs = 6 * 60 * 1000; // 6 minutes for LLM generation (allows for slow models)
      }
      // Status/health checks should be fast
      else if (path.includes('/status') || path.includes('/health') || path.includes('/list')) {
        timeoutMs = 30 * 1000; // 30 seconds for status checks
      }
      
      // Initialize debug info for this tab
      if (tabId) {
        setDebugInfo(prev => ({
          ...prev,
          [tabId]: {
            request: payload,
            logs: [`[${new Date().toISOString()}] Starting request to ${path}`],
            timestamp: new Date().toISOString()
          }
        }));
      }
      
      try {
        // Create AbortController for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
        
        const res = await fetch(`${baseUrl}${path}`, {
          method,
          headers,
          body: method !== 'GET' ? JSON.stringify(payload) : undefined,
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        const duration = Date.now() - startTime;
        const text = await res.text();
        
        if (!res.ok) {
          // Try to parse error message from response
          let errorDetail = text;
          try {
            const errorJson = JSON.parse(text);
            errorDetail = errorJson.detail || errorJson.error || errorJson.message || text;
          } catch {
            // If not JSON, use text as-is
          }
          throw new Error(`${res.status} ${res.statusText}: ${errorDetail}`);
        }
        
        const json = text ? JSON.parse(text) : {};
        
        // Update debug info
        if (tabId) {
          setDebugInfo(prev => ({
            ...prev,
            [tabId]: {
              ...prev[tabId],
              response: json,
              duration,
              logs: [
                ...(prev[tabId]?.logs || []),
                `[${new Date().toISOString()}] Response received (${res.status})`,
                `[${new Date().toISOString()}] Duration: ${duration}ms`
              ]
            }
          }));
        }
        
        // Record test result
        const testResult: TestResult = {
          endpoint: path,
          request: payload,
          response: pretty(json) as React.ReactNode,
          duration,
          timestamp: new Date().toISOString(),
          success: true
        };
        setTestHistory(prev => [testResult, ...prev].slice(0, 50)); // Keep last 50
        
        return json;
      } catch (err: any) {
        const duration = Date.now() - startTime;
        let errorMsg = err?.message ?? 'Unknown error';
        
        // Handle abort/timeout errors with context-aware messages
        if (err.name === 'AbortError' || errorMsg.includes('aborted')) {
          if (path.includes('/llm-services') || path.includes('/discover')) {
            errorMsg = 'Service discovery timed out after 10 minutes. Docker network scanning may be slow. Try again or check Docker network connectivity.';
          } else if (path.includes('/process') || path.includes('/generate')) {
            errorMsg = 'Request timed out after 6 minutes. The LLM may be slow. Try reducing max_tokens or using a faster model.';
          } else if (path.includes('/status') || path.includes('/health') || path.includes('/list')) {
            errorMsg = 'Request timed out after 30 seconds.';
          } else {
            errorMsg = 'Request timed out after 5 minutes.';
          }
        }
        
        setError(errorMsg);
        
        // Update debug info with error
        if (tabId) {
          setDebugInfo(prev => ({
            ...prev,
            [tabId]: {
              ...prev[tabId],
              error: errorMsg,
              duration,
              logs: [
                ...(prev[tabId]?.logs || []),
                `[${new Date().toISOString()}] ERROR: ${errorMsg}`
              ]
            }
          }));
        }
        
        // Record failed test
        const testResult: TestResult = {
          endpoint: path,
          request: payload,
          response: null,
          duration,
          timestamp: new Date().toISOString(),
          success: false,
          error: errorMsg
        };
        setTestHistory(prev => [testResult, ...prev].slice(0, 50));
        
        throw err;
      } finally {
        setLoading(null);
      }
    },
    [baseUrl, headers]
  );

  const pretty = (data: unknown) => JSON.stringify(data, null, 2);

  // Orchestration tests
  const testOrchestration = async () => {
    setStreamingResponse(''); // Clear streaming response when using regular process
    setOrchestrationResult(''); // Clear previous result
    try {
      const result = await callEndpoint('/orchestration/process', {
        user_input: orchestrationInput,
        user_id: 'test-user',
        use_rag: useRAG,
        use_tools: useTools
      }, 'POST', 'orchestration');
      const formattedResult = result ? pretty(result) : 'No response received';
      setOrchestrationResult(formattedResult);
    } catch (err: any) {
      setOrchestrationResult(`Error: ${err.message}`);
    }
  };

  const testStreaming = async () => {
    setIsStreaming(true);
    setOrchestrationResult(''); // Clear regular result when using streaming
    setStreamingResponse('');
    try {
      // Try the agent message stream endpoint
      const response = await fetch(`${baseUrl}/agent/message/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          message: orchestrationInput,
          user_id: 'test-user'
        })
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      if (!reader) throw new Error('No response body');
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        setStreamingResponse(prev => prev + chunk);
      }
    } catch (err: any) {
      setStreamingResponse(`Error: ${err.message}`);
    } finally {
      setIsStreaming(false);
    }
  };

  // Workflow tests
  const testWorkflow = async () => {
    try {
      const wfId = workflowId || `workflow_${Date.now()}`;
      const steps = JSON.parse(workflowSteps);
      
      const result = await callEndpoint('/orchestration/workflow', {
        workflow_id: wfId,
        steps,
        initial_state: {}
      }, 'POST', 'workflow');
      
      setWorkflowResult(pretty(result));
      setWorkflowId(wfId);
      await loadWorkflowStates();
    } catch (err: any) {
      setWorkflowResult(`Error: ${err.message}`);
    }
  };

  const loadWorkflowStates = async () => {
    try {
      // This would need a list endpoint, for now we'll try to get known workflows
      if (workflowId) {
        const state = await callEndpoint(`/orchestration/workflow/${workflowId}`, {}, 'GET');
        setWorkflowStates([state as WorkflowState]);
      }
    } catch (err) {
      // Ignore errors for now
    }
  };

  // Local LLM tests
  const testLocalLLM = async () => {
    try {
      // Use orchestration endpoint with local LLM forced
      const result = await callEndpoint('/orchestration/process', {
        user_input: llmPrompt,
        user_id: 'llm-test',
        use_rag: false,
        use_tools: false,
        force_local_llm: true,
        llm_backend: llmBackend,
        llm_model: llmModel
      }, 'POST', 'llm');
      setLlmResult(pretty(result));
    } catch (err: any) {
      setLlmResult(`Error: ${err.message}`);
    }
  };

  // RAG tests
  const testRAGQuery = async () => {
    try {
      const result = await callEndpoint('/rag/query', {
        query: ragQuery,
        top_k: ragTopK,
        rerank: true
      }, 'POST', 'rag');
      setRagResult(pretty(result));
    } catch (err: any) {
      setRagResult(`Error: ${err.message}`);
    }
  };

  const testAddDocument = async () => {
    try {
      const metadata = JSON.parse(documentMetadata);
      const result = await callEndpoint('/rag/index', {
        challenges: [{
          content: documentContent,
          metadata: metadata
        }]
      }, 'POST', 'rag');
      setRagResult(pretty(result));
    } catch (err: any) {
      setRagResult(`Error: ${err.message}`);
    }
  };

  // Tools tests
  const loadTools = async () => {
    try {
      // Try to get tools from orchestrator status
      const status = await callEndpoint('/orchestration/status', {}, 'GET');
      const tools = (status as any)?.tools || (status as any)?.available_tools || [];
      setToolsList(Array.isArray(tools) ? tools : []);
    } catch (err: any) {
      // If that fails, just set empty list - don't try adventure-data endpoint
      // as it requires lat/lng parameters which we don't have here
      setError(`Failed to load tools: ${err.message}`);
      setToolsList([]);
    }
  };

  const testTool = async () => {
    try {
      // For now, we'll use the orchestration process endpoint with tool usage
      const result = await callEndpoint('/orchestration/process', {
        user_input: `Execute tool ${toolName} with input: ${toolInput}`,
        user_id: 'test-user',
        use_tools: true
      }, 'POST', 'tools');
      setToolResult(pretty(result));
    } catch (err: any) {
      setToolResult(`Error: ${err.message}`);
    }
  };

  // State management tests
  const testCreateState = async () => {
    try {
      const wfId = stateWorkflowId || `state_${Date.now()}`;
      const data = JSON.parse(stateData);
      const result = await callEndpoint('/orchestration/workflow', {
        workflow_id: wfId,
        steps: [],
        initial_state: data
      }, 'POST', 'state');
      setStateResult(pretty(result));
      setStateWorkflowId(wfId);
    } catch (err: any) {
      setStateResult(`Error: ${err.message}`);
    }
  };

  const testCheckpoint = async () => {
    try {
      if (!stateWorkflowId) {
        throw new Error('Workflow ID required');
      }
      const result = await callEndpoint(`/orchestration/workflow/${stateWorkflowId}/checkpoint`, {
        step_name: checkpointName || 'test_checkpoint',
        state_data: JSON.parse(stateData)
      }, 'POST', 'state');
      setStateResult(pretty(result));
    } catch (err: any) {
      setStateResult(`Error: ${err.message}`);
    }
  };

  // Monitoring
  const loadSystemStatus = async () => {
    try {
      const status = await callEndpoint('/orchestration/status', {}, 'GET', 'monitoring');
      setSystemStatus(status);
    } catch (err: any) {
      setError(`Failed to load status: ${err.message}`);
    }
  };

  // Safety tests
  const testSafety = async () => {
    try {
      const result = await callEndpoint('/orchestration/process', {
        user_input: safetyInput,
        user_id: 'test-user'
      }, 'POST', 'safety');
      setSafetyResult(pretty(result));
    } catch (err: any) {
      setSafetyResult(`Error: ${err.message}`);
    }
  };

  // Chat
  const handleChatSend = async () => {
    if (!chatInput.trim()) return;
    
    const userMsg: ChatMessage = {
      role: 'user',
      content: chatInput.trim(),
      timestamp: new Date().toISOString()
    };
    setChatHistory(prev => [...prev, userMsg]);
    setChatInput('');
    
    try {
      const payload: any = {
        user_input: userMsg.content,
        user_id: 'chat-user',
        use_rag: true,
        use_tools: true
      };
      
      // If using local LLM, add local LLM parameters
      if (chatProvider === 'local') {
        payload.force_local_llm = true;
        payload.llm_backend = chatLocalLLMBackend;
        payload.llm_model = chatLocalLLMModel;
        payload.max_tokens = chatMaxTokens; // Use reduced tokens for faster response
        // Get base_url from selected service
        const selectedService = availableLLMServices.find(s => s.name === chatLocalLLMService);
        if (selectedService) {
          payload.llm_base_url = selectedService.base_url;
        }
      }
      
      const result = await callEndpoint('/orchestration/process', payload, 'POST', 'chat');
      
      // Handle different response formats
      let responseText = 'No response';
      let isError = false;
      
      if (!result || Object.keys(result).length === 0) {
        responseText = 'Empty response received from server';
        isError = true;
      } else if ((result as any).success === false) {
        // Error response from orchestrator
        responseText = `Error: ${(result as any).error || 'Request failed'}`;
        isError = true;
      } else if ((result as any).error) {
        // Error field present
        responseText = `Error: ${(result as any).error}`;
        isError = true;
      } else if (typeof result === 'string') {
        responseText = result;
      } else if ((result as any).response) {
        responseText = (result as any).response;
      } else if ((result as any).message) {
        responseText = (result as any).message;
      } else if ((result as any).content) {
        responseText = (result as any).content;
      } else {
        // Try to extract any text from the response
        responseText = JSON.stringify(result, null, 2);
      }
      
      const assistantMsg: ChatMessage = {
        role: isError ? 'system' : 'assistant',
        content: responseText,
        meta: result,
        timestamp: new Date().toISOString()
      };
      setChatHistory(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      let errorMessage = err.message || 'Unknown error';
      
      // Handle timeout errors more gracefully
      if (errorMessage.includes('timeout') || errorMessage.includes('timed out')) {
        errorMessage = `Request timed out. The LLM may be slow or unresponsive. Try reducing max_tokens or using a faster model.`;
      }
      
      // Handle connection errors
      if (errorMessage.includes('connection') || errorMessage.includes('connect')) {
        errorMessage = `Connection failed. Make sure the LLM service is running and accessible.`;
      }
      
      setChatHistory(prev => [...prev, {
        role: 'system',
        content: `Error: ${errorMessage}`,
        timestamp: new Date().toISOString()
      }]);
    }
  };

  // Testing functions - wrapped in useCallback to avoid dependency issues
  const loadTestList = useCallback(async () => {
    try {
      // Use separate fetch with short timeout for test list
      // Don't use AbortController to avoid "signal is aborted" errors
      const fetchPromise = fetch(`${baseUrl}/testing/list`, {
        method: 'GET',
        headers: headers || { 'Content-Type': 'application/json' }
      });
      
      // Use Promise.race with timeout instead of AbortController
      const timeoutPromise = new Promise<never>((_, reject) => 
        setTimeout(() => reject(new Error('Request timeout')), 5 * 1000)
      );
      
      const res = await Promise.race([fetchPromise, timeoutPromise]);
      
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      
      const result = await res.json();
      console.log('Test list loaded:', result);
      console.log('Result type:', typeof result);
      console.log('Result keys:', Object.keys(result || {}));
      
      // Handle both direct response and wrapped response
      const categories = result?.categories || result?.data?.categories || {};
      const files = result?.files || result?.data?.files || {};
      
      if (categories && Object.keys(categories).length > 0) {
        setTestCategories(categories);
        console.log('Set test categories:', Object.keys(categories));
      } else {
        console.warn('No categories found in response:', result);
      }
      
      if (files && Object.keys(files).length > 0) {
        setTestFiles(files);
        console.log('Set test files:', Object.keys(files));
      } else {
        console.warn('No files found in response:', result);
      }
    } catch (err: any) {
      // Don't show error for test list - it's optional
      // Just log it and use empty defaults
      console.warn('Failed to load test list:', err.message);
      // Set empty defaults so UI doesn't break
      setTestCategories({});
      setTestFiles({});
    }
  }, [baseUrl, headers]);

  const loadTestStatus = useCallback(async () => {
    try {
      // Use Promise.race with timeout instead of AbortController to avoid "signal is aborted" errors
      // Create a completely independent fetch that won't be affected by test abort controllers
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 4 * 1000); // 4 second timeout (backend has 3s)
      
      try {
        const res = await fetch(`${baseUrl}/testing/status`, {
          method: 'GET',
          headers: headers || { 'Content-Type': 'application/json' },
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        
        const status = await res.json();
        setTestStatus(status);
      } catch (fetchErr: any) {
        clearTimeout(timeoutId);
        // Handle abort errors gracefully
        if (fetchErr.name === 'AbortError' || fetchErr.message?.includes('aborted')) {
          throw new Error('Status check timed out');
        }
        throw fetchErr;
      }
    } catch (err: any) {
      // Don't set error for status checks - they're optional
      // Just set a default status with helpful message
      console.warn('Failed to load test status:', err.message);
      const errorMsg = err.message || 'Failed to load status';
      setTestStatus({
        tests_dir_exists: false,
        available_categories: 0,
        available_files: 0,
        status: 'error',
        error: errorMsg.includes('timeout') || errorMsg.includes('aborted') 
          ? 'Status check timed out (this is normal if tests are running)' 
          : errorMsg
      });
    }
  }, [baseUrl, headers]);

  const stopTests = () => {
    if (testAbortController) {
      testAbortController.abort();
      setTestAbortController(null);
      setTestRunning(false);
      setTestOutput(prev => [...prev, '\n[STOPPED] Test execution was stopped by user']);
    }
  };

  const runTestsStream = async () => {
    setTestRunning(true);
    setTestOutput([]);
    setTestResult(null);
    setTestSummary(null);
    setError(null);

    // Create abort controller for stop functionality
    const controller = new AbortController();
    setTestAbortController(controller);

    // Initialize debug info for testing tab
    setDebugInfo(prev => ({
      ...prev,
      testing: {
        request: {
          verbose: testVerbose,
          coverage: testCoverage,
          skip_slow: testSkipSlow,
          timeout: testTimeout,
          category: selectedTestCategory,
          file: selectedTestFile,
          test_path: selectedTestPath
        },
        logs: [`[${new Date().toISOString()}] Starting test execution`],
        timestamp: new Date().toISOString()
      }
    }));

    try {
      const payload: any = {
        verbose: testVerbose,
        coverage: testCoverage,
        skip_slow: testSkipSlow,
        timeout: testTimeout,
        output_format: 'json'
      };

      if (selectedTestCategory) {
        payload.category = selectedTestCategory;
      } else if (selectedTestFile) {
        payload.file = selectedTestFile;
        if (selectedTestPath) {
          payload.test_path = selectedTestPath;
        }
      }

      const response = await fetch(`${baseUrl}/testing/run/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error('No response body');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'start') {
                setTestOutput(prev => [...prev, `Starting tests: ${data.command}`]);
                setDebugInfo(prev => ({
                  ...prev,
                  testing: {
                    ...prev.testing,
                    logs: [...(prev.testing?.logs || []), `[${new Date().toISOString()}] ${data.command}`]
                  }
                }));
              } else if (data.type === 'output') {
                setTestOutput(prev => [...prev, data.line]);
              } else if (data.type === 'complete') {
                setTestResult({
                  success: data.success,
                  return_code: data.return_code
                });
                setTestOutput(prev => {
                  const updated = [...prev, `\nTests completed with return code: ${data.return_code}`];
                  return updated;
                });
                setDebugInfo(prev => {
                  const startTime = prev.testing?.timestamp ? new Date(prev.testing.timestamp).getTime() : Date.now();
                  const duration = Date.now() - startTime;
                  return {
                    ...prev,
                    testing: {
                      ...prev.testing,
                      response: { success: data.success, return_code: data.return_code },
                      duration,
                      logs: [...(prev.testing?.logs || []), `[${new Date().toISOString()}] Tests completed (${data.return_code})`]
                    }
                  };
                });
                // Parse summary after a brief delay to ensure all output is captured
                setTimeout(() => {
                  setTestOutput(current => {
                    parseTestSummary(current);
                    return current;
                  });
                }, 200);
              } else if (data.type === 'error') {
                setError(data.message);
                setTestOutput(prev => [...prev, `Error: ${data.message}`]);
                setDebugInfo(prev => ({
                  ...prev,
                  testing: {
                    ...prev.testing,
                    error: data.message,
                    logs: [...(prev.testing?.logs || []), `[${new Date().toISOString()}] ERROR: ${data.message}`]
                  }
                }));
              }
            } catch (e) {
              // Ignore JSON parse errors for malformed chunks
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setTestOutput(prev => [...prev, '\n[STOPPED] Test execution was stopped']);
        setDebugInfo(prev => ({
          ...prev,
          testing: {
            ...prev.testing,
            error: 'Test execution stopped by user',
            logs: [...(prev.testing?.logs || []), `[${new Date().toISOString()}] STOPPED by user`]
          }
        }));
      } else {
        setError(`Test execution failed: ${err.message}`);
        setTestOutput(prev => [...prev, `Error: ${err.message}`]);
        setDebugInfo(prev => ({
          ...prev,
          testing: {
            ...prev.testing,
            error: err.message,
            logs: [...(prev.testing?.logs || []), `[${new Date().toISOString()}] ERROR: ${err.message}`]
          }
        }));
      }
    } finally {
      setTestRunning(false);
      setTestAbortController(null);
    }
  };

  const parseTestSummary = (output: string[]) => {
    const outputText = output.join('\n');
    
    // Parse pytest summary line: "1 failed, 103 passed, 1 warning in 83.16s"
    const summaryMatch = outputText.match(/(\d+)\s+failed[,\s]+(\d+)\s+passed[,\s]+(\d+)\s+warning[^]*?in\s+([\d.]+)s/);
    if (summaryMatch) {
      const [, failed, passed, warnings, duration] = summaryMatch;
      
      // Extract failure details
      const failures: Array<{ test: string; error: string }> = [];
      const failureSections = outputText.split('=================================== FAILURES ===================================');
      if (failureSections.length > 1) {
        const failureText = failureSections[1];
        const testMatches = failureText.matchAll(/^([^\s]+::[^\s]+::[^\s]+)\s+([\s\S]*?)(?=^===|$)/gm);
        for (const match of testMatches) {
          failures.push({
            test: match[1],
            error: match[2].trim().substring(0, 500) // Limit error length
          });
        }
      }
      
      setTestSummary({
        passed: parseInt(passed),
        failed: parseInt(failed),
        warnings: parseInt(warnings),
        duration: `${duration}s`,
        failures
      });
    } else {
      // Try alternative format
      const altMatch = outputText.match(/(\d+)\s+passed[,\s]+(\d+)\s+failed/);
      if (altMatch) {
        setTestSummary({
          passed: parseInt(altMatch[1]),
          failed: parseInt(altMatch[2]),
          warnings: 0,
          duration: 'N/A',
          failures: []
        });
      }
    }
  };

  const runTestsSync = async () => {
    setTestRunning(true);
    setTestOutput([]);
    setTestResult(null);
    setTestSummary(null);
    setError(null);

    try {
      const payload: any = {
        verbose: testVerbose,
        coverage: testCoverage,
        skip_slow: testSkipSlow,
        timeout: testTimeout,
        output_format: 'json'
      };

      if (selectedTestCategory) {
        payload.category = selectedTestCategory;
      } else if (selectedTestFile) {
        payload.file = selectedTestFile;
        if (selectedTestPath) {
          payload.test_path = selectedTestPath;
        }
      }

      setTestOutput(['Starting tests...']);
      const result = await callEndpoint('/testing/run', payload);
      
      setTestResult({
        success: (result as any).success,
        return_code: (result as any).return_code
      });

      // Split stdout into lines for display
      const outputLines = (result as any).stdout?.split('\n') || [];
      const errorLines = (result as any).stderr?.split('\n') || [];
      const allLines = [
        ...outputLines.filter((l: string) => l.trim()),
        ...errorLines.filter((l: string) => l.trim())
      ];
      
      setTestOutput(allLines);
      parseTestSummary(allLines);
    } catch (err: any) {
      setError(`Test execution failed: ${err.message}`);
      setTestOutput(prev => [...prev, `Error: ${err.message}`]);
    } finally {
      setTestRunning(false);
    }
  };

  // Scan for LLM services
  const scanForLLMServices = useCallback(async () => {
    setScanningLLMServices(true);
    try {
      const result = await callEndpoint('/orchestration/llm-services', {}, 'GET');
      const services = (result as any)?.services || [];
      setAvailableLLMServices(services);
      
      // Auto-select first service if available
      if (services.length > 0 && !chatLocalLLMService) {
        const firstService = services[0];
        setChatLocalLLMService(firstService.name);
        if (firstService.models && firstService.models.length > 0) {
          setChatLocalLLMModel(firstService.models[0]);
        }
        if (firstService.type) {
          setChatLocalLLMBackend(firstService.type);
        }
      }
    } catch (err: any) {
      setError(`Failed to scan for LLM services: ${err.message}`);
    } finally {
      setScanningLLMServices(false);
    }
  }, [callEndpoint, chatLocalLLMService]);

  useEffect(() => {
    loadSystemStatus();
    loadTools();
    // Load test data immediately
    loadTestList();
    loadTestStatus();
    // Scan for LLM services on mount
    scanForLLMServices();
    const interval = setInterval(loadSystemStatus, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [baseUrl, apiKey, scanForLLMServices, loadTestList, loadTestStatus]);

  const isLoading = (key: string) => loading === key;

  // Auto-load health status when health tab is opened
  useEffect(() => {
    if (activeTab === 'health' && !healthStatus && !isLoading('/orchestration/health-comprehensive')) {
      const loadHealth = async () => {
        try {
          const health = await callEndpoint('/orchestration/health-comprehensive', {}, 'GET', 'health');
          if (health) {
            setHealthStatus(health);
            // Clear any previous errors if we got data
            if (health.errors && health.errors.length === 0) {
              setError(null);
            }
          }
        } catch (err: any) {
          console.error('Health status load error:', err);
          setError(`Failed to load health status: ${err.message || 'Unknown error'}`);
          // Set a minimal health status so the UI can still render
          setHealthStatus({
            timestamp: Date.now() / 1000,
            version: 'unknown',
            services: {},
            configuration: {},
            errors: [err.message || 'Failed to load health status']
          });
        }
      };
      loadHealth();
    }
  }, [activeTab, healthStatus, callEndpoint]);

  // Helper function to render debug panel
  const renderConnectionPanel = () => (
    <section style={{ 
      background: 'transparent', 
      padding: '0.75rem 1.25rem', 
      borderRadius: '12px',
      border: 'none',
      display: 'flex',
      gap: '1rem',
      alignItems: 'flex-start',
      flexWrap: 'wrap',
      boxShadow: 'none',
      flex: '0 0 auto'
    }}>
      <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ color: '#999', fontSize: '0.9rem' }}>Base URL:</span>
        <input
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          style={{
            padding: '0.5rem',
            background: '#222',
            border: '1px solid #444',
            borderRadius: '4px',
            color: '#e0e0e0',
            minWidth: '200px',
            fontSize: '0.9rem'
          }}
        />
      </label>
      <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ color: '#999', fontSize: '0.9rem' }}>API Key:</span>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="change-me"
          style={{
            padding: '0.5rem',
            background: '#222',
            border: '1px solid #444',
            borderRadius: '4px',
            color: '#e0e0e0',
            minWidth: '150px',
            fontSize: '0.9rem'
          }}
        />
      </label>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.25rem' }}>
        <button
          onClick={loadSystemStatus}
          disabled={isLoading('/orchestration/status')}
          style={{
            padding: '0.5rem 1rem',
            background: isLoading('/orchestration/status') ? '#444' : '#FFB84D',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: isLoading('/orchestration/status') ? 'not-allowed' : 'pointer',
            fontSize: '0.9rem',
            whiteSpace: 'nowrap'
          }}
        >
          {isLoading('/orchestration/status') ? 'Loading...' : 'Refresh Status'}
        </button>
      </div>
    </section>
  );

  const renderDebugPanel = (tabId: string) => {
    const debug = debugInfo[tabId];
    if (!debug) return null;

    return (
      <div style={{
        background: '#1a1a1a',
        padding: '1rem',
        borderRadius: '8px',
        border: '1px solid #333',
        marginTop: '1rem'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <h3 style={{ marginTop: 0, color: '#999', fontSize: '0.9rem' }}>Debug Information</h3>
          <button
            onClick={() => setShowDebug(prev => ({ ...prev, [tabId]: !prev[tabId] }))}
            style={{
              padding: '0.25rem 0.5rem',
              background: '#222',
              border: '1px solid #444',
              borderRadius: '4px',
              color: '#999',
              cursor: 'pointer',
              fontSize: '0.75rem'
            }}
          >
            {showDebug[tabId] ? '▼ Hide' : '▶ Show'}
          </button>
        </div>
        <div style={{ marginBottom: '0.5rem', fontSize: '0.85rem' }}>
          <span style={{ color: '#999' }}>Status: </span>
          <span style={{ 
            color: (() => {
              if (debug.error) return '#ff4444';
              if (debug.duration === undefined) return '#ffaa00';
              // Check response for success/failure
              if (debug.response && typeof debug.response === 'object' && debug.response !== null) {
                const resp = debug.response as Record<string, any>;
                if (resp.success === false || (resp.return_code !== undefined && resp.return_code !== 0)) {
                  return '#ff4444';
                }
              }
              return '#44ff44';
            })(),
            fontWeight: '500'
          }}>
            {(() => {
              if (debug.error) return 'Failed';
              if (debug.duration === undefined) return 'Loading';
              // Check response for success/failure
              if (debug.response && typeof debug.response === 'object' && debug.response !== null) {
                const resp = debug.response as Record<string, any>;
                if (resp.success === false || (resp.return_code !== undefined && resp.return_code !== 0)) {
                  return 'Failed';
                }
              }
              return 'Success';
            })()}
          </span>
        </div>
        {showDebug[tabId] && (
          <div style={{ display: 'grid', gap: '1rem' }}>
            {debug.timestamp && (
              <div style={{ fontSize: '0.85rem', color: '#666' }}>
                Timestamp: {new Date(debug.timestamp).toLocaleString()}
              </div>
            )}
            {debug.duration !== undefined && (
              <div style={{ fontSize: '0.85rem', color: '#666' }}>
                Duration: {debug.duration}ms
              </div>
            )}
            {debug.request !== undefined && debug.request !== null && (
              <div>
                <strong style={{ color: '#999', fontSize: '0.85rem', display: 'block', marginBottom: '0.5rem' }}>Request:</strong>
                <pre style={{
                  background: '#0a0a0a',
                  padding: '0.75rem',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                  overflow: 'auto',
                  maxHeight: '300px',
                  border: '1px solid #333'
                }}>
                  {pretty(debug.request)}
                </pre>
              </div>
            )}
            {debug.response !== undefined && debug.response !== null && (
              <div>
                <strong style={{ color: '#999', fontSize: '0.85rem', display: 'block', marginBottom: '0.5rem' }}>Response:</strong>
                <pre style={{
                  background: '#0a0a0a',
                  padding: '0.75rem',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                  overflow: 'auto',
                  maxHeight: '300px',
                  border: '1px solid #333',
                  color: (() => {
                    const response = debug.response;
                    if (typeof response === 'object' && response !== null) {
                      const resp = response as Record<string, any>;
                      if (resp.success === false || (resp.return_code !== undefined && resp.return_code !== 0)) {
                        return '#ff4444';
                      }
                      if (resp.success === true || (resp.return_code !== undefined && resp.return_code === 0)) {
                        return '#44ff44';
                      }
                    }
                    return '#ccc';
                  })()
                }}>
                  {pretty(debug.response)}
                </pre>
              </div>
            )}
            {debug.error && (
              <div>
                <strong style={{ color: '#ff4444', fontSize: '0.85rem', display: 'block', marginBottom: '0.5rem' }}>Error:</strong>
                <div style={{
                  background: '#ff444420',
                  padding: '0.75rem',
                  borderRadius: '4px',
                  color: '#ff4444',
                  fontSize: '0.85rem',
                  border: '1px solid #ff4444'
                }}>
                  {debug.error}
                </div>
              </div>
            )}
            {debug.logs && debug.logs.length > 0 && (
              <div>
                <strong style={{ color: '#999', fontSize: '0.85rem', display: 'block', marginBottom: '0.5rem' }}>Logs:</strong>
                <div style={{
                  background: '#0a0a0a',
                  padding: '0.75rem',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                  maxHeight: '200px',
                  overflow: 'auto',
                  border: '1px solid #333',
                  fontFamily: 'monospace'
                }}>
                  {debug.logs.map((log, idx) => (
                    <div key={idx} style={{ color: '#999', marginBottom: '0.25rem' }}>
                      {log}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0a', color: '#e0e0e0', fontFamily: 'system-ui', display: 'flex' }}>
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div style={{ 
        flex: 1,
        marginLeft: uiState.sidebarCollapsed ? '70px' : '250px',
        transition: 'margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh'
      }}>
        {/* Tab Content */}
        <div style={{ 
          padding: '1rem', 
          maxWidth: '100%', 
          margin: '0 auto',
          width: '100%',
          boxSizing: 'border-box',
          flex: 1
        }}>
        <React.Fragment>
          {/* Testing Tab */}
          {activeTab === 'testing' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 600 }}>Test Runner</h2>
                {renderConnectionPanel()}
              </div>
              
              {/* Test Infrastructure Status */}
              <div style={{
                background: '#1a1a1a',
                padding: '1rem',
                borderRadius: '8px',
                border: '1px solid #333',
                marginBottom: '1.5rem'
              }}>
                <h3 style={{ marginTop: 0, color: '#999', fontSize: '0.9rem' }}>Test Infrastructure Status</h3>
                {testStatus ? (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.5rem', fontSize: '0.85rem' }}>
                    <div>
                      <span style={{ color: '#666' }}>Tests Dir:</span>{' '}
                      <span style={{ color: testStatus.tests_dir_exists ? '#00aa00' : '#ff4444' }}>
                        {testStatus.tests_dir_exists ? 'Exists' : 'Missing'}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: '#666' }}>Categories:</span>{' '}
                      <span style={{ color: '#00aaff' }}>{testStatus.available_categories || testStatus.categories?.length || 0}</span>
                    </div>
                    <div>
                      <span style={{ color: '#666' }}>Files:</span>{' '}
                      <span style={{ color: '#00aaff' }}>{testStatus.available_files || testStatus.files?.length || 0}</span>
                    </div>
                    {testStatus.status === 'error' && testStatus.error && (
                      <div style={{ gridColumn: '1 / -1', color: '#ff4444', fontSize: '0.8rem', marginTop: '0.5rem' }}>
                        Error: {testStatus.error}
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{ color: '#666', fontSize: '0.85rem' }}>Loading status...</div>
                )}
              </div>

              <div style={{ display: 'grid', gap: '1.5rem' }}>
              {/* Test Selection, Options, and Output - Side by Side */}
              <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                {/* Left Column: Test Selection and Options */}
                <div style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: '300px', maxWidth: '100%' }}>
                  {/* Test Selection */}
                  <div style={{
                    background: '#1a1a1a',
                    padding: '1.5rem',
                    borderRadius: '8px',
                    border: '1px solid #333'
                  }}>
                    <h3 style={{ marginTop: 0, color: '#999' }}>Test Selection</h3>
                    <div style={{ display: 'grid', gap: '1rem' }}>
                      <div>
                        <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                          Run by Category
                        </label>
                        <select
                          value={selectedTestCategory}
                          onChange={(e) => {
                            setSelectedTestCategory(e.target.value);
                            setSelectedTestFile('');
                            setSelectedTestPath('');
                          }}
                          style={{
                            width: '100%',
                            padding: '0.75rem',
                            background: '#222',
                            border: '1px solid #444',
                            borderRadius: '4px',
                            color: '#e0e0e0'
                          }}
                        >
                          <option value="">-- Select Category --</option>
                          {Object.entries(testCategories).map(([key, cat]: [string, any]) => (
                            <option key={key} value={key}>
                              {cat.name} - {cat.description}
                            </option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                          Run by File
                        </label>
                        <select
                          value={selectedTestFile}
                          onChange={(e) => {
                            setSelectedTestFile(e.target.value);
                            setSelectedTestCategory('');
                            setSelectedTestPath('');
                          }}
                          style={{
                            width: '100%',
                            padding: '0.75rem',
                            background: '#222',
                            border: '1px solid #444',
                            borderRadius: '4px',
                            color: '#e0e0e0'
                          }}
                        >
                          <option value="">-- Select File --</option>
                          {Object.entries(testFiles).map(([key, path]) => (
                            <option key={key} value={key}>
                              {key} - {path}
                            </option>
                          ))}
                        </select>
                      </div>
                      
                      {selectedTestFile && (
                        <div>
                          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                            Specific Test (optional, e.g., TestClass::test_method)
                          </label>
                          <input
                            value={selectedTestPath}
                            onChange={(e) => setSelectedTestPath(e.target.value)}
                            placeholder="TestClass::test_method"
                            style={{
                              width: '100%',
                              padding: '0.75rem',
                              background: '#222',
                              border: '1px solid #444',
                              borderRadius: '4px',
                              color: '#e0e0e0'
                            }}
                          />
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Test Options */}
                  <div style={{
                    background: '#1a1a1a',
                    padding: '1.5rem',
                    borderRadius: '8px',
                    border: '1px solid #333'
                  }}>
                    <h3 style={{ marginTop: 0, color: '#999' }}>Test Options</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                      <Switch
                        checked={testVerbose}
                        onChange={setTestVerbose}
                        label="Verbose Output"
                      />
                      <Switch
                        checked={testCoverage}
                        onChange={setTestCoverage}
                        label="Coverage Report"
                      />
                      <Switch
                        checked={testSkipSlow}
                        onChange={setTestSkipSlow}
                        label="Skip Slow Tests"
                      />
                      <div>
                        <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999', fontSize: '0.9rem' }}>
                          Timeout (seconds)
                        </label>
                        <input
                          type="number"
                          value={testTimeout}
                          onChange={(e) => setTestTimeout(Number(e.target.value))}
                          min="0"
                          style={{
                            width: '100%',
                            padding: '0.5rem',
                            background: '#222',
                            border: '1px solid #444',
                            borderRadius: '4px',
                            color: '#e0e0e0'
                          }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Test Controls */}
                  <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap', width: '100%', minWidth: 0, overflow: 'visible', position: 'relative', zIndex: 1 }}>
                    <button
                      onClick={runTestsStream}
                      disabled={testRunning || (!selectedTestCategory && !selectedTestFile)}
                      style={{
                        padding: '0.75rem 1.5rem',
                        background: testRunning ? '#444' : '#FFB84D',
                        border: 'none',
                        borderRadius: '4px',
                        color: '#fff',
                        cursor: testRunning ? 'not-allowed' : 'pointer',
                        fontWeight: 600,
                        fontSize: '1rem',
                        whiteSpace: 'nowrap',
                        flexShrink: 0
                      }}
                    >
                      {testRunning ? 'Running...' : 'Run Tests (Streaming)'}
                    </button>
                    <button
                      onClick={runTestsSync}
                      disabled={testRunning || (!selectedTestCategory && !selectedTestFile)}
                      style={{
                        padding: '0.75rem 1.5rem',
                        background: testRunning ? '#444' : '#00aa00',
                        border: 'none',
                        borderRadius: '4px',
                        color: '#fff',
                        cursor: testRunning ? 'not-allowed' : 'pointer',
                        fontWeight: 600,
                        whiteSpace: 'nowrap',
                        flexShrink: 0
                      }}
                    >
                      {testRunning ? 'Running...' : 'Run Tests (Sync)'}
                    </button>
                    {testRunning && (
                      <button
                        onClick={stopTests}
                        style={{
                          padding: '0.75rem 1.5rem',
                          background: '#ff4444',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#fff',
                          cursor: 'pointer',
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                          flexShrink: 0
                        }}
                      >
                        Stop Tests
                      </button>
                    )}
                    <button
                      onClick={() => {
                        setTestOutput([]);
                        setTestResult(null);
                        setTestSummary(null);
                        setError(null);
                      }}
                      disabled={testRunning}
                      style={{
                        padding: '0.75rem 1.5rem',
                        background: '#666',
                        border: 'none',
                        borderRadius: '4px',
                        color: '#fff',
                        cursor: 'pointer',
                        whiteSpace: 'nowrap',
                        flexShrink: 0
                      }}
                    >
                      Clear Output
                    </button>
                    {testResult && (
                      <div style={{
                        padding: '0.5rem 1rem',
                        background: testResult.success ? '#00aa0020' : '#ff444420',
                        border: `1px solid ${testResult.success ? '#00aa00' : '#ff4444'}`,
                        borderRadius: '4px',
                        color: testResult.success ? '#00aa00' : '#ff4444',
                        fontWeight: 600,
                        whiteSpace: 'nowrap',
                        flexShrink: 0
                      }}>
                        {testResult.success ? 'PASSED' : 'FAILED'} (Code: {testResult.return_code})
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Column: Test Output */}
                <div style={{
                  background: '#0a0a0a',
                  padding: '1rem',
                  borderRadius: '8px',
                  border: '1px solid #333',
                  height: '567px',
                  overflow: 'auto',
                  fontFamily: 'monospace',
                  fontSize: '0.85rem',
                  lineHeight: '1.5',
                  flex: '2',
                  minWidth: '300px',
                  position: 'relative',
                  zIndex: 0
                }}>
                  {testOutput.length === 0 ? (
                    <div style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
                      No test output yet. Select tests and click "Run Tests" to start.
                    </div>
                  ) : (
                    <div style={{ color: '#e0e0e0' }}>
                      {testOutput.map((line, idx) => {
                        // Color code different types of output
                        let color = '#e0e0e0';
                        if (line.includes('PASSED') || line.includes('passed')) {
                          color = '#00aa00';
                        } else if (line.includes('FAILED') || line.includes('FAILED') || line.includes('Error')) {
                          color = '#ff4444';
                        } else if (line.includes('WARNING')) {
                          color = '#ffaa00';
                        } else if (line.includes('test_') || line.includes('::')) {
                          color = '#FFB84D';
                        }
                        
                        return (
                          <div key={idx} style={{ color, marginBottom: '0.25rem', whiteSpace: 'pre-wrap' }}>
                            {line}
                          </div>
                        );
                      })}
                      {testRunning && (
                        <div style={{ color: '#FFB84D', marginTop: '0.5rem' }}>
                          Running...
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Test Summary */}
              {testSummary && (
                <div style={{
                  background: '#1a1a1a',
                  padding: '1.5rem',
                  borderRadius: '8px',
                  border: `1px solid ${testSummary.failed > 0 ? '#ff4444' : '#00aa00'}`,
                  display: 'grid',
                  gap: '1rem'
                }}>
                  <h3 style={{ marginTop: 0, color: '#999' }}>Test Summary</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
                    <div style={{
                      padding: '1rem',
                      background: '#00aa0020',
                      border: '1px solid #00aa00',
                      borderRadius: '4px',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '2rem', fontWeight: 600, color: '#00aa00' }}>
                        {testSummary.passed}
                      </div>
                      <div style={{ color: '#999', fontSize: '0.9rem' }}>Passed</div>
                    </div>
                    <div style={{
                      padding: '1rem',
                      background: testSummary.failed > 0 ? '#ff444420' : '#222',
                      border: `1px solid ${testSummary.failed > 0 ? '#ff4444' : '#444'}`,
                      borderRadius: '4px',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '2rem', fontWeight: 600, color: testSummary.failed > 0 ? '#ff4444' : '#999' }}>
                        {testSummary.failed}
                      </div>
                      <div style={{ color: '#999', fontSize: '0.9rem' }}>Failed</div>
                    </div>
                    {testSummary.warnings > 0 && (
                      <div style={{
                        padding: '1rem',
                        background: '#ffaa0020',
                        border: '1px solid #ffaa00',
                        borderRadius: '4px',
                        textAlign: 'center'
                      }}>
                        <div style={{ fontSize: '2rem', fontWeight: 600, color: '#ffaa00' }}>
                          {testSummary.warnings}
                        </div>
                        <div style={{ color: '#999', fontSize: '0.9rem' }}>Warnings</div>
                      </div>
                    )}
                    <div style={{
                      padding: '1rem',
                      background: '#222',
                      border: '1px solid #444',
                      borderRadius: '4px',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '2rem', fontWeight: 600, color: '#FFB84D' }}>
                        {testSummary.duration}
                      </div>
                      <div style={{ color: '#999', fontSize: '0.9rem' }}>Duration</div>
                    </div>
                  </div>
                  
                  {testSummary.failures.length > 0 && (
                    <div>
                      <h4 style={{ color: '#ff4444', marginBottom: '0.5rem' }}>Failed Tests:</h4>
                      <div style={{ display: 'grid', gap: '0.5rem' }}>
                        {testSummary.failures.map((failure, idx) => (
                          <details key={idx} style={{
                            background: '#ff444410',
                            border: '1px solid #ff4444',
                            borderRadius: '4px',
                            padding: '0.75rem'
                          }}>
                            <summary style={{ cursor: 'pointer', color: '#ff4444', fontWeight: 600 }}>
                              {failure.test}
                            </summary>
                            <pre style={{
                              marginTop: '0.5rem',
                              padding: '0.5rem',
                              background: '#0a0a0a',
                              borderRadius: '4px',
                              fontSize: '0.8rem',
                              overflow: 'auto',
                              maxHeight: '200px',
                              color: '#ff8888'
                            }}>
                              {failure.error}
                            </pre>
                          </details>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {error && (
                <div style={{
                  background: '#ff444420',
                  border: '1px solid #ff4444',
                  borderRadius: '4px',
                  padding: '1rem',
                  color: '#ff4444'
                }}>
                  <strong>Error:</strong> {error}
                </div>
              )}

              {/* Debug Panel */}
              {renderDebugPanel('testing')}
              </div>
            </div>
          )}

          {/* Orchestration Tab */}
          {activeTab === 'orchestration' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 600 }}>Orchestration Testing</h2>
                {renderConnectionPanel()}
              </div>
              <div style={{ display: 'grid', gap: '1.5rem' }}>
                {/* Input, Options, and Output - Side by Side */}
                <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start' }}>
                  {/* Left Column: Input, Options, and Controls */}
                  <div style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: 0 }}>
                    {/* User Input */}
                    <div style={{
                      background: '#1a1a1a',
                      padding: '1.5rem',
                      borderRadius: '8px',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ marginTop: 0, color: '#999' }}>User Input</h3>
                      <textarea
                        value={orchestrationInput}
                        onChange={(e) => setOrchestrationInput(e.target.value)}
                        style={{
                          width: '100%',
                          minHeight: '150px',
                          padding: '0.75rem',
                          background: '#222',
                          border: '1px solid #444',
                          borderRadius: '4px',
                          color: '#e0e0e0',
                          fontFamily: 'monospace',
                          resize: 'vertical'
                        }}
                      />
                    </div>

                    {/* Options */}
                    <div style={{
                      background: '#1a1a1a',
                      padding: '1.5rem',
                      borderRadius: '8px',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ marginTop: 0, color: '#999' }}>Options</h3>
                      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                        <Switch
                          checked={useRAG}
                          onChange={setUseRAG}
                          label="Use RAG"
                        />
                        <Switch
                          checked={useTools}
                          onChange={setUseTools}
                          label="Use Tools"
                        />
                      </div>
                    </div>

                    {/* Controls */}
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                      <button
                        onClick={testOrchestration}
                        disabled={isLoading('/orchestration/process')}
                        style={{
                          padding: '0.75rem 1.5rem',
                          background: isLoading('/orchestration/process') ? '#444' : '#FFB84D',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#fff',
                          cursor: isLoading('/orchestration/process') ? 'not-allowed' : 'pointer',
                          fontWeight: 600
                        }}
                      >
                        {isLoading('/orchestration/process') ? 'Processing...' : 'Process Request'}
                      </button>
                      <button
                        onClick={testStreaming}
                        disabled={isStreaming}
                        style={{
                          padding: '0.75rem 1.5rem',
                          background: isStreaming ? '#444' : '#00aa00',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#fff',
                          cursor: isStreaming ? 'not-allowed' : 'pointer',
                          fontWeight: 600
                        }}
                      >
                        {isStreaming ? 'Streaming...' : 'Stream Response'}
                      </button>
                    </div>
                  </div>

                  {/* Right Column: Response Output */}
                  <div style={{
                    background: '#0a0a0a',
                    padding: '1rem',
                    borderRadius: '8px',
                    border: '1px solid #333',
                    height: '400px',
                    overflow: 'auto',
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    lineHeight: '1.5',
                    flex: '2',
                    minWidth: 0
                  }}>
                    {(orchestrationResult && orchestrationResult.trim()) || (streamingResponse && streamingResponse.trim()) ? (
                      <>
                        {orchestrationResult && orchestrationResult.trim() && (
                          <div>
                            <h3 style={{ color: '#999', fontSize: '0.9rem', marginTop: 0, marginBottom: '1rem' }}>Response:</h3>
                            <div style={{ color: '#e0e0e0', whiteSpace: 'pre-wrap' }}>
                              {orchestrationResult}
                            </div>
                          </div>
                        )}
                        {streamingResponse && streamingResponse.trim() && (
                          <div style={{ marginTop: (orchestrationResult && orchestrationResult.trim()) ? '2rem' : 0 }}>
                            <h3 style={{ color: '#999', fontSize: '0.9rem', marginTop: 0, marginBottom: '1rem' }}>Streaming Response:</h3>
                            <div style={{ color: '#e0e0e0', whiteSpace: 'pre-wrap' }}>
                              {streamingResponse}
                            </div>
                          </div>
                        )}
                      </>
                    ) : (
                      <div style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
                        No response yet. Enter input and click "Process Request" or "Stream Response" to start.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            
            {/* Debug Panel for Orchestration */}
            {renderDebugPanel('orchestration')}
          </div>
          )}

          {/* Workflow Tab */}
          {activeTab === 'workflow' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 600 }}>Workflow Execution</h2>
                {renderConnectionPanel()}
              </div>
              <div style={{ display: 'grid', gap: '1.5rem' }}>
                {/* Input and Output - Side by Side */}
                <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                  {/* Left Column: Input and Controls */}
                  <div style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: '300px', maxWidth: '100%' }}>
                    {/* Workflow Configuration */}
                    <div style={{
                      background: '#1a1a1a',
                      padding: '1.5rem',
                      borderRadius: '8px',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ marginTop: 0, color: '#999' }}>Workflow Configuration</h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div>
                          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                            Workflow ID
                          </label>
                          <input
                            value={workflowId}
                            onChange={(e) => setWorkflowId(e.target.value)}
                            placeholder="workflow_123 (auto-generated if empty)"
                            style={{
                              width: '100%',
                              padding: '0.75rem',
                              background: '#222',
                              border: '1px solid #444',
                              borderRadius: '4px',
                              color: '#e0e0e0'
                            }}
                          />
                        </div>
                        <div>
                          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                            Workflow Steps (JSON)
                          </label>
                          <textarea
                            value={workflowSteps}
                            onChange={(e) => setWorkflowSteps(e.target.value)}
                            style={{
                              width: '100%',
                              minHeight: '200px',
                              padding: '0.75rem',
                              background: '#222',
                              border: '1px solid #444',
                              borderRadius: '4px',
                              color: '#e0e0e0',
                              fontFamily: 'monospace',
                              resize: 'vertical'
                            }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Controls */}
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                      <button
                        onClick={testWorkflow}
                        disabled={isLoading('/orchestration/workflow')}
                        style={{
                          padding: '0.75rem 1.5rem',
                          background: isLoading('/orchestration/workflow') ? '#444' : '#FFB84D',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#fff',
                          cursor: isLoading('/orchestration/workflow') ? 'not-allowed' : 'pointer',
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                          flexShrink: 0
                        }}
                      >
                        {isLoading('/orchestration/workflow') ? 'Executing...' : 'Execute Workflow'}
                      </button>
                    </div>
                  </div>

                  {/* Right Column: Result Output */}
                  <div style={{
                    background: '#0a0a0a',
                    padding: '1rem',
                    borderRadius: '8px',
                    border: '1px solid #333',
                    height: '425px',
                    overflow: 'auto',
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    lineHeight: '1.5',
                    flex: '2',
                    minWidth: '300px',
                    position: 'relative',
                    zIndex: 0
                  }}>
                    {workflowResult && workflowResult.trim() ? (
                      <div>
                        <h3 style={{ color: '#999', fontSize: '0.9rem', marginTop: 0, marginBottom: '1rem' }}>Result:</h3>
                        <div style={{ color: '#e0e0e0', whiteSpace: 'pre-wrap' }}>
                          {workflowResult}
                        </div>
                      </div>
                    ) : (
                      <div style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
                        No result yet. Configure workflow and click "Execute Workflow" to start.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            
            {/* Debug Panel for Workflow */}
            {renderDebugPanel('workflow')}
          </div>
          )}

          {/* Local LLM Tab */}
          {activeTab === 'llm' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 600 }}>Local LLM Testing</h2>
                {renderConnectionPanel()}
              </div>
              <div style={{ display: 'grid', gap: '1.5rem' }}>
                {/* Input and Output - Side by Side */}
                <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                  {/* Left Column: Configuration and Controls */}
                  <div style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: '300px', maxWidth: '100%' }}>
                    {/* LLM Configuration */}
                    <div style={{
                      background: '#1a1a1a',
                      padding: '1.5rem',
                      borderRadius: '8px',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ marginTop: 0, color: '#999' }}>LLM Configuration</h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                          <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                              Backend
                            </label>
                            <select
                              value={llmBackend}
                              onChange={(e) => setLlmBackend(e.target.value)}
                              style={{
                                width: '100%',
                                padding: '0.75rem',
                                background: '#222',
                                border: '1px solid #444',
                                borderRadius: '4px',
                                color: '#e0e0e0'
                              }}
                            >
                              <option value="ollama">Ollama</option>
                              <option value="llama_cpp">llama.cpp</option>
                              <option value="transformers">Transformers</option>
                            </select>
                          </div>
                          <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                              Model
                            </label>
                            <input
                              value={llmModel}
                              onChange={(e) => setLlmModel(e.target.value)}
                              placeholder="llama3:8b"
                              style={{
                                width: '100%',
                                padding: '0.75rem',
                                background: '#222',
                                border: '1px solid #444',
                                borderRadius: '4px',
                                color: '#e0e0e0'
                              }}
                            />
                          </div>
                        </div>
                        <div>
                          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                            Prompt
                          </label>
                          <textarea
                            value={llmPrompt}
                            onChange={(e) => setLlmPrompt(e.target.value)}
                            style={{
                              width: '100%',
                              minHeight: '150px',
                              padding: '0.75rem',
                              background: '#222',
                              border: '1px solid #444',
                              borderRadius: '4px',
                              color: '#e0e0e0',
                              fontFamily: 'monospace',
                              resize: 'vertical'
                            }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Controls */}
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                      <button
                        onClick={testLocalLLM}
                        disabled={isLoading('/orchestration/process')}
                        style={{
                          padding: '0.75rem 1.5rem',
                          background: isLoading('/orchestration/process') ? '#444' : '#FFB84D',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#fff',
                          cursor: isLoading('/orchestration/process') ? 'not-allowed' : 'pointer',
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                          flexShrink: 0
                        }}
                      >
                        {isLoading('/orchestration/process') ? 'Generating...' : 'Generate Response'}
                      </button>
                    </div>
                  </div>

                  {/* Right Column: Result Output */}
                  <div style={{
                    background: '#0a0a0a',
                    padding: '1rem',
                    borderRadius: '8px',
                    border: '1px solid #333',
                    height: '376px',
                    overflow: 'auto',
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    lineHeight: '1.5',
                    flex: '2',
                    minWidth: '300px',
                    position: 'relative',
                    zIndex: 0
                  }}>
                    {llmResult && llmResult.trim() ? (
                      <div>
                        <h3 style={{ color: '#999', fontSize: '0.9rem', marginTop: 0, marginBottom: '1rem' }}>Response:</h3>
                        <div style={{ color: '#e0e0e0', whiteSpace: 'pre-wrap' }}>
                          {llmResult}
                        </div>
                      </div>
                    ) : (
                      <div style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
                        No response yet. Enter a prompt and click "Generate Response" to start.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            
            {/* Debug Panel for Local LLM */}
            {renderDebugPanel('llm')}
          </div>
          )}

          {/* RAG Tab */}
          {activeTab === 'rag' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 600 }}>RAG / Vector Store Testing</h2>
                {renderConnectionPanel()}
              </div>
              <div style={{ display: 'grid', gap: '1.5rem' }}>
                {/* Input and Output - Side by Side */}
                <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                  {/* Left Column: Query Documents and Add Document */}
                  <div style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: '300px', maxWidth: '100%' }}>
                    {/* Query Documents */}
                    <div style={{
                      background: '#1a1a1a',
                      padding: '1.5rem',
                      borderRadius: '8px',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ marginTop: 0, color: '#999' }}>Query Documents</h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem', minWidth: 0 }}>
                          <input
                            value={ragQuery}
                            onChange={(e) => setRagQuery(e.target.value)}
                            placeholder="Search query..."
                            style={{
                              padding: '0.75rem',
                              background: '#222',
                              border: '1px solid #444',
                              borderRadius: '4px',
                              color: '#e0e0e0',
                              width: '100%',
                              minWidth: 0,
                              boxSizing: 'border-box'
                            }}
                          />
                          <input
                            type="number"
                            value={ragTopK}
                            onChange={(e) => setRagTopK(Number(e.target.value))}
                            placeholder="Top K"
                            style={{
                              padding: '0.75rem',
                              background: '#222',
                              border: '1px solid #444',
                              borderRadius: '4px',
                              color: '#e0e0e0',
                              width: '100%',
                              minWidth: 0,
                              boxSizing: 'border-box'
                            }}
                          />
                        </div>
                        <button
                          onClick={testRAGQuery}
                          disabled={isLoading('/rag/query')}
                          style={{
                            padding: '0.75rem 1.5rem',
                            background: isLoading('/rag/query') ? '#444' : '#FFB84D',
                            border: 'none',
                            borderRadius: '4px',
                            color: '#fff',
                            cursor: isLoading('/rag/query') ? 'not-allowed' : 'pointer',
                            fontWeight: 600,
                            whiteSpace: 'nowrap',
                            flexShrink: 0
                          }}
                        >
                          {isLoading('/rag/query') ? 'Searching...' : 'Search'}
                        </button>
                      </div>
                    </div>

                    {/* Add Document */}
                    <div style={{
                      background: '#1a1a1a',
                      padding: '1.5rem',
                      borderRadius: '8px',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ marginTop: 0, color: '#999' }}>Add Document</h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <textarea
                          value={documentContent}
                          onChange={(e) => setDocumentContent(e.target.value)}
                          placeholder="Document content..."
                          style={{
                            width: '100%',
                            minHeight: '100px',
                            padding: '0.75rem',
                            background: '#222',
                            border: '1px solid #444',
                            borderRadius: '4px',
                            color: '#e0e0e0',
                            resize: 'vertical'
                          }}
                        />
                        <textarea
                          value={documentMetadata}
                          onChange={(e) => setDocumentMetadata(e.target.value)}
                          placeholder='{"source": "test", "type": "document"}'
                          style={{
                            width: '100%',
                            minHeight: '60px',
                            padding: '0.75rem',
                            background: '#222',
                            border: '1px solid #444',
                            borderRadius: '4px',
                            color: '#e0e0e0',
                            fontFamily: 'monospace',
                            resize: 'vertical'
                          }}
                        />
                        <button
                          onClick={testAddDocument}
                          disabled={isLoading('/rag/index')}
                          style={{
                            padding: '0.75rem 1.5rem',
                            background: isLoading('/rag/index') ? '#444' : '#00aa00',
                            border: 'none',
                            borderRadius: '4px',
                            color: '#fff',
                            cursor: isLoading('/rag/index') ? 'not-allowed' : 'pointer',
                            fontWeight: 600,
                            whiteSpace: 'nowrap',
                            flexShrink: 0
                          }}
                        >
                          {isLoading('/rag/index') ? 'Adding...' : 'Add Document'}
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Right Column: Result Output */}
                  <div style={{
                    background: '#0a0a0a',
                    padding: '1rem',
                    borderRadius: '8px',
                    border: '1px solid #333',
                    height: '546px',
                    overflow: 'auto',
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    lineHeight: '1.5',
                    flex: '2',
                    minWidth: '300px',
                    position: 'relative',
                    zIndex: 0
                  }}>
                    {ragResult && ragResult.trim() ? (
                      <div>
                        <h3 style={{ color: '#999', fontSize: '0.9rem', marginTop: 0, marginBottom: '1rem' }}>Result:</h3>
                        <div style={{ color: '#e0e0e0', whiteSpace: 'pre-wrap' }}>
                          {ragResult}
                        </div>
                      </div>
                    ) : (
                      <div style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
                        No result yet. Query documents or add a document to see results.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            
            {/* Debug Panel for RAG */}
            {renderDebugPanel('rag')}
          </div>
          )}

          {/* Tools Tab */}
          {activeTab === 'tools' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 600 }}>Tool Registry Testing</h2>
                {renderConnectionPanel()}
              </div>
              <div style={{ display: 'grid', gap: '1.5rem' }}>
                {/* Input and Output - Side by Side */}
                <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                  {/* Left Column: Tools List and Execute Tool */}
                  <div style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: '300px', maxWidth: '100%' }}>
                    {/* Load Tools */}
                    <div style={{
                      background: '#1a1a1a',
                      padding: '1.5rem',
                      borderRadius: '8px',
                      border: '1px solid #333'
                    }}>
                      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                        <button
                          onClick={loadTools}
                          disabled={isLoading('/orchestration/status')}
                          style={{
                            padding: '0.75rem 1.5rem',
                            background: isLoading('/orchestration/status') ? '#444' : '#FFB84D',
                            border: 'none',
                            borderRadius: '4px',
                            color: '#fff',
                            cursor: isLoading('/orchestration/status') ? 'not-allowed' : 'pointer',
                            fontWeight: 600,
                            whiteSpace: 'nowrap',
                            flexShrink: 0
                          }}
                        >
                          {isLoading('/orchestration/status') ? 'Loading...' : 'Load Tools'}
                        </button>
                        <span style={{ color: '#999' }}>
                          {toolsList.length} tools available
                        </span>
                      </div>
                    </div>

                    {/* Available Tools */}
                    {toolsList.length > 0 && (
                      <div style={{
                        background: '#1a1a1a',
                        padding: '1.5rem',
                        borderRadius: '8px',
                        border: '1px solid #333',
                        maxHeight: '300px',
                        overflow: 'auto'
                      }}>
                        <h3 style={{ marginTop: 0, color: '#999' }}>Available Tools:</h3>
                        <div style={{ display: 'grid', gap: '0.5rem' }}>
                          {toolsList.map((tool: any, idx: number) => (
                            <div
                              key={idx}
                              style={{
                                padding: '0.75rem',
                                background: '#222',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                border: '1px solid #333'
                              }}
                              onClick={() => {
                                setToolName(tool.name || '');
                                setToolInput('{}');
                              }}
                            >
                              <strong style={{ color: '#00aaff' }}>{tool.name}</strong>
                              <div style={{ color: '#999', fontSize: '0.85rem', marginTop: '0.25rem' }}>
                                {tool.description}
                              </div>
                              <div style={{ color: '#666', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                                Category: {tool.category} | Calls: {tool.calls || 0}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Execute Tool */}
                    <div style={{
                      background: '#1a1a1a',
                      padding: '1.5rem',
                      borderRadius: '8px',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ marginTop: 0, color: '#999' }}>Execute Tool</h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <input
                          value={toolName}
                          onChange={(e) => setToolName(e.target.value)}
                          placeholder="Tool name"
                          style={{
                            padding: '0.75rem',
                            background: '#222',
                            border: '1px solid #444',
                            borderRadius: '4px',
                            color: '#e0e0e0'
                          }}
                        />
                        <textarea
                          value={toolInput}
                          onChange={(e) => setToolInput(e.target.value)}
                          placeholder='{"param1": "value1"}'
                          style={{
                            width: '100%',
                            minHeight: '100px',
                            padding: '0.75rem',
                            background: '#222',
                            border: '1px solid #444',
                            borderRadius: '4px',
                            color: '#e0e0e0',
                            fontFamily: 'monospace',
                            resize: 'vertical'
                          }}
                        />
                        <button
                          onClick={testTool}
                          disabled={!toolName || isLoading('/orchestration/process')}
                          style={{
                            padding: '0.75rem 1.5rem',
                            background: (!toolName || isLoading('/orchestration/process')) ? '#444' : '#FFB84D',
                            border: 'none',
                            borderRadius: '4px',
                            color: '#fff',
                            cursor: (!toolName || isLoading('/orchestration/process')) ? 'not-allowed' : 'pointer',
                            fontWeight: 600,
                            whiteSpace: 'nowrap',
                            flexShrink: 0
                          }}
                        >
                          {isLoading('/orchestration/process') ? 'Executing...' : 'Execute Tool'}
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Right Column: Result Output */}
                  <div style={{
                    background: '#0a0a0a',
                    padding: '1rem',
                    borderRadius: '8px',
                    border: '1px solid #333',
                    height: '424px',
                    overflow: 'auto',
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    lineHeight: '1.5',
                    flex: '2',
                    minWidth: '300px',
                    position: 'relative',
                    zIndex: 0
                  }}>
                    {toolResult && toolResult.trim() ? (
                      <div>
                        <h3 style={{ color: '#999', fontSize: '0.9rem', marginTop: 0, marginBottom: '1rem' }}>Result:</h3>
                        <div style={{ color: '#e0e0e0', whiteSpace: 'pre-wrap' }}>
                          {toolResult}
                        </div>
                      </div>
                    ) : (
                      <div style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
                        No result yet. Load tools, select a tool, and click "Execute Tool" to see results.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            
            {/* Debug Panel for Tools */}
            {renderDebugPanel('tools')}
          </div>
          )}

          {/* State Management Tab */}
          {activeTab === 'state' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 600 }}>State Management</h2>
                {renderConnectionPanel()}
              </div>
              <div style={{ display: 'grid', gap: '1.5rem' }}>
                {/* Input and Output - Side by Side */}
                <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                  {/* Left Column: Configuration and Controls */}
                  <div style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: '300px', maxWidth: '100%' }}>
                    {/* State Configuration */}
                    <div style={{
                      background: '#1a1a1a',
                      padding: '1.5rem',
                      borderRadius: '8px',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ marginTop: 0, color: '#999' }}>State Configuration</h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div>
                          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                            Workflow ID
                          </label>
                          <input
                            value={stateWorkflowId}
                            onChange={(e) => setStateWorkflowId(e.target.value)}
                            placeholder="workflow_123"
                            style={{
                              width: '100%',
                              padding: '0.75rem',
                              background: '#222',
                              border: '1px solid #444',
                              borderRadius: '4px',
                              color: '#e0e0e0'
                            }}
                          />
                        </div>
                        <div>
                          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                            State Data (JSON)
                          </label>
                          <textarea
                            value={stateData}
                            onChange={(e) => setStateData(e.target.value)}
                            style={{
                              width: '100%',
                              minHeight: '100px',
                              padding: '0.75rem',
                              background: '#222',
                              border: '1px solid #444',
                              borderRadius: '4px',
                              color: '#e0e0e0',
                              fontFamily: 'monospace',
                              resize: 'vertical'
                            }}
                          />
                        </div>
                        <div>
                          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                            Checkpoint Name
                          </label>
                          <input
                            value={checkpointName}
                            onChange={(e) => setCheckpointName(e.target.value)}
                            placeholder="checkpoint_1"
                            style={{
                              width: '100%',
                              padding: '0.75rem',
                              background: '#222',
                              border: '1px solid #444',
                              borderRadius: '4px',
                              color: '#e0e0e0'
                            }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Controls */}
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                      <button
                        onClick={testCreateState}
                        disabled={isLoading('/orchestration/workflow')}
                        style={{
                          padding: '0.75rem 1.5rem',
                          background: isLoading('/orchestration/workflow') ? '#444' : '#FFB84D',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#fff',
                          cursor: isLoading('/orchestration/workflow') ? 'not-allowed' : 'pointer',
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                          flexShrink: 0
                        }}
                      >
                        Create State
                      </button>
                      <button
                        onClick={testCheckpoint}
                        disabled={!stateWorkflowId || isLoading('/orchestration/workflow')}
                        style={{
                          padding: '0.75rem 1.5rem',
                          background: (!stateWorkflowId || isLoading('/orchestration/workflow')) ? '#444' : '#00aa00',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#fff',
                          cursor: (!stateWorkflowId || isLoading('/orchestration/workflow')) ? 'not-allowed' : 'pointer',
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                          flexShrink: 0
                        }}
                      >
                        Create Checkpoint
                      </button>
                      <button
                        onClick={loadWorkflowStates}
                        disabled={!stateWorkflowId}
                        style={{
                          padding: '0.75rem 1.5rem',
                          background: !stateWorkflowId ? '#444' : '#666',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#fff',
                          cursor: !stateWorkflowId ? 'not-allowed' : 'pointer',
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                          flexShrink: 0
                        }}
                      >
                        Load State
                      </button>
                    </div>
                  </div>

                  {/* Right Column: Result Output */}
                  <div style={{
                    background: '#0a0a0a',
                    padding: '1rem',
                    borderRadius: '8px',
                    border: '1px solid #333',
                    height: '414px',
                    overflow: 'auto',
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    lineHeight: '1.5',
                    flex: '2',
                    minWidth: '300px',
                    position: 'relative',
                    zIndex: 0
                  }}>
                    {stateResult && stateResult.trim() ? (
                      <div>
                        <h3 style={{ color: '#999', fontSize: '0.9rem', marginTop: 0, marginBottom: '1rem' }}>Result:</h3>
                        <div style={{ color: '#e0e0e0', whiteSpace: 'pre-wrap' }}>
                          {stateResult}
                        </div>
                      </div>
                    ) : (
                      <div style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
                        No result yet. Configure state and click an action button to see results.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            
            {/* Debug Panel for State Management */}
            {renderDebugPanel('state')}
          </div>
          )}

          {/* Monitoring Tab */}
          {activeTab === 'monitoring' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 600 }}>System Monitoring</h2>
                {renderConnectionPanel()}
              </div>
            <div style={{ display: 'grid', gap: '1rem' }}>
              {systemStatus && (
                <div>
                  <h3 style={{ color: '#999' }}>System Status</h3>
                  <pre style={{
                    background: '#1a1a1a',
                    padding: '1rem',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '500px',
                    border: '1px solid #333'
                  }}>
                    {pretty(systemStatus)}
                  </pre>
                </div>
              )}
              {metrics && (
                <div>
                  <h3 style={{ color: '#999' }}>Metrics</h3>
                  <pre style={{
                    background: '#1a1a1a',
                    padding: '1rem',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '400px',
                    border: '1px solid #333'
                  }}>
                    {pretty(metrics)}
                  </pre>
                </div>
              )}
            </div>
            
            {/* Debug Panel for Monitoring */}
            {renderDebugPanel('monitoring')}
          </div>
          )}

          {/* Safety Tab */}
          {activeTab === 'safety' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 600 }}>Safety & Guardrails Testing</h2>
                {renderConnectionPanel()}
              </div>
              <div style={{ display: 'grid', gap: '1.5rem' }}>
                {/* Input and Output - Side by Side */}
                <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                  {/* Left Column: Configuration and Controls */}
                  <div style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: '300px', maxWidth: '100%' }}>
                    {/* Safety Test Input */}
                    <div style={{
                      background: '#1a1a1a',
                      padding: '1.5rem',
                      borderRadius: '8px',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ marginTop: 0, color: '#999' }}>Test Input</h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <label style={{ display: 'block', color: '#999', fontSize: '0.9rem' }}>
                          Input to be checked for safety
                        </label>
                        <textarea
                          value={safetyInput}
                          onChange={(e) => setSafetyInput(e.target.value)}
                          style={{
                            width: '100%',
                            minHeight: '150px',
                            padding: '0.75rem',
                            background: '#222',
                            border: '1px solid #444',
                            borderRadius: '4px',
                            color: '#e0e0e0',
                            resize: 'vertical'
                          }}
                        />
                      </div>
                    </div>

                    {/* Controls */}
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                      <button
                        onClick={testSafety}
                        disabled={isLoading('/orchestration/process')}
                        style={{
                          padding: '0.75rem 1.5rem',
                          background: isLoading('/orchestration/process') ? '#444' : '#FFB84D',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#fff',
                          cursor: isLoading('/orchestration/process') ? 'not-allowed' : 'pointer',
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                          flexShrink: 0
                        }}
                      >
                        {isLoading('/orchestration/process') ? 'Checking...' : 'Test Safety'}
                      </button>
                    </div>
                  </div>

                  {/* Right Column: Result Output */}
                  <div style={{
                    background: '#0a0a0a',
                    padding: '1rem',
                    borderRadius: '8px',
                    border: '1px solid #333',
                    height: '284px',
                    overflow: 'auto',
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    lineHeight: '1.5',
                    flex: '2',
                    minWidth: '300px',
                    position: 'relative',
                    zIndex: 0
                  }}>
                    {safetyResult && safetyResult.trim() ? (
                      <div>
                        <h3 style={{ color: '#999', fontSize: '0.9rem', marginTop: 0, marginBottom: '1rem' }}>Safety Check Result:</h3>
                        <div style={{ color: '#e0e0e0', whiteSpace: 'pre-wrap' }}>
                          {safetyResult}
                        </div>
                      </div>
                    ) : (
                      <div style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
                        No result yet. Enter test input and click "Test Safety" to see results.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            
            {/* Debug Panel for Safety */}
            {renderDebugPanel('safety')}
          </div>
          )}

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 600 }}>Interactive Chat</h2>
                {renderConnectionPanel()}
              </div>
              <div style={{ display: 'grid', gap: '1.5rem' }}>
                {/* Chat Box - Full Width */}
                <div style={{
                  background: '#0a0a0a',
                  padding: '1rem',
                  borderRadius: '8px',
                  border: '1px solid #333',
                  height: '600px',
                  fontFamily: 'monospace',
                  fontSize: '0.85rem',
                  lineHeight: '1.5',
                  position: 'relative',
                  zIndex: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  width: '100%'
                }}>
                    <div style={{ flex: 1, overflow: 'auto', marginBottom: '1rem' }}>
                      {chatHistory.length === 0 ? (
                        <div style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
                          No messages yet. Start a conversation!
                        </div>
                      ) : (
                        chatHistory.map((msg, idx) => (
                          <div
                            key={idx}
                            style={{
                              marginBottom: '1rem',
                              padding: '0.75rem',
                              background: msg.role === 'user' ? '#FFB84D20' : msg.role === 'assistant' ? '#00aa0020' : '#ff660020',
                              borderRadius: '4px',
                              borderLeft: `3px solid ${
                                msg.role === 'user' ? '#FFB84D' : msg.role === 'assistant' ? '#00aa00' : '#ff6600'
                              }`
                            }}
                          >
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                              <strong style={{ color: msg.role === 'user' ? '#FFB84D' : msg.role === 'assistant' ? '#00aa00' : '#ff6600' }}>
                                {msg.role.toUpperCase()}
                              </strong>
                              {msg.timestamp && (
                                <span style={{ color: '#666', fontSize: '0.75rem' }}>
                                  {new Date(msg.timestamp).toLocaleTimeString()}
                                </span>
                              )}
                            </div>
                            <div style={{ whiteSpace: 'pre-wrap', color: '#e0e0e0' }}>{msg.content}</div>
                            {msg.meta && (
                              <details style={{ marginTop: '0.5rem' }}>
                                <summary style={{ cursor: 'pointer', color: '#999', fontSize: '0.85rem' }}>Details</summary>
                                <pre style={{
                                  marginTop: '0.5rem',
                                  padding: '0.5rem',
                                  background: '#1a1a1a',
                                  borderRadius: '4px',
                                  fontSize: '0.8rem',
                                  overflow: 'auto',
                                  color: '#e0e0e0'
                                }}>
                                  {pretty(msg.meta)}
                                </pre>
                              </details>
                            )}
                          </div>
                        ))
                      )}
                    </div>
                    {/* Chat Input at Bottom */}
                    <div style={{ display: 'flex', gap: '0.5rem', paddingTop: '1rem', borderTop: '1px solid #333', alignItems: 'center' }}>
                      <input
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleChatSend()}
                        placeholder="Type your message..."
                        style={{
                          flex: 1,
                          padding: '0.75rem',
                          background: '#222',
                          border: '1px solid #444',
                          borderRadius: '4px',
                          color: '#e0e0e0'
                        }}
                      />
                      {/* Model Button */}
                      <div style={{ position: 'relative' }} data-provider-dropdown>
                        <button
                          onClick={() => setShowProviderDropdown(!showProviderDropdown)}
                          style={{
                            padding: '0.75rem 1rem',
                            background: 'transparent',
                            border: 'none',
                            color: '#999',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.25rem',
                            fontSize: '0.9rem',
                            whiteSpace: 'nowrap'
                          }}
                        >
                          <span>{chatProvider === 'api' ? 'API Key' : 'Local LLM'}</span>
                          <span style={{ fontSize: '0.75rem', marginLeft: '0.25rem' }}>▼</span>
                        </button>
                        {showProviderDropdown && (
                          <div style={{
                            position: 'absolute',
                            bottom: '100%',
                            right: 0,
                            marginBottom: '0.5rem',
                            background: '#1a1a1a',
                            border: '1px solid #333',
                            borderRadius: '4px',
                            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
                            minWidth: '200px',
                            zIndex: 1000
                          }}>
                            <button
                              onClick={() => {
                                setChatProvider('api');
                                setShowProviderDropdown(false);
                              }}
                              style={{
                                width: '100%',
                                padding: '0.75rem 1rem',
                                background: chatProvider === 'api' ? '#222' : 'transparent',
                                border: 'none',
                                color: '#e0e0e0',
                                cursor: 'pointer',
                                textAlign: 'left',
                                fontSize: '0.9rem'
                              }}
                            >
                              API Key (OpenAI/Cloud)
                            </button>
                            <button
                              onClick={() => {
                                setChatProvider('local');
                                setShowProviderDropdown(false);
                              }}
                              style={{
                                width: '100%',
                                padding: '0.75rem 1rem',
                                background: chatProvider === 'local' ? '#222' : 'transparent',
                                border: 'none',
                                borderTop: '1px solid #333',
                                color: '#e0e0e0',
                                cursor: 'pointer',
                                textAlign: 'left',
                                fontSize: '0.9rem'
                              }}
                            >
                              Local LLM (Ollama/LocalAI)
                            </button>
                          </div>
                        )}
                      </div>
                      <button
                        onClick={handleChatSend}
                        disabled={!chatInput.trim() || isLoading('/orchestration/process')}
                        style={{
                          padding: '0.75rem 1.5rem',
                          background: (!chatInput.trim() || isLoading('/orchestration/process')) ? '#444' : '#FFB84D',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#fff',
                          cursor: (!chatInput.trim() || isLoading('/orchestration/process')) ? 'not-allowed' : 'pointer',
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                          flexShrink: 0
                        }}
                      >
                        Send
                      </button>
                    </div>
                  </div>

                {/* Local LLM Configuration - Below Chat Box */}
                {chatProvider === 'local' && (
                  <div style={{
                    background: '#1a1a1a',
                    padding: '1.5rem',
                    borderRadius: '8px',
                    border: '1px solid #333'
                  }}>
                    <h3 style={{ marginTop: 0, color: '#999' }}>Local LLM Configuration</h3>
                    <div style={{ display: 'grid', gap: '1rem' }}>
                      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                        <button
                          onClick={scanForLLMServices}
                          disabled={scanningLLMServices}
                          style={{
                            padding: '0.5rem 1rem',
                            background: scanningLLMServices ? '#444' : '#FFB84D',
                            border: 'none',
                            borderRadius: '4px',
                            color: '#fff',
                            cursor: scanningLLMServices ? 'not-allowed' : 'pointer',
                            fontSize: '0.85rem',
                            fontWeight: 600
                          }}
                        >
                          {scanningLLMServices ? 'Scanning...' : 'Scan for Services'}
                        </button>
                        <span style={{ color: '#999', fontSize: '0.85rem' }}>
                          {availableLLMServices.length} service(s) found
                        </span>
                      </div>
                      
                      {availableLLMServices.length > 0 && (
                        <>
                          <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                              Service
                            </label>
                            <select
                              value={chatLocalLLMService}
                              onChange={(e) => {
                                setChatLocalLLMService(e.target.value);
                                const service = availableLLMServices.find(s => s.name === e.target.value);
                                if (service) {
                                  if (service.models && service.models.length > 0) {
                                    setChatLocalLLMModel(service.models[0]);
                                  }
                                  if (service.type) {
                                    setChatLocalLLMBackend(service.type);
                                  }
                                }
                              }}
                              style={{
                                width: '100%',
                                padding: '0.75rem',
                                background: '#222',
                                border: '1px solid #444',
                                borderRadius: '4px',
                                color: '#e0e0e0'
                              }}
                            >
                              {availableLLMServices.map(service => (
                                <option key={service.name} value={service.name}>
                                  {service.name} ({service.type}) - {service.base_url}
                                </option>
                              ))}
                            </select>
                          </div>
                          
                          <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                              Backend
                            </label>
                            <select
                              value={chatLocalLLMBackend}
                              onChange={(e) => setChatLocalLLMBackend(e.target.value)}
                              style={{
                                width: '100%',
                                padding: '0.75rem',
                                background: '#222',
                                border: '1px solid #444',
                                borderRadius: '4px',
                                color: '#e0e0e0'
                              }}
                            >
                              <option value="ollama">Ollama</option>
                              <option value="llama_cpp">llama.cpp</option>
                              <option value="transformers">Transformers</option>
                            </select>
                          </div>
                          
                          <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                              Model
                            </label>
                            {(() => {
                              const selectedService = availableLLMServices.find(s => s.name === chatLocalLLMService);
                              const models = selectedService?.models || [];
                              
                              if (models.length > 0) {
                                return (
                                  <select
                                    value={chatLocalLLMModel}
                                    onChange={(e) => setChatLocalLLMModel(e.target.value)}
                                    style={{
                                      width: '100%',
                                      padding: '0.75rem',
                                      background: '#222',
                                      border: '1px solid #444',
                                      borderRadius: '4px',
                                      color: '#e0e0e0'
                                    }}
                                  >
                                    {models.map(model => (
                                      <option key={model} value={model}>{model}</option>
                                    ))}
                                  </select>
                                );
                              } else {
                                return (
                                  <input
                                    value={chatLocalLLMModel}
                                    onChange={(e) => setChatLocalLLMModel(e.target.value)}
                                    placeholder="llama3:8b"
                                    style={{
                                      width: '100%',
                                      padding: '0.75rem',
                                      background: '#222',
                                      border: '1px solid #444',
                                      borderRadius: '4px',
                                      color: '#e0e0e0'
                                    }}
                                  />
                                );
                              }
                            })()}
                          </div>
                          
                          <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                              Max Tokens (lower = faster, default: 200)
                            </label>
                            <input
                              type="number"
                              value={chatMaxTokens}
                              onChange={(e) => setChatMaxTokens(Number(e.target.value))}
                              min="50"
                              max="2000"
                              style={{
                                width: '100%',
                                padding: '0.75rem',
                                background: '#222',
                                border: '1px solid #444',
                                borderRadius: '4px',
                                color: '#e0e0e0'
                              }}
                            />
                          </div>
                        </>
                      )}
                      
                      {availableLLMServices.length === 0 && !scanningLLMServices && (
                        <div style={{ color: '#ffaa00', fontSize: '0.85rem', padding: '0.5rem', background: '#ffaa0020', borderRadius: '4px' }}>
                          No local LLM services found. Click "Scan for Services" to discover services on the Docker network.
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            
            {/* Debug Panel for Chat */}
            {renderDebugPanel('chat')}
          </div>
          )}

          {/* Health Tab */}
          {activeTab === 'health' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 600 }}>System Health</h2>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <button
                    onClick={async () => {
                      try {
                        const health = await callEndpoint('/orchestration/health-comprehensive', {}, 'GET', 'health');
                        setHealthStatus(health);
                      } catch (err: any) {
                        setError(`Failed to load health status: ${err.message}`);
                      }
                    }}
                    disabled={isLoading('/orchestration/health-comprehensive')}
                    style={{
                      padding: '0.5rem 1rem',
                      background: isLoading('/orchestration/health-comprehensive') ? '#444' : '#FFB84D',
                      color: '#000',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: isLoading('/orchestration/health-comprehensive') ? 'not-allowed' : 'pointer',
                      fontWeight: 600,
                      fontSize: '0.9rem'
                    }}
                  >
                    {isLoading('/orchestration/health-comprehensive') ? 'Refreshing...' : 'Refresh Health'}
                  </button>
                  {renderConnectionPanel()}
                </div>
              </div>

              {isLoading('/orchestration/health-comprehensive') ? (
                <div style={{ 
                  color: '#999', 
                  textAlign: 'center', 
                  padding: '2rem',
                  background: '#1a1a1a',
                  borderRadius: '8px',
                  border: '1px solid #333'
                }}>
                  Loading health status...
                </div>
              ) : !healthStatus ? (
                <div style={{ 
                  color: '#666', 
                  textAlign: 'center', 
                  padding: '2rem',
                  background: '#1a1a1a',
                  borderRadius: '8px',
                  border: '1px solid #333'
                }}>
                  Failed to load health status. Click "Refresh Health" to retry.
                </div>
              ) : (
                <div style={{ display: 'grid', gap: '1rem' }}>
                  {/* Error Display */}
                  {healthStatus.errors && healthStatus.errors.length > 0 && (
                    <div style={{
                      background: '#ff444420',
                      borderRadius: '8px',
                      padding: '1rem',
                      border: '1px solid #ff4444'
                    }}>
                      <h4 style={{ margin: '0 0 0.5rem 0', color: '#ff4444' }}>Errors:</h4>
                      <ul style={{ margin: 0, paddingLeft: '1.5rem', color: '#ff4444' }}>
                        {healthStatus.errors.map((err: string, idx: number) => (
                          <li key={idx} style={{ fontSize: '0.9rem' }}>{err}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Orchestrator Status */}
                  {healthStatus.services?.orchestrator && (
                    <div style={{
                      background: '#1a1a1a',
                      borderRadius: '8px',
                      padding: '1.5rem',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.25rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ color: healthStatus.services.orchestrator.llm?.status === 'healthy' ? '#00aa00' : '#ff4444' }}>●</span>
                        Orchestrator
                      </h3>
                      <div style={{ display: 'grid', gap: '0.75rem' }}>
                        {healthStatus.services.orchestrator.llm && (
                          <div>
                            <div style={{ color: '#999', fontSize: '0.85rem', marginBottom: '0.25rem' }}>LLM Provider</div>
                            <div style={{ 
                              padding: '0.75rem', 
                              background: '#0a0a0a', 
                              borderRadius: '4px',
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center'
                            }}>
                              <div>
                                <div style={{ fontWeight: 600 }}>{healthStatus.services.orchestrator.llm.backend || 'Unknown'}</div>
                                {healthStatus.services.orchestrator.llm.model && (
                                  <div style={{ fontSize: '0.85rem', color: '#999', marginTop: '0.25rem' }}>
                                    Model: {healthStatus.services.orchestrator.llm.model}
                                  </div>
                                )}
                              </div>
                              <span style={{
                                padding: '0.25rem 0.75rem',
                                borderRadius: '4px',
                                background: healthStatus.services.orchestrator.llm.status === 'healthy' ? '#00aa0020' : '#ff444420',
                                color: healthStatus.services.orchestrator.llm.status === 'healthy' ? '#00aa00' : '#ff4444',
                                fontSize: '0.85rem',
                                fontWeight: 600
                              }}>
                                {healthStatus.services.orchestrator.llm.status || 'unknown'}
                              </span>
                            </div>
                            {healthStatus.services.orchestrator.llm.error && (
                              <div style={{ color: '#ff4444', fontSize: '0.85rem', marginTop: '0.25rem' }}>
                                Error: {healthStatus.services.orchestrator.llm.error}
                              </div>
                            )}
                          </div>
                        )}
                        {healthStatus.services.orchestrator.vector_store && (
                          <div>
                            <div style={{ color: '#999', fontSize: '0.85rem', marginBottom: '0.25rem' }}>Vector Store</div>
                            <div style={{ 
                              padding: '0.75rem', 
                              background: '#0a0a0a', 
                              borderRadius: '4px',
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center'
                            }}>
                              <div style={{ fontWeight: 600 }}>
                                {healthStatus.services.orchestrator.vector_store.type || 'Milvus'}
                              </div>
                              <span style={{
                                padding: '0.25rem 0.75rem',
                                borderRadius: '4px',
                                background: healthStatus.services.orchestrator.vector_store.healthy ? '#00aa0020' : '#ff444420',
                                color: healthStatus.services.orchestrator.vector_store.healthy ? '#00aa00' : '#ff4444',
                                fontSize: '0.85rem',
                                fontWeight: 600
                              }}>
                                {healthStatus.services.orchestrator.vector_store.healthy ? 'healthy' : 'unhealthy'}
                              </span>
                            </div>
                          </div>
                        )}
                        {healthStatus.services.orchestrator.tools && Object.keys(healthStatus.services.orchestrator.tools).length > 0 && (
                          <div>
                            <div style={{ color: '#999', fontSize: '0.85rem', marginBottom: '0.25rem' }}>Tools</div>
                            <div style={{ 
                              padding: '0.75rem', 
                              background: '#0a0a0a', 
                              borderRadius: '4px'
                            }}>
                              <div style={{ fontWeight: 600 }}>
                                {healthStatus.services.orchestrator.tools.total || 0} tools available
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* ModelKit Status */}
                  {healthStatus.services?.modelkit && (
                    <div style={{
                      background: '#1a1a1a',
                      borderRadius: '8px',
                      padding: '1.5rem',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.25rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ color: healthStatus.services.modelkit.status === 'available' ? '#00aa00' : '#ff4444' }}>●</span>
                        ModelKit
                        {healthStatus.services.modelkit.version && (
                          <span style={{ fontSize: '0.85rem', color: '#999', fontWeight: 400 }}>
                            v{healthStatus.services.modelkit.version}
                          </span>
                        )}
                      </h3>
                      <div style={{ display: 'grid', gap: '0.75rem' }}>
                        <div style={{ 
                          padding: '0.75rem', 
                          background: '#0a0a0a', 
                          borderRadius: '4px',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center'
                        }}>
                          <div>
                            <div style={{ fontWeight: 600 }}>Status</div>
                            <div style={{ fontSize: '0.85rem', color: '#999', marginTop: '0.25rem' }}>
                              {healthStatus.services.modelkit.status === 'available' ? 'Installed and available' : 
                               healthStatus.services.modelkit.status === 'not_installed' ? 'Not installed' : 
                               healthStatus.services.modelkit.status}
                            </div>
                          </div>
                          <span style={{
                            padding: '0.25rem 0.75rem',
                            borderRadius: '4px',
                            background: healthStatus.services.modelkit.status === 'available' ? '#00aa0020' : '#ff444420',
                            color: healthStatus.services.modelkit.status === 'available' ? '#00aa00' : '#ff4444',
                            fontSize: '0.85rem',
                            fontWeight: 600
                          }}>
                            {healthStatus.services.modelkit.status || 'unknown'}
                          </span>
                        </div>
                        {healthStatus.services.modelkit.models && healthStatus.services.modelkit.models.length > 0 && (
                          <div>
                            <div style={{ color: '#999', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                              Loaded Models ({healthStatus.services.modelkit.models.length})
                            </div>
                            <div style={{ display: 'grid', gap: '0.5rem' }}>
                              {healthStatus.services.modelkit.models.map((model: any, idx: number) => (
                                <div key={idx} style={{
                                  padding: '0.75rem',
                                  background: '#0a0a0a',
                                  borderRadius: '4px',
                                  border: '1px solid #333'
                                }}>
                                  <div style={{ fontWeight: 600 }}>{model.name || model}</div>
                                  {model.version && (
                                    <div style={{ fontSize: '0.85rem', color: '#999', marginTop: '0.25rem' }}>
                                      Version: {model.version}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        {healthStatus.services.modelkit.error && (
                          <div style={{ 
                            padding: '0.75rem', 
                            background: '#ff444420', 
                            borderRadius: '4px',
                            color: '#ff4444',
                            fontSize: '0.85rem'
                          }}>
                            Error: {healthStatus.services.modelkit.error}
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Milvus Status */}
                  {healthStatus.services?.milvus && (
                    <div style={{
                      background: '#1a1a1a',
                      borderRadius: '8px',
                      padding: '1.5rem',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.25rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ color: healthStatus.services.milvus.healthy ? '#00aa00' : '#ff4444' }}>●</span>
                        Milvus Vector Store
                      </h3>
                      <div style={{ display: 'grid', gap: '0.75rem' }}>
                        <div style={{ 
                          padding: '0.75rem', 
                          background: '#0a0a0a', 
                          borderRadius: '4px',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center'
                        }}>
                          <div>
                            <div style={{ fontWeight: 600 }}>Status</div>
                            <div style={{ fontSize: '0.85rem', color: '#999', marginTop: '0.25rem' }}>
                              {healthStatus.services.milvus.status || 'unknown'}
                            </div>
                          </div>
                          <span style={{
                            padding: '0.25rem 0.75rem',
                            borderRadius: '4px',
                            background: healthStatus.services.milvus.healthy ? '#00aa0020' : '#ff444420',
                            color: healthStatus.services.milvus.healthy ? '#00aa00' : '#ff4444',
                            fontSize: '0.85rem',
                            fontWeight: 600
                          }}>
                            {healthStatus.services.milvus.healthy ? 'healthy' : 'unhealthy'}
                          </span>
                        </div>
                        {healthStatus.services.milvus.connection && (
                          <div>
                            <div style={{ color: '#999', fontSize: '0.85rem', marginBottom: '0.25rem' }}>Connection</div>
                            <div style={{ 
                              padding: '0.75rem', 
                              background: '#0a0a0a', 
                              borderRadius: '4px'
                            }}>
                              <div style={{ display: 'grid', gap: '0.25rem', fontSize: '0.85rem' }}>
                                <div>Host: {healthStatus.services.milvus.connection.host || 'unknown'}</div>
                                <div>Port: {healthStatus.services.milvus.connection.port || 'unknown'}</div>
                                <div style={{ color: healthStatus.services.milvus.connection.connected ? '#00aa00' : '#ff4444' }}>
                                  Connected: {healthStatus.services.milvus.connection.connected ? 'Yes' : 'No'}
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                        {healthStatus.services.milvus.collection && (
                          <div>
                            <div style={{ color: '#999', fontSize: '0.85rem', marginBottom: '0.25rem' }}>Collection</div>
                            <div style={{ 
                              padding: '0.75rem', 
                              background: '#0a0a0a', 
                              borderRadius: '4px'
                            }}>
                              <div style={{ display: 'grid', gap: '0.25rem', fontSize: '0.85rem' }}>
                                <div>Name: {healthStatus.services.milvus.collection.name || 'unknown'}</div>
                                <div>Entities: {healthStatus.services.milvus.collection.num_entities || 0}</div>
                                <div>Mode: {healthStatus.services.milvus.collection.mode || 'unknown'}</div>
                                <div style={{ color: healthStatus.services.milvus.collection.loaded ? '#00aa00' : '#ff4444' }}>
                                  Loaded: {healthStatus.services.milvus.collection.loaded ? 'Yes' : 'No'}
                                </div>
                                <div style={{ color: healthStatus.services.milvus.collection.exists ? '#00aa00' : '#ff4444' }}>
                                  Exists: {healthStatus.services.milvus.collection.exists ? 'Yes' : 'No'}
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                        {healthStatus.services.milvus.embeddings && (
                          <div>
                            <div style={{ color: '#999', fontSize: '0.85rem', marginBottom: '0.25rem' }}>Embeddings</div>
                            <div style={{ 
                              padding: '0.75rem', 
                              background: '#0a0a0a', 
                              borderRadius: '4px'
                            }}>
                              <div style={{ fontSize: '0.85rem' }}>
                                Initialized: {healthStatus.services.milvus.embeddings.initialized ? 'Yes' : 'No'}
                                {healthStatus.services.milvus.embeddings.dimension && (
                                  <div style={{ marginTop: '0.25rem' }}>
                                    Dimension: {healthStatus.services.milvus.embeddings.dimension}
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        )}
                        {healthStatus.services.milvus.errors && healthStatus.services.milvus.errors.length > 0 && (
                          <div style={{ 
                            padding: '0.75rem', 
                            background: '#ff444420', 
                            borderRadius: '4px',
                            color: '#ff4444',
                            fontSize: '0.85rem'
                          }}>
                            <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>Errors:</div>
                            {healthStatus.services.milvus.errors.map((err: string, idx: number) => (
                              <div key={idx}>• {err}</div>
                            ))}
                          </div>
                        )}
                        {healthStatus.services.milvus.error && (
                          <div style={{ 
                            padding: '0.75rem', 
                            background: '#ff444420', 
                            borderRadius: '4px',
                            color: '#ff4444',
                            fontSize: '0.85rem'
                          }}>
                            Error: {healthStatus.services.milvus.error}
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* LLM Services */}
                  {healthStatus.services?.llm_services && healthStatus.services.llm_services.length > 0 && (
                    <div style={{
                      background: '#1a1a1a',
                      borderRadius: '8px',
                      padding: '1.5rem',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.25rem', fontWeight: 600 }}>
                        LLM Services ({healthStatus.services.llm_services.length})
                      </h3>
                      <div style={{ display: 'grid', gap: '0.75rem' }}>
                        {healthStatus.services.llm_services.map((service: any, idx: number) => (
                          <div key={idx} style={{
                            padding: '0.75rem',
                            background: '#0a0a0a',
                            borderRadius: '4px',
                            border: '1px solid #333'
                          }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                              <div style={{ fontWeight: 600 }}>{service.name || 'Unknown Service'}</div>
                              <span style={{
                                padding: '0.25rem 0.75rem',
                                borderRadius: '4px',
                                background: '#00aa0020',
                                color: '#00aa00',
                                fontSize: '0.85rem',
                                fontWeight: 600
                              }}>
                                {service.type || 'ollama'}
                              </span>
                            </div>
                            {service.base_url && (
                              <div style={{ fontSize: '0.85rem', color: '#999' }}>
                                URL: {service.base_url}
                              </div>
                            )}
                            {service.models && service.models.length > 0 && (
                              <div style={{ fontSize: '0.85rem', color: '#999', marginTop: '0.5rem' }}>
                                Models: {service.models.join(', ')}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Configuration */}
                  {healthStatus.configuration && (
                    <div style={{
                      background: '#1a1a1a',
                      borderRadius: '8px',
                      padding: '1.5rem',
                      border: '1px solid #333'
                    }}>
                      <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.25rem', fontWeight: 600 }}>Configuration</h3>
                      <div style={{ display: 'grid', gap: '0.5rem' }}>
                        {Object.entries(healthStatus.configuration).map(([key, value]) => (
                          <div key={key} style={{
                            padding: '0.5rem',
                            background: '#0a0a0a',
                            borderRadius: '4px',
                            display: 'flex',
                            justifyContent: 'space-between'
                          }}>
                            <span style={{ color: '#999', fontSize: '0.85rem' }}>{key.replace(/_/g, ' ')}</span>
                            <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>
                              {typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              {renderDebugPanel('health')}
            </div>
          )}

          {/* Test History Tab */}
          {activeTab === 'history' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 600 }}>Test History</h2>
                {renderConnectionPanel()}
              </div>
            <div style={{ 
              display: 'grid', 
              gap: '0.5rem',
              gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))'
            }}>
              {testHistory.length === 0 && (
                <div style={{ 
                  color: '#666', 
                  textAlign: 'center', 
                  padding: '2rem',
                  gridColumn: '1 / -1'
                }}>
                  No tests run yet. Start testing to see history here.
                </div>
              )}
              {testHistory.map((test, idx) => {
                // Extract request summary
                const requestSummary = typeof test.request === 'object' && test.request !== null
                  ? Object.keys(test.request).slice(0, 3).map(key => `${key}: ${JSON.stringify((test.request as any)[key]).substring(0, 30)}`).join(', ')
                  : String(test.request).substring(0, 100);
                
                // Extract response summary
                const responseSummary = test.response 
                  ? (typeof test.response === 'string' ? test.response.substring(0, 150) : JSON.stringify(test.response).substring(0, 150))
                  : null;
                
                // Format duration
                const durationStr = test.duration < 1000 
                  ? `${test.duration}ms` 
                  : `${(test.duration / 1000).toFixed(2)}s`;
                
                return (
                  <div
                    key={idx}
                    style={{
                      background: test.success ? '#00aa0020' : '#ff444420',
                      border: `1px solid ${test.success ? '#00aa00' : '#ff4444'}`,
                      borderRadius: '4px',
                      padding: '1rem',
                      minWidth: 0,
                      overflow: 'hidden',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.75rem'
                    }}
                  >
                    {/* Header with status badge */}
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      alignItems: 'flex-start',
                      flexWrap: 'wrap',
                      gap: '0.5rem'
                    }}>
                      <div style={{ flex: '1 1 auto', minWidth: 0 }}>
                        <div style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: '0.5rem',
                          marginBottom: '0.25rem'
                        }}>
                          <span style={{
                            padding: '0.25rem 0.5rem',
                            background: test.success ? '#00aa00' : '#ff4444',
                            borderRadius: '4px',
                            fontSize: '0.7rem',
                            fontWeight: 600,
                            color: '#fff'
                          }}>
                            {test.success ? '✓ SUCCESS' : '✗ FAILED'}
                          </span>
                          <strong style={{ 
                            color: test.success ? '#00aa00' : '#ff4444',
                            wordBreak: 'break-word',
                            fontSize: '0.9rem'
                          }}>
                            {test.endpoint}
                          </strong>
                        </div>
                        <div style={{ 
                          color: '#999', 
                          fontSize: '0.75rem',
                          display: 'flex',
                          gap: '0.75rem',
                          flexWrap: 'wrap'
                        }}>
                          <span>{durationStr}</span>
                          <span>{new Date(test.timestamp).toLocaleString()}</span>
                        </div>
                      </div>
                    </div>
                    
                    {/* Error display */}
                    {test.error && (
                      <div style={{ 
                        background: '#ff444420',
                        border: '1px solid #ff4444',
                        borderRadius: '4px',
                        padding: '0.5rem',
                        color: '#ff4444',
                        fontSize: '0.85rem'
                      }}>
                        <strong>Error:</strong> {test.error}
                      </div>
                    )}
                    
                    {/* Request Summary */}
                    <div>
                      <div style={{ 
                        color: '#999', 
                        fontSize: '0.75rem', 
                        marginBottom: '0.25rem',
                        fontWeight: 600
                      }}>
                        Request:
                      </div>
                      <div style={{
                        background: '#0a0a0a',
                        padding: '0.5rem',
                        borderRadius: '4px',
                        fontSize: '0.8rem',
                        color: '#ccc',
                        fontFamily: 'monospace',
                        wordBreak: 'break-word',
                        maxHeight: '80px',
                        overflow: 'auto'
                      }}>
                        {requestSummary}
                        {requestSummary.length >= 100 && '...'}
                      </div>
                    </div>
                    
                    {/* Response Summary */}
                    {responseSummary && (
                      <div>
                        <div style={{ 
                          color: '#999', 
                          fontSize: '0.75rem', 
                          marginBottom: '0.25rem',
                          fontWeight: 600
                        }}>
                          Response:
                        </div>
                        <div style={{
                          background: '#0a0a0a',
                          padding: '0.5rem',
                          borderRadius: '4px',
                          fontSize: '0.8rem',
                          color: '#ccc',
                          fontFamily: 'monospace',
                          wordBreak: 'break-word',
                          maxHeight: '80px',
                          overflow: 'auto'
                        }}>
                          {responseSummary}
                          {responseSummary.length >= 150 && '...'}
                        </div>
                      </div>
                    )}
                    
                    {/* Expandable Details */}
                    <details>
                      <summary style={{ 
                        cursor: 'pointer', 
                        color: '#999', 
                        fontSize: '0.85rem',
                        padding: '0.5rem',
                        background: '#222',
                        borderRadius: '4px',
                        border: '1px solid #333'
                      }}>
                        View Full Details
                      </summary>
                      <div style={{ marginTop: '0.5rem', display: 'grid', gap: '0.5rem' }}>
                        <div>
                          <strong style={{ color: '#999', fontSize: '0.85rem', display: 'block', marginBottom: '0.25rem' }}>Full Request:</strong>
                          <pre style={{
                            background: '#0a0a0a',
                            padding: '0.75rem',
                            borderRadius: '4px',
                            fontSize: '0.75rem',
                            overflow: 'auto',
                            maxHeight: '300px',
                            border: '1px solid #333'
                          }}>
                            {pretty(test.request)}
                          </pre>
                        </div>
                        {test.response != null && (
                          <div>
                            <strong style={{ color: '#999', fontSize: '0.85rem', display: 'block', marginBottom: '0.25rem' }}>Full Response:</strong>
                            <pre style={{
                              background: '#0a0a0a',
                              padding: '0.75rem',
                              borderRadius: '4px',
                              fontSize: '0.75rem',
                              overflow: 'auto',
                              maxHeight: '300px',
                              border: '1px solid #333'
                            }}>
                              {test.response}
                            </pre>
                          </div>
                        )}
                      </div>
                    </details>
                  </div>
                );
              })}
            </div>
          </div>
          )}
        </React.Fragment>
        </div>
      </div>
    </div>
  );
}
