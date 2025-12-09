import React, { useCallback, useMemo, useState, useEffect } from 'react';
import './App.css';

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

export default function App() {
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl);
  const [apiKey, setApiKey] = useState('');
  const [activeTab, setActiveTab] = useState('orchestration');
  
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
  
  // Safety/Guardrails state
  const [safetyInput, setSafetyInput] = useState('Generate a summary of my tasks');
  const [safetyResult, setSafetyResult] = useState('');
  
  // Chat state
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatProvider, setChatProvider] = useState<'api' | 'local'>('api');
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
    async (path: string, payload: unknown, method = 'POST') => {
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
    try {
      const result = await callEndpoint('/orchestration/process', {
        user_input: orchestrationInput,
        user_id: 'test-user',
        use_rag: useRAG,
        use_tools: useTools
      });
      setOrchestrationResult(pretty(result));
    } catch (err: any) {
      setOrchestrationResult(`Error: ${err.message}`);
    }
  };

  const testStreaming = async () => {
    setIsStreaming(true);
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
      });
      
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
      });
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
      });
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
      });
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
      });
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
      });
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
      });
      setStateResult(pretty(result));
    } catch (err: any) {
      setStateResult(`Error: ${err.message}`);
    }
  };

  // Monitoring
  const loadSystemStatus = async () => {
    try {
      const status = await callEndpoint('/orchestration/status', {}, 'GET');
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
      });
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
      
      const result = await callEndpoint('/orchestration/process', payload);
      
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
      const fetchPromise = fetch(`${baseUrl}/testing/status`, {
        method: 'GET',
        headers: headers || { 'Content-Type': 'application/json' }
      });
      
      const timeoutPromise = new Promise<never>((_, reject) => 
        setTimeout(() => reject(new Error('Request timeout')), 5 * 1000)
      );
      
      const res = await Promise.race([fetchPromise, timeoutPromise]);
      
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      
      const status = await res.json();
      setTestStatus(status);
    } catch (err: any) {
      // Don't set error for status checks - they're optional
      // Just set a default status
      console.warn('Failed to load test status:', err.message);
      setTestStatus({
        tests_dir_exists: false,
        available_categories: 0,
        available_files: 0,
        status: 'error',
        error: err.message || 'Failed to load status'
      });
    }
  }, [baseUrl, headers]);

  const runTestsStream = async () => {
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

      const response = await fetch(`${baseUrl}/testing/run/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload)
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
              }
            } catch (e) {
              // Ignore JSON parse errors for malformed chunks
            }
          }
        }
      }
    } catch (err: any) {
      setError(`Test execution failed: ${err.message}`);
      setTestOutput(prev => [...prev, `Error: ${err.message}`]);
    } finally {
      setTestRunning(false);
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

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0a', color: '#e0e0e0', fontFamily: 'system-ui' }}>
      <header style={{ 
        background: '#1a1a1a', 
        padding: '1.5rem', 
        borderBottom: '1px solid #333',
        position: 'sticky',
        top: 0,
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        gap: '1rem'
      }}>
        <div style={{
          background: 'rgba(26, 26, 26, 0.9)',
          padding: '0.5rem',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }}>
          <img 
            src="/logo.png" 
            alt="Deepiri Logo" 
            className="header-logo"
            style={{
              height: '2.5rem',
              width: 'auto',
              objectFit: 'contain',
              filter: 'brightness(1.1) contrast(1.1) drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))'
            }}
          />
        </div>
        <h1 style={{ margin: 0, fontSize: '2.5rem', fontWeight: 600 }}>
          <span
            className="cyrex-shimmer"
            style={{
              background: 'linear-gradient(90deg, #FFD700 0%, #FFA500 25%, #FFD700 50%, #FFA500 75%, #FFD700 100%)',
              backgroundSize: '200% 100%',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              display: 'inline-block',
            }}
          >
            Cyrex
          </span>
          {' Testing Interface'}
        </h1>
        <p style={{ margin: '0.5rem 0 0', color: '#999', fontSize: '0.9rem' }}>
          Comprehensive testing dashboard for orchestration, local LLMs, RAG, tools, and workflows
        </p>
      </header>

      {/* Connection Panel */}
      <section style={{ 
        background: '#151515', 
        padding: '1rem 1.5rem', 
        borderBottom: '1px solid #333',
        display: 'flex',
        gap: '1rem',
        alignItems: 'center',
        flexWrap: 'wrap'
      }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ color: '#999' }}>Base URL:</span>
          <input
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            style={{
              padding: '0.5rem',
              background: '#222',
              border: '1px solid #444',
              borderRadius: '4px',
              color: '#e0e0e0',
              minWidth: '250px'
            }}
          />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ color: '#999' }}>API Key:</span>
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
              minWidth: '200px'
            }}
          />
        </label>
        <button
          onClick={loadSystemStatus}
          disabled={isLoading('/orchestration/status')}
          style={{
            padding: '0.5rem 1rem',
            background: isLoading('/orchestration/status') ? '#444' : '#FFB84D',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: isLoading('/orchestration/status') ? 'not-allowed' : 'pointer'
          }}
        >
          {isLoading('/orchestration/status') ? 'Loading...' : '🔄 Refresh Status'}
        </button>
        {error && (
          <span style={{ color: '#ff4444', fontSize: '0.9rem' }}>{error}</span>
        )}
      </section>

      {/* Tabs */}
      <div className="tab-container" style={{ 
        display: 'flex', 
        borderBottom: '1px solid #333', 
        background: '#151515',
        flexWrap: 'wrap',
        overflowX: 'auto',
        gap: '0.25rem'
      }}>
        {[
          { id: 'testing', label: 'Testing', icon: '' },
          { id: 'orchestration', label: 'Orchestration', icon: '' },
          { id: 'workflow', label: 'Workflows', icon: '' },
          { id: 'llm', label: 'Local LLM', icon: '' },
          { id: 'rag', label: 'RAG/Vector Store', icon: '' },
          { id: 'tools', label: 'Tools', icon: '' },
          { id: 'state', label: 'State Management', icon: '' },
          { id: 'monitoring', label: 'Monitoring', icon: '' },
          { id: 'safety', label: 'Safety/Guardrails', icon: '' },
          { id: 'chat', label: 'Chat', icon: '' },
          { id: 'history', label: 'Test History', icon: '' }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '0.75rem 1rem',
              background: activeTab === tab.id ? '#FFB84D' : 'transparent',
              border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid #FFA500' : '2px solid transparent',
              color: activeTab === tab.id ? '#000' : '#999',
              cursor: 'pointer',
              fontSize: '0.85rem',
              fontWeight: activeTab === tab.id ? 600 : 400,
              whiteSpace: 'nowrap',
              flexShrink: 0,
              transition: 'background 0.2s, color 0.2s'
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div style={{ 
        padding: '1rem', 
        maxWidth: '100%', 
        margin: '0 auto',
        width: '100%',
        boxSizing: 'border-box'
      }}>
        <React.Fragment>
          {/* Testing Tab */}
          {activeTab === 'testing' && (
            <div>
              <h2 style={{ marginTop: 0 }}>Test Runner</h2>
              <div style={{ display: 'grid', gap: '1.5rem' }}>
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
                  
                  <div style={{ textAlign: 'center', color: '#666' }}>OR</div>
                  
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
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={testVerbose}
                      onChange={(e) => setTestVerbose(e.target.checked)}
                    />
                    <span>Verbose Output</span>
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={testCoverage}
                      onChange={(e) => setTestCoverage(e.target.checked)}
                    />
                    <span>Coverage Report</span>
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={testSkipSlow}
                      onChange={(e) => setTestSkipSlow(e.target.checked)}
                    />
                    <span>Skip Slow Tests</span>
                  </label>
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
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
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
                    fontSize: '1rem'
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
                    fontWeight: 600
                  }}
                >
                  {testRunning ? 'Running...' : 'Run Tests (Sync)'}
                </button>
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
                    cursor: 'pointer'
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
                    fontWeight: 600
                  }}>
                    {testResult.success ? 'PASSED' : 'FAILED'} (Code: {testResult.return_code})
                  </div>
                )}
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
              </div>

              {/* Test Output */}
              <div style={{
                background: '#0a0a0a',
                padding: '1rem',
                borderRadius: '8px',
                border: '1px solid #333',
                minHeight: '400px',
                maxHeight: '600px',
                overflow: 'auto',
                fontFamily: 'monospace',
                fontSize: '0.85rem',
                lineHeight: '1.5'
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

              {/* Test Status */}
              <div style={{
                background: '#1a1a1a',
                padding: '1rem',
                borderRadius: '8px',
                border: '1px solid #333'
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
            </div>
          )}

          {/* Orchestration Tab */}
          {activeTab === 'orchestration' && (
            <div>
              <h2 style={{ marginTop: 0 }}>Orchestration Testing</h2>
            <div style={{ display: 'grid', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                  User Input
                </label>
                <textarea
                  value={orchestrationInput}
                  onChange={(e) => setOrchestrationInput(e.target.value)}
                  style={{
                    width: '100%',
                    minHeight: '100px',
                    padding: '0.75rem',
                    background: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '4px',
                    color: '#e0e0e0',
                    fontFamily: 'monospace'
                  }}
                />
              </div>
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={useRAG}
                    onChange={(e) => setUseRAG(e.target.checked)}
                  />
                  <span>Use RAG</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={useTools}
                    onChange={(e) => setUseTools(e.target.checked)}
                  />
                  <span>Use Tools</span>
                </label>
                <button
                  onClick={testOrchestration}
                  disabled={isLoading('/orchestration/process')}
                  style={{
                    padding: '0.75rem 1.5rem',
                    background: '#FFB84D',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    cursor: 'pointer',
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
                    background: '#00aa00',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    cursor: 'pointer'
                  }}
                >
                  {isStreaming ? 'Streaming...' : '📡 Stream Response'}
                </button>
              </div>
              {orchestrationResult && (
                <div>
                  <h3 style={{ color: '#999', fontSize: '0.9rem' }}>Response:</h3>
                  <pre style={{
                    background: '#1a1a1a',
                    padding: '1rem',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '400px',
                    border: '1px solid #333'
                  }}>
                    {orchestrationResult}
                  </pre>
                </div>
              )}
              {streamingResponse && (
                <div>
                  <h3 style={{ color: '#999', fontSize: '0.9rem' }}>Streaming Response:</h3>
                  <pre style={{
                    background: '#1a1a1a',
                    padding: '1rem',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '400px',
                    border: '1px solid #333',
                    whiteSpace: 'pre-wrap'
                  }}>
                    {streamingResponse}
                  </pre>
                </div>
              )}
            </div>
          </div>
          )}

          {/* Workflow Tab */}
          {activeTab === 'workflow' && (
            <div>
            <h2 style={{ marginTop: 0 }}>Workflow Execution</h2>
            <div style={{ display: 'grid', gap: '1rem' }}>
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
                    background: '#1a1a1a',
                    border: '1px solid #333',
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
                    background: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '4px',
                    color: '#e0e0e0',
                    fontFamily: 'monospace'
                  }}
                />
              </div>
              <button
                onClick={testWorkflow}
                disabled={isLoading('/orchestration/workflow')}
                style={{
                  padding: '0.75rem 1.5rem',
                  background: '#0066ff',
                  border: 'none',
                  borderRadius: '4px',
                  color: '#fff',
                  cursor: 'pointer',
                  fontWeight: 600
                }}
              >
                {isLoading('/orchestration/workflow') ? 'Executing...' : 'Execute Workflow'}
              </button>
              {workflowResult && (
                <div>
                  <h3 style={{ color: '#999', fontSize: '0.9rem' }}>Result:</h3>
                  <pre style={{
                    background: '#1a1a1a',
                    padding: '1rem',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '400px',
                    border: '1px solid #333'
                  }}>
                    {workflowResult}
                  </pre>
                </div>
              )}
            </div>
          </div>
          )}

          {/* Local LLM Tab */}
          {activeTab === 'llm' && (
            <div>
            <h2 style={{ marginTop: 0 }}>Local LLM Testing</h2>
            <div style={{ display: 'grid', gap: '1rem' }}>
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
                      background: '#1a1a1a',
                      border: '1px solid #333',
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
                      background: '#1a1a1a',
                      border: '1px solid #333',
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
                    minHeight: '100px',
                    padding: '0.75rem',
                    background: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '4px',
                    color: '#e0e0e0',
                    fontFamily: 'monospace'
                  }}
                />
              </div>
              <button
                onClick={testLocalLLM}
                disabled={isLoading('/orchestration/process')}
                style={{
                  padding: '0.75rem 1.5rem',
                  background: '#0066ff',
                  border: 'none',
                  borderRadius: '4px',
                  color: '#fff',
                  cursor: 'pointer',
                  fontWeight: 600
                }}
              >
                {isLoading('/orchestration/process') ? 'Generating...' : 'Generate Response'}
              </button>
              {llmResult && (
                <div>
                  <h3 style={{ color: '#999', fontSize: '0.9rem' }}>Response:</h3>
                  <pre style={{
                    background: '#1a1a1a',
                    padding: '1rem',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '400px',
                    border: '1px solid #333',
                    whiteSpace: 'pre-wrap'
                  }}>
                    {llmResult}
                  </pre>
                </div>
              )}
            </div>
          </div>
          )}

          {/* RAG Tab */}
          {activeTab === 'rag' && (
            <div>
            <h2 style={{ marginTop: 0 }}>RAG / Vector Store Testing</h2>
            <div style={{ display: 'grid', gap: '1.5rem' }}>
              <div>
                <h3 style={{ color: '#999' }}>Query Documents</h3>
                <div style={{ display: 'grid', gap: '1rem' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem' }}>
                    <input
                      value={ragQuery}
                      onChange={(e) => setRagQuery(e.target.value)}
                      placeholder="Search query..."
                      style={{
                        padding: '0.75rem',
                        background: '#1a1a1a',
                        border: '1px solid #333',
                        borderRadius: '4px',
                        color: '#e0e0e0'
                      }}
                    />
                    <input
                      type="number"
                      value={ragTopK}
                      onChange={(e) => setRagTopK(Number(e.target.value))}
                      placeholder="Top K"
                      style={{
                        padding: '0.75rem',
                        background: '#1a1a1a',
                        border: '1px solid #333',
                        borderRadius: '4px',
                        color: '#e0e0e0'
                      }}
                    />
                  </div>
                  <button
                    onClick={testRAGQuery}
                    disabled={isLoading('/rag/query')}
                    style={{
                      padding: '0.75rem 1.5rem',
                      background: '#FFB84D',
                      border: 'none',
                      borderRadius: '4px',
                      color: '#fff',
                      cursor: 'pointer'
                    }}
                  >
                    {isLoading('/rag/query') ? 'Searching...' : '🔍 Search'}
                  </button>
                </div>
              </div>
              
              <div>
                <h3 style={{ color: '#999' }}>Add Document</h3>
                <div style={{ display: 'grid', gap: '1rem' }}>
                  <textarea
                    value={documentContent}
                    onChange={(e) => setDocumentContent(e.target.value)}
                    placeholder="Document content..."
                    style={{
                      width: '100%',
                      minHeight: '100px',
                      padding: '0.75rem',
                      background: '#1a1a1a',
                      border: '1px solid #333',
                      borderRadius: '4px',
                      color: '#e0e0e0'
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
                      background: '#1a1a1a',
                      border: '1px solid #333',
                      borderRadius: '4px',
                      color: '#e0e0e0',
                      fontFamily: 'monospace'
                    }}
                  />
                  <button
                    onClick={testAddDocument}
                    disabled={isLoading('/rag/index')}
                    style={{
                      padding: '0.75rem 1.5rem',
                      background: '#00aa00',
                      border: 'none',
                      borderRadius: '4px',
                      color: '#fff',
                      cursor: 'pointer'
                    }}
                  >
                    {isLoading('/rag/index') ? 'Adding...' : '➕ Add Document'}
                  </button>
                </div>
              </div>
              
              {ragResult && (
                <div>
                  <h3 style={{ color: '#999', fontSize: '0.9rem' }}>Result:</h3>
                  <pre style={{
                    background: '#1a1a1a',
                    padding: '1rem',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '400px',
                    border: '1px solid #333'
                  }}>
                    {ragResult}
                  </pre>
                </div>
              )}
            </div>
          </div>
          )}

          {/* Tools Tab */}
          {activeTab === 'tools' && (
            <div>
            <h2 style={{ marginTop: 0 }}>Tool Registry Testing</h2>
            <div style={{ display: 'grid', gap: '1rem' }}>
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                  <button
                    onClick={loadTools}
                    disabled={isLoading('/orchestration/status')}
                    style={{
                      padding: '0.75rem 1.5rem',
                      background: '#FFB84D',
                      border: 'none',
                      borderRadius: '4px',
                      color: '#fff',
                      cursor: 'pointer'
                    }}
                  >
                    {isLoading('/orchestration/status') ? 'Loading...' : '🔄 Load Tools'}
                  </button>
                <span style={{ color: '#999' }}>
                  {toolsList.length} tools available
                </span>
              </div>
              
              {toolsList.length > 0 && (
                <div style={{
                  background: '#1a1a1a',
                  padding: '1rem',
                  borderRadius: '4px',
                  border: '1px solid #333',
                  maxHeight: '300px',
                  overflow: 'auto'
                }}>
                  <h3 style={{ color: '#999', fontSize: '0.9rem', marginTop: 0 }}>Available Tools:</h3>
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
              
              <div>
                <h3 style={{ color: '#999' }}>Execute Tool</h3>
                <div style={{ display: 'grid', gap: '1rem' }}>
                  <input
                    value={toolName}
                    onChange={(e) => setToolName(e.target.value)}
                    placeholder="Tool name"
                    style={{
                      padding: '0.75rem',
                      background: '#1a1a1a',
                      border: '1px solid #333',
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
                      minHeight: '80px',
                      padding: '0.75rem',
                      background: '#1a1a1a',
                      border: '1px solid #333',
                      borderRadius: '4px',
                      color: '#e0e0e0',
                      fontFamily: 'monospace'
                    }}
                  />
                  <button
                    onClick={testTool}
                    disabled={!toolName || isLoading('/orchestration/process')}
                    style={{
                      padding: '0.75rem 1.5rem',
                      background: '#FFB84D',
                      border: 'none',
                      borderRadius: '4px',
                      color: '#fff',
                      cursor: 'pointer'
                    }}
                  >
                    {isLoading('/orchestration/process') ? 'Executing...' : 'Execute Tool'}
                  </button>
                </div>
              </div>
              
              {toolResult && (
                <div>
                  <h3 style={{ color: '#999', fontSize: '0.9rem' }}>Result:</h3>
                  <pre style={{
                    background: '#1a1a1a',
                    padding: '1rem',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '400px',
                    border: '1px solid #333'
                  }}>
                    {toolResult}
                  </pre>
                </div>
              )}
            </div>
          </div>
          )}

          {/* State Management Tab */}
          {activeTab === 'state' && (
            <div>
            <h2 style={{ marginTop: 0 }}>State Management</h2>
            <div style={{ display: 'grid', gap: '1rem' }}>
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
                    background: '#1a1a1a',
                    border: '1px solid #333',
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
                    background: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '4px',
                    color: '#e0e0e0',
                    fontFamily: 'monospace'
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
                    background: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '4px',
                    color: '#e0e0e0'
                  }}
                />
              </div>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <button
                  onClick={testCreateState}
                  disabled={isLoading('/orchestration/workflow')}
                  style={{
                    padding: '0.75rem 1.5rem',
                    background: '#FFB84D',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    cursor: 'pointer'
                  }}
                >
                  Create State
                </button>
                <button
                  onClick={testCheckpoint}
                  disabled={!stateWorkflowId || isLoading('/orchestration/workflow')}
                  style={{
                    padding: '0.75rem 1.5rem',
                    background: '#00aa00',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    cursor: 'pointer'
                  }}
                >
                  📍 Create Checkpoint
                </button>
                <button
                  onClick={loadWorkflowStates}
                  disabled={!stateWorkflowId}
                  style={{
                    padding: '0.75rem 1.5rem',
                    background: '#666',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    cursor: 'pointer'
                  }}
                >
                  🔄 Load State
                </button>
              </div>
              {stateResult && (
                <div>
                  <h3 style={{ color: '#999', fontSize: '0.9rem' }}>Result:</h3>
                  <pre style={{
                    background: '#1a1a1a',
                    padding: '1rem',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '400px',
                    border: '1px solid #333'
                  }}>
                    {stateResult}
                  </pre>
                </div>
              )}
            </div>
          </div>
          )}

          {/* Monitoring Tab */}
          {activeTab === 'monitoring' && (
            <div>
            <h2 style={{ marginTop: 0 }}>System Monitoring</h2>
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
          </div>
          )}

          {/* Safety Tab */}
          {activeTab === 'safety' && (
            <div>
            <h2 style={{ marginTop: 0 }}>Safety & Guardrails Testing</h2>
            <div style={{ display: 'grid', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                  Test Input (will be checked for safety)
                </label>
                <textarea
                  value={safetyInput}
                  onChange={(e) => setSafetyInput(e.target.value)}
                  style={{
                    width: '100%',
                    minHeight: '100px',
                    padding: '0.75rem',
                    background: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '4px',
                    color: '#e0e0e0'
                  }}
                />
              </div>
              <button
                onClick={testSafety}
                disabled={isLoading('/orchestration/process')}
                style={{
                  padding: '0.75rem 1.5rem',
                  background: '#ff6600',
                  border: 'none',
                  borderRadius: '4px',
                  color: '#fff',
                  cursor: 'pointer',
                  fontWeight: 600
                }}
              >
                {isLoading('/orchestration/process') ? 'Checking...' : 'Test Safety'}
              </button>
              {safetyResult && (
                <div>
                  <h3 style={{ color: '#999', fontSize: '0.9rem' }}>Safety Check Result:</h3>
                  <pre style={{
                    background: '#1a1a1a',
                    padding: '1rem',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '400px',
                    border: '1px solid #333'
                  }}>
                    {safetyResult}
                  </pre>
                </div>
              )}
            </div>
          </div>
          )}

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div>
            <h2 style={{ marginTop: 0 }}>Interactive Chat</h2>
            
            {/* Provider Selection */}
            <div style={{
              background: '#1a1a1a',
              padding: '1rem',
              borderRadius: '4px',
              border: '1px solid #333',
              marginBottom: '1rem'
            }}>
              <h3 style={{ marginTop: 0, color: '#999', fontSize: '0.9rem' }}>Provider Configuration</h3>
              <div style={{ display: 'grid', gap: '1rem' }}>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', color: '#999' }}>
                    Provider
                  </label>
                  <select
                    value={chatProvider}
                    onChange={(e) => setChatProvider(e.target.value as 'api' | 'local')}
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      background: '#222',
                      border: '1px solid #444',
                      borderRadius: '4px',
                      color: '#e0e0e0'
                    }}
                  >
                    <option value="api">API Key (OpenAI/Cloud)</option>
                    <option value="local">Local LLM (Ollama/LocalAI)</option>
                  </select>
                </div>
                
                {chatProvider === 'local' && (
                  <>
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
                          fontSize: '0.85rem'
                        }}
                      >
                        {scanningLLMServices ? 'Scanning...' : '🔍 Scan for Services'}
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
                  </>
                )}
              </div>
            </div>
            
            <div style={{
              background: '#1a1a1a',
              border: '1px solid #333',
              borderRadius: '4px',
              padding: '1rem',
              minHeight: '400px',
              display: 'flex',
              flexDirection: 'column',
              width: '100%',
              boxSizing: 'border-box'
            }}>
              <div style={{ flex: 1, overflow: 'auto', marginBottom: '1rem' }}>
                {chatHistory.length === 0 && (
                  <div style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
                    No messages yet. Start a conversation!
                  </div>
                )}
                {chatHistory.map((msg, idx) => (
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
                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                    {msg.meta && (
                      <details style={{ marginTop: '0.5rem' }}>
                        <summary style={{ cursor: 'pointer', color: '#999', fontSize: '0.85rem' }}>Details</summary>
                        <pre style={{
                          marginTop: '0.5rem',
                          padding: '0.5rem',
                          background: '#0a0a0a',
                          borderRadius: '4px',
                          fontSize: '0.8rem',
                          overflow: 'auto'
                        }}>
                          {pretty(msg.meta)}
                        </pre>
                      </details>
                    )}
                  </div>
                ))}
              </div>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
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
                <button
                  onClick={handleChatSend}
                  disabled={!chatInput.trim() || isLoading('/orchestration/process')}
                  style={{
                    padding: '0.75rem 1.5rem',
                    background: '#FFB84D',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    cursor: 'pointer',
                    fontWeight: 600
                  }}
                >
                  Send
                </button>
              </div>
            </div>
          </div>
          )}

          {/* Test History Tab */}
          {activeTab === 'history' && (
            <div>
            <h2 style={{ marginTop: 0 }}>Test History</h2>
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
              {testHistory.map((test, idx) => (
                <div
                  key={idx}
                  style={{
                    background: test.success ? '#00aa0020' : '#ff444420',
                    border: `1px solid ${test.success ? '#00aa00' : '#ff4444'}`,
                    borderRadius: '4px',
                    padding: '1rem',
                    minWidth: 0,
                    overflow: 'hidden'
                  }}
                >
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    marginBottom: '0.5rem',
                    flexWrap: 'wrap',
                    gap: '0.5rem'
                  }}>
                    <strong style={{ 
                      color: test.success ? '#00aa00' : '#ff4444',
                      wordBreak: 'break-word',
                      flex: '1 1 auto',
                      minWidth: 0
                    }}>
                      {test.endpoint}
                    </strong>
                    <span style={{ 
                      color: '#999', 
                      fontSize: '0.85rem',
                      whiteSpace: 'nowrap',
                      flexShrink: 0
                    }}>
                      {test.duration}ms | {new Date(test.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  {test.error && (
                    <div style={{ color: '#ff4444', marginBottom: '0.5rem' }}>
                      Error: {test.error}
                    </div>
                  )}
                  <details>
                    <summary style={{ cursor: 'pointer', color: '#999', fontSize: '0.85rem' }}>View Details</summary>
                    <div style={{ marginTop: '0.5rem', display: 'grid', gap: '0.5rem' }}>
                      <div>
                        <strong style={{ color: '#999', fontSize: '0.85rem' }}>Request:</strong>
                        <pre style={{
                          background: '#0a0a0a',
                          padding: '0.5rem',
                          borderRadius: '4px',
                          fontSize: '0.8rem',
                          overflow: 'auto',
                          maxHeight: '200px'
                        }}>
                          {pretty(test.request)}
                        </pre>
                      </div>
                      {test.response != null && (
                        <div>
                          <strong style={{ color: '#999', fontSize: '0.85rem' }}>Response:</strong>
                          <pre style={{
                            background: '#0a0a0a',
                            padding: '0.5rem',
                            borderRadius: '4px',
                            fontSize: '0.8rem',
                            overflow: 'auto',
                            maxHeight: '200px'
                          }}>
                            {test.response}
                          </pre>
                        </div>
                      )}
                    </div>
                  </details>
                </div>
              ))}
            </div>
          </div>
          )}
        </React.Fragment>
      </div>
    </div>
  );
}
