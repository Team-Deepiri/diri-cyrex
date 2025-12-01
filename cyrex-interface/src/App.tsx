import { useCallback, useMemo, useState, useEffect } from 'react';

type ChatMessage = {
  role: 'user' | 'assistant' | 'system';
  content: string;
  meta?: Record<string, unknown>;
  timestamp?: string;
};

type TestResult = {
  endpoint: string;
  request: unknown;
  response: unknown;
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
      
      try {
        const res = await fetch(`${baseUrl}${path}`, {
          method,
          headers,
          body: method !== 'GET' ? JSON.stringify(payload) : undefined
        });
        
        const duration = Date.now() - startTime;
        const text = await res.text();
        
        if (!res.ok) {
          throw new Error(`${res.status} ${res.statusText}: ${text}`);
        }
        
        const json = text ? JSON.parse(text) : {};
        
        // Record test result
        const testResult: TestResult = {
          endpoint: path,
          request: payload,
          response: json,
          duration,
          timestamp: new Date().toISOString(),
          success: true
        };
        setTestHistory(prev => [testResult, ...prev].slice(0, 50)); // Keep last 50
        
        return json;
      } catch (err: any) {
        const duration = Date.now() - startTime;
        const errorMsg = err?.message ?? 'Unknown error';
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
      // If that fails, try the agent tools endpoint
      try {
        const result = await callEndpoint('/agent/tools/external/adventure-data', {}, 'GET');
        setToolsList([{ name: 'adventure-data', description: 'Adventure data tool' }]);
      } catch (e: any) {
        setError(`Failed to load tools: ${err.message}`);
        setToolsList([]);
      }
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
      const result = await callEndpoint('/orchestration/process', {
        user_input: userMsg.content,
        user_id: 'chat-user',
        use_rag: true,
        use_tools: true
      });
      
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: (result as any).response || 'No response',
        meta: result,
        timestamp: new Date().toISOString()
      };
      setChatHistory(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      setChatHistory(prev => [...prev, {
        role: 'system',
        content: `⚠️ Error: ${err.message}`,
        timestamp: new Date().toISOString()
      }]);
    }
  };

  useEffect(() => {
    loadSystemStatus();
    loadTools();
    const interval = setInterval(loadSystemStatus, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [baseUrl, apiKey]);

  const isLoading = (key: string) => loading === key;

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0a', color: '#e0e0e0', fontFamily: 'system-ui' }}>
      <header style={{ 
        background: '#1a1a1a', 
        padding: '1.5rem', 
        borderBottom: '1px solid #333',
        position: 'sticky',
        top: 0,
        zIndex: 100
      }}>
        <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 600 }}>🧪 Cyrex Testing Interface</h1>
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
            background: isLoading('/orchestration/status') ? '#444' : '#0066ff',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: isLoading('/orchestration/status') ? 'not-allowed' : 'pointer'
          }}
        >
          {isLoading('/orchestration/status') ? 'Loading...' : '🔄 Refresh Status'}
        </button>
        {error && (
          <span style={{ color: '#ff4444', fontSize: '0.9rem' }}>⚠️ {error}</span>
        )}
      </section>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid #333', background: '#151515' }}>
        {[
          { id: 'orchestration', label: '🎯 Orchestration', icon: '🎯' },
          { id: 'workflow', label: '⚙️ Workflows', icon: '⚙️' },
          { id: 'llm', label: '🤖 Local LLM', icon: '🤖' },
          { id: 'rag', label: '📚 RAG/Vector Store', icon: '📚' },
          { id: 'tools', label: '🛠️ Tools', icon: '🛠️' },
          { id: 'state', label: '💾 State Management', icon: '💾' },
          { id: 'monitoring', label: '📊 Monitoring', icon: '📊' },
          { id: 'safety', label: '🛡️ Safety/Guardrails', icon: '🛡️' },
          { id: 'chat', label: '💬 Chat', icon: '💬' },
          { id: 'history', label: '📜 Test History', icon: '📜' }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '0.75rem 1.5rem',
              background: activeTab === tab.id ? '#0066ff' : 'transparent',
              border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid #00aaff' : '2px solid transparent',
              color: activeTab === tab.id ? '#fff' : '#999',
              cursor: 'pointer',
              fontSize: '0.9rem',
              fontWeight: activeTab === tab.id ? 600 : 400
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div style={{ padding: '1.5rem', maxWidth: '1400px', margin: '0 auto' }}>
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
                    background: '#0066ff',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    cursor: 'pointer',
                    fontWeight: 600
                  }}
                >
                  {isLoading('/orchestration/process') ? 'Processing...' : '🚀 Process Request'}
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
                {isLoading('/orchestration/workflow') ? 'Executing...' : '▶️ Execute Workflow'}
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
                {isLoading('/orchestration/process') ? 'Generating...' : '🤖 Generate Response'}
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
                      background: '#0066ff',
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
                      background: '#0066ff',
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
                      background: '#0066ff',
                      border: 'none',
                      borderRadius: '4px',
                      color: '#fff',
                      cursor: 'pointer'
                    }}
                  >
                    {isLoading('/orchestration/process') ? 'Executing...' : '▶️ Execute Tool'}
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
                    background: '#0066ff',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    cursor: 'pointer'
                  }}
                >
                  💾 Create State
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
                {isLoading('/orchestration/process') ? 'Checking...' : '🛡️ Test Safety'}
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
            <div style={{
              background: '#1a1a1a',
              border: '1px solid #333',
              borderRadius: '4px',
              padding: '1rem',
              minHeight: '400px',
              display: 'flex',
              flexDirection: 'column'
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
                      background: msg.role === 'user' ? '#0066ff20' : msg.role === 'assistant' ? '#00aa0020' : '#ff660020',
                      borderRadius: '4px',
                      borderLeft: `3px solid ${
                        msg.role === 'user' ? '#0066ff' : msg.role === 'assistant' ? '#00aa00' : '#ff6600'
                      }`
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                      <strong style={{ color: msg.role === 'user' ? '#0066ff' : msg.role === 'assistant' ? '#00aa00' : '#ff6600' }}>
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
                    background: '#0066ff',
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
            <div style={{ display: 'grid', gap: '0.5rem' }}>
              {testHistory.length === 0 && (
                <div style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
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
                    padding: '1rem'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <strong style={{ color: test.success ? '#00aa00' : '#ff4444' }}>
                      {test.success ? '✅' : '❌'} {test.endpoint}
                    </strong>
                    <span style={{ color: '#999', fontSize: '0.85rem' }}>
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
                      {test.response && (
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
                            {pretty(test.response)}
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
      </div>
    </div>
  );
}
