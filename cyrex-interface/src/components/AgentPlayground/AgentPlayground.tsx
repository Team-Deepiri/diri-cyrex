/**
 * Agent Playground Component
 * 
 * Features:
 * 1. See agents get spun up internally
 * 2. Test out the agent and evaluate it
 * 3. Visually see the agent performing
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  FaRobot,
  FaPlay,
  FaStop,
  FaSpinner,
  FaCog,
  FaMemory,
  FaTools,
  FaCheckCircle,
  FaExclamationCircle,
  FaCode,
  FaTerminal,
  FaChartLine,
  FaBrain,
  FaNetworkWired,
  FaSyncAlt
} from 'react-icons/fa';
import {
  TargetIcon,
  CodeIcon,
  ChartIcon,
  SearchIcon,
  ChatIcon,
  GearIcon,
  TimerIcon
} from './AgentIcons';
import './AgentPlayground.css';

// Types
interface AgentConfig {
  agentId: string;
  agentType: string;
  name: string;
  model: string;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
  tools: string[];
}

interface AgentInstance {
  instanceId: string;
  agentId: string;
  status: 'initializing' | 'idle' | 'processing' | 'completed' | 'error';
  currentTask?: string;
  startedAt: string;
  metrics: {
    tokensUsed: number;
    responseTime: number;
    toolCalls: number;
  };
}

interface AgentEvent {
  eventId: string;
  eventType: string;
  timestamp: string;
  data: Record<string, unknown>;
}

interface ToolCall {
  toolId: string;
  toolName: string;
  parameters: Record<string, unknown>;
  result?: unknown;
  status: 'pending' | 'executing' | 'completed' | 'failed';
  duration?: number;
}

interface ConversationMessage {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  toolCalls?: ToolCall[];
  timestamp: string;
  streaming?: boolean;
  isError?: boolean;
}

// Available agent types
const AGENT_TYPES = [
  { id: 'task_decomposer', name: 'Task Decomposer', icon: TargetIcon },
  { id: 'code_generator', name: 'Code Generator', icon: CodeIcon },
  { id: 'data_analyst', name: 'Data Analyst', icon: ChartIcon },
  { id: 'vendor_fraud', name: 'Vendor Fraud Detector', icon: SearchIcon },
  { id: 'conversational', name: 'Conversational', icon: ChatIcon },
  { id: 'automation', name: 'Automation Agent', icon: GearIcon },
];

// Models will be auto-detected from Ollama container

// Available tools
const AVAILABLE_TOOLS = [
  { id: 'search_memory', name: 'Search Memory', category: 'memory' },
  { id: 'store_memory', name: 'Store Memory', category: 'memory' },
  { id: 'http_get', name: 'HTTP GET', category: 'http' },
  { id: 'http_post', name: 'HTTP POST', category: 'http' },
  { id: 'db_query', name: 'Database Query', category: 'database' },
  { id: 'calculate', name: 'Calculator', category: 'math' },
  { id: 'search_documents', name: 'Search Documents', category: 'search' },
];

const API_BASE = import.meta.env.VITE_CYREX_BASE_URL || 'http://localhost:8000';

export function AgentPlayground() {
  // State
  const [activeTab, setActiveTab] = useState<'configure' | 'test' | 'monitor'>('configure');
  const [agentConfig, setAgentConfig] = useState<AgentConfig>({
    agentId: '',
    agentType: 'conversational',
    name: 'Test Agent',
    model: 'llama3:8b',
    temperature: 0.7,
    maxTokens: 2000,
    systemPrompt: 'You are a helpful AI assistant.',
    tools: ['search_memory', 'store_memory'],
  });
  const [agentInstance, setAgentInstance] = useState<AgentInstance | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [conversation, setConversation] = useState<ConversationMessage[]>([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [ollamaStatus, setOllamaStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [liveToolCalls, setLiveToolCalls] = useState<ToolCall[]>([]);
  
  // Model info interface
  interface ModelInfo {
    id: string;
    name: string;
    size?: string;
  }
  
  const [modelInfo, setModelInfo] = useState<ModelInfo[]>([]);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  // Check Ollama status
  const checkOllamaStatus = useCallback(async () => {
    // Create timeout manually (AbortSignal.timeout may not be available in all browsers)
    const createTimeout = (ms: number) => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), ms);
      return { signal: controller.signal, cleanup: () => clearTimeout(timeout) };
    };

    // First, try the models endpoint as it's more reliable
    // If models endpoint works, Ollama is definitely connected
    let modelsTimeout: { cleanup: () => void } | null = null;
    try {
      const timeout = createTimeout(5000);
      modelsTimeout = timeout;
      
      const modelsResponse = await fetch(`${API_BASE}/api/agent/models`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: timeout.signal
      });
      
      if (modelsResponse.ok) {
        const modelsData = await modelsResponse.json();
        console.debug('Models endpoint response:', modelsData);
        
        // If status is connected OR if we have models, Ollama is working
        const hasModels = (modelsData.model_names && modelsData.model_names.length > 0) ||
                         (modelsData.models && modelsData.models.length > 0);
        const isConnected = modelsData.status === 'connected' || 
                           modelsData.is_connected === true ||
                           hasModels;
        
        if (isConnected) {
          console.debug('Ollama connected via models endpoint');
          setOllamaStatus('connected');
          // If we have models, also update the model list
          if (hasModels) {
            return; // Success - models will be fetched by fetchAvailableModels
          }
          return; // Connected but no models yet
        }
      } else {
        // Response not OK, but might still be working
        console.debug('Models endpoint returned non-OK status:', modelsResponse.status);
      }
    } catch (modelsError: any) {
      // Models check failed, but don't mark as disconnected yet
      // Could be network issue or timeout
      console.debug('Models check failed:', modelsError.name, modelsError.message);
      
      // If it's just a timeout or network error, check current state before marking disconnected
      if (modelsError.name === 'AbortError' || modelsError.message?.includes('fetch') || modelsError.message?.includes('aborted')) {
        // Use functional update to access current state without dependency
        setOllamaStatus(prev => {
          if (prev === 'connected') {
            console.debug('Models check timeout but was connected, staying connected');
            return 'connected';
          }
          return prev;
        });
        return;
      }
    } finally {
      if (modelsTimeout) {
        modelsTimeout.cleanup();
      }
    }
    
    // Fallback to health endpoint
    let healthTimeout: { cleanup: () => void } | null = null;
    try {
      const timeout = createTimeout(5000);
      healthTimeout = timeout;
      
      const response = await fetch(`${API_BASE}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: timeout.signal
      });
      
      if (response.ok) {
        const data = await response.json();
        console.debug('Health endpoint response - ollama:', data.ollama);
        
        // Check if Ollama is healthy - be lenient
        // Health endpoint returns: { status: "healthy", is_connected: true, models: ["llama3:8b"], ... }
        const ollamaHealthy = data.ollama?.status === 'healthy';
        const ollamaConnected = data.ollama?.is_connected === true;
        const hasOllamaModels = data.ollama?.models && Array.isArray(data.ollama.models) && data.ollama.models.length > 0;
        
        if (ollamaHealthy || ollamaConnected || hasOllamaModels) {
          console.debug('Ollama connected via health endpoint');
          setOllamaStatus('connected');
          return;
        } else {
          console.debug('Health endpoint shows Ollama not connected:', data.ollama);
        }
      } else {
        console.debug('Health endpoint returned non-OK status:', response.status);
      }
    } catch (error: any) {
      console.debug('Health check failed:', error.name, error.message);
    } finally {
      if (healthTimeout) {
        healthTimeout.cleanup();
      }
    }
    
    // Only mark as disconnected if all checks failed
    // Use functional update to check current state
    setOllamaStatus(prev => {
      if (prev === 'connected') {
        // Was connected, but checks failed - might be transient, keep connected
        console.debug('Checks failed but was connected, keeping connected (might be transient)');
        return 'connected';
      }
      // Was not connected, mark as disconnected
      console.debug('No cached connection and all checks failed, marking as disconnected');
      setAvailableModels([]);
      setModelInfo([]);
      return 'disconnected';
    });
  }, []);

  // Fetch available models from Ollama
  const fetchAvailableModels = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/agent/models`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(10000) // 10 second timeout for model listing
      });
      
      if (!response.ok) {
        throw new Error(`Models endpoint returned ${response.status}`);
      }
      
      const data = await response.json();
      
      // If models endpoint says we're connected OR has models, we're connected
      const hasModels = (data.model_names && data.model_names.length > 0) ||
                       (data.models && data.models.length > 0);
      const isConnected = data.status === 'connected' || 
                          data.is_connected === true ||
                          hasModels;
      
      if (isConnected && hasModels) {
        const models = data.model_names.map((name: string) => {
          // Find full model info if available
          const fullInfo = data.models?.find((m: any) => m.name === name);
          const sizeGB = fullInfo?.size ? (fullInfo.size / (1024 * 1024 * 1024)).toFixed(1) + 'GB' : '';
          
          return {
            id: name,
            name: name,
            size: sizeGB,
          };
        });
        
        setAvailableModels(data.model_names);
        setModelInfo(models);
        setOllamaStatus('connected'); // Update status based on models endpoint
        
        // If current model is not in the list, switch to first available model
        setAgentConfig(prev => {
          if (models.length > 0 && !data.model_names.includes(prev.model)) {
            return { ...prev, model: models[0].id };
          }
          return prev;
        });
      } else if (isConnected) {
        // Connected but no models yet (might be loading)
        setOllamaStatus('connected');
        // Keep existing models if we have them (use functional update)
        setAvailableModels(prev => prev.length > 0 ? prev : []);
        setModelInfo(prev => prev.length > 0 ? prev : []);
      } else {
        // Not connected - use functional update to check current state
        setAvailableModels(prev => {
          if (prev.length === 0) {
            setOllamaStatus('disconnected');
            setModelInfo([]);
            return [];
          }
          // Have cached models, might be transient issue
          return prev;
        });
      }
    } catch (error: any) {
      console.error('Failed to fetch models:', error);
      
      // Use functional update to check current state without dependency
      setAvailableModels(prev => {
        if (prev.length === 0) {
          // Check if it's a timeout/network error vs actual failure
          if (error.name === 'TimeoutError' || error.name === 'AbortError') {
            // Timeout - keep current status
            setOllamaStatus(currentStatus => currentStatus === 'connected' ? 'connected' : 'disconnected');
          } else {
            setOllamaStatus('disconnected');
            setModelInfo([]);
          }
          return [];
        }
        // Have cached models, might be transient issue - keep them
        return prev;
      });
    }
  }, []);

  // Check Ollama status and fetch models on mount and periodically
  useEffect(() => {
    // Initial check - run both in parallel
    const initialCheck = async () => {
      await Promise.all([
        checkOllamaStatus(),
        fetchAvailableModels()
      ]);
    };
    initialCheck();
    
    // Set up periodic checks - every 30 seconds
    const interval = setInterval(() => {
      checkOllamaStatus();
      fetchAvailableModels();
    }, 30000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty deps - callbacks are stable now

  // Add event
  const addEvent = useCallback((eventType: string, data: Record<string, unknown>) => {
    const event: AgentEvent = {
      eventId: `evt-${Date.now()}`,
      eventType,
      timestamp: new Date().toISOString(),
      data,
    };
    setEvents(prev => [...prev.slice(-49), event]); // Keep last 50 events
  }, []);

  // Initialize agent
  const initializeAgent = async () => {
    setIsLoading(true);
    addEvent('agent_initializing', { config: agentConfig });

    try {
      const response = await fetch(`${API_BASE}/api/agent/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(agentConfig),
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = 'Failed to initialize agent';
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch {
          errorMessage = errorText || errorMessage;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      
      if (!data.instance_id) {
        throw new Error('Invalid response from server: missing instance_id');
      }
      
      const instance: AgentInstance = {
        instanceId: data.instance_id,
        agentId: data.agent_id || agentConfig.agentId || `agent-${Date.now()}`,
        status: 'idle',
        startedAt: data.started_at || new Date().toISOString(),
        metrics: {
          tokensUsed: 0,
          responseTime: 0,
          toolCalls: 0,
        },
      };

      setAgentInstance(instance);
      addEvent('agent_initialized', { instanceId: instance.instanceId });
      setActiveTab('test');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      addEvent('agent_error', { error: errorMessage });
      // Don't create mock instance - show error to user
      setAgentInstance(null);
      alert(`Failed to initialize agent: ${errorMessage}\n\nPlease check:\n- Ollama is running\n- Model is available\n- Network connection`);
    } finally {
      setIsLoading(false);
    }
  };

  // Initialize multiple test agents
  const initializeMultipleAgents = async () => {
    setIsLoading(true);
    addEvent('multiple_agents_initializing', {});

    try {
      // Create 3 different test agents
      const testAgents = [
        {
          ...agentConfig,
          name: 'Code Assistant',
          agentType: 'code_generator',
          systemPrompt: 'You are a helpful code assistant. Write clean, efficient code.',
          tools: ['calculate', 'http_get'],
        },
        {
          ...agentConfig,
          name: 'Data Analyst',
          agentType: 'data_analyst',
          systemPrompt: 'You are a data analyst. Help analyze and interpret data.',
          tools: ['calculate', 'search_documents'],
        },
        {
          ...agentConfig,
          name: 'Task Decomposer',
          agentType: 'task_decomposer',
          systemPrompt: 'You are a task decomposer. Break down complex tasks into smaller steps.',
          tools: ['search_memory', 'store_memory'],
        },
      ];

      const response = await fetch(`${API_BASE}/api/agent/initialize-multiple`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agents: testAgents }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = 'Failed to initialize multiple agents';
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch {
          errorMessage = errorText || errorMessage;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      
      if (data.successful > 0) {
        addEvent('multiple_agents_initialized', { 
          total: data.total, 
          successful: data.successful 
        });
        alert(`Successfully initialized ${data.successful} out of ${data.total} agents!\n\nYou can now chat with them using the Messages widget (💬 button).`);
      } else {
        throw new Error('Failed to initialize any agents');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      addEvent('agent_error', { error: errorMessage });
      alert(`Failed to initialize multiple agents: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Stop agent
  const stopAgent = async () => {
    if (!agentInstance) return;

    addEvent('agent_stopping', { instanceId: agentInstance.instanceId });
    
    try {
      await fetch(`${API_BASE}/api/agent/${agentInstance.instanceId}/stop`, {
        method: 'POST',
      });
    } catch (error) {
      // Ignore errors
    }

    setAgentInstance(null);
    setConversation([]);
    setLiveToolCalls([]);
    addEvent('agent_stopped', {});
  };

  // Verify agent instance is still valid
  const verifyAgentInstance = async (instanceId: string): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE}/api/agent/${instanceId}/status`);
      return response.ok;
    } catch {
      return false;
    }
  };

  // Send message to agent
  const sendMessage = async () => {
    if (!userInput.trim()) return;
    
    if (!agentInstance) {
      alert('Please initialize the agent first before sending messages.');
      return;
    }
    
    if (!agentInstance.instanceId) {
      alert('Invalid agent instance. Please re-initialize the agent.');
      return;
    }
    
    // Verify instance is still valid
    const isValid = await verifyAgentInstance(agentInstance.instanceId);
    if (!isValid) {
      alert('Agent instance is no longer valid. Please re-initialize the agent.');
      setAgentInstance(null);
      return;
    }

    const userMessage: ConversationMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: userInput,
      timestamp: new Date().toISOString(),
    };

    setConversation(prev => [...prev, userMessage]);
    setUserInput('');
    setIsStreaming(true);
    setLiveToolCalls([]);
    
    addEvent('message_sent', { content: userInput });

    // Update agent status
    setAgentInstance(prev => prev ? { ...prev, status: 'processing', currentTask: userInput } : null);

    // Add placeholder for streaming response
    const assistantMessage: ConversationMessage = {
      id: `msg-${Date.now()}-response`,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      streaming: true,
    };
    setConversation(prev => [...prev, assistantMessage]);

    try {
      const response = await fetch(`${API_BASE}/api/agent/invoke`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instance_id: agentInstance.instanceId,
          input: userInput,
          config: agentConfig,
          stream: true,
        }),
      });

      if (!response.ok) {
        let errorMessage = 'Failed to invoke agent';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch {
          const errorText = await response.text();
          errorMessage = errorText || errorMessage;
        }
        
        if (response.status === 404) {
          errorMessage = `Agent instance not found. Please re-initialize the agent. (${errorMessage})`;
          // Clear invalid instance
          setAgentInstance(null);
        }
        
        throw new Error(errorMessage);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullContent = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n').filter(line => line.trim());

          for (const line of lines) {
            try {
              const data = JSON.parse(line);
              
              if (data.type === 'token') {
                fullContent += data.content;
                setConversation(prev => {
                  const updated = [...prev];
                  const lastIdx = updated.length - 1;
                  if (updated[lastIdx]?.streaming) {
                    updated[lastIdx] = { ...updated[lastIdx], content: fullContent };
                  }
                  return updated;
                });
              } else if (data.type === 'error') {
                // Handle error messages from the stream
                const errorContent = data.content || 'An error occurred while processing your request.';
                fullContent = errorContent;
                setConversation(prev => {
                  const updated = [...prev];
                  const lastIdx = updated.length - 1;
                  if (updated[lastIdx]?.streaming) {
                    updated[lastIdx] = { 
                      ...updated[lastIdx], 
                      content: errorContent,
                      streaming: false,
                      isError: true,
                    };
                  }
                  return updated;
                });
                addEvent('agent_error', { error: errorContent });
                // Break on error to stop processing
                break;
              } else if (data.type === 'warning') {
                // Handle warning messages
                const warningContent = data.content || '';
                if (warningContent) {
                  fullContent += `\n[Warning: ${warningContent}]`;
                  setConversation(prev => {
                    const updated = [...prev];
                    const lastIdx = updated.length - 1;
                    if (updated[lastIdx]?.streaming) {
                      updated[lastIdx] = { ...updated[lastIdx], content: fullContent };
                    }
                    return updated;
                  });
                  addEvent('agent_warning', { warning: warningContent });
                }
              } else if (data.type === 'tool_call') {
                const toolCall: ToolCall = {
                  toolId: `tool-${Date.now()}`,
                  toolName: data.tool,
                  parameters: data.parameters,
                  status: 'executing',
                };
                setLiveToolCalls(prev => [...prev, toolCall]);
                addEvent('tool_called', { tool: data.tool });
              } else if (data.type === 'tool_result') {
                setLiveToolCalls(prev => 
                  prev.map(tc => 
                    tc.toolName === data.tool 
                      ? { ...tc, result: data.result, status: 'completed' }
                      : tc
                  )
                );
                addEvent('tool_completed', { tool: data.tool });
              } else if (data.type === 'done') {
                // Stream completed successfully
                addEvent('stream_done', { total_tokens: data.total_tokens });
              }
            } catch {
              // Non-JSON line, treat as token
              fullContent += line;
            }
          }
        }
      }

      // Finalize the message (only if not already finalized by error handler)
      setConversation(prev => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        if (updated[lastIdx]?.streaming) {
          // Only set fallback message if we have no content and it's not an error
          const finalContent = fullContent || 
            (updated[lastIdx]?.isError ? updated[lastIdx].content : 'I apologize, but I encountered an issue processing your request.');
          updated[lastIdx] = { 
            ...updated[lastIdx], 
            content: finalContent,
            streaming: false,
            toolCalls: liveToolCalls,
          };
        }
        return updated;
      });

      addEvent('response_complete', { length: fullContent.length });
      
      // Update metrics
      setAgentInstance(prev => prev ? {
        ...prev,
        status: 'idle',
        currentTask: undefined,
        metrics: {
          ...prev.metrics,
          tokensUsed: prev.metrics.tokensUsed + (fullContent.length / 4),
          responseTime: Date.now() - new Date(userMessage.timestamp).getTime(),
          toolCalls: prev.metrics.toolCalls + liveToolCalls.length,
        },
      } : null);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      addEvent('agent_error', { error: errorMessage });
      
      // Update conversation with error message
      setConversation(prev => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        if (updated[lastIdx]?.streaming) {
          updated[lastIdx] = {
            ...updated[lastIdx],
            content: `Error: ${errorMessage}`,
            streaming: false,
            isError: true,
          };
        }
        return updated;
      });

      setAgentInstance(prev => prev ? { ...prev, status: 'error' } : null);
    } finally {
      setIsStreaming(false);
    }
  };

  // Render configuration panel
  const renderConfigPanel = () => (
    <div className="config-panel">
      <h3><FaCog /> Agent Configuration</h3>
      
      <div className="config-section">
        <label>Agent Type</label>
        <div className="agent-type-grid">
          {AGENT_TYPES.map(type => {
            const IconComponent = type.icon;
            return (
              <button
                key={type.id}
                className={`agent-type-btn ${agentConfig.agentType === type.id ? 'active' : ''}`}
                onClick={() => setAgentConfig(prev => ({ ...prev, agentType: type.id }))}
              >
                <span className="icon"><IconComponent size={20} /></span>
                <span className="name">{type.name}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="config-section">
        <label>Agent Name</label>
        <input
          type="text"
          value={agentConfig.name}
          onChange={e => setAgentConfig(prev => ({ ...prev, name: e.target.value }))}
          placeholder="Enter agent name"
        />
      </div>

      <div className="config-section">
        <label>
          Model 
          <span className={`status-dot ${ollamaStatus}`} />
          {ollamaStatus === 'connected' && availableModels.length > 0 && (
            <span className="model-count">({availableModels.length} available)</span>
          )}
          <button 
            className="btn-icon btn-small" 
            onClick={() => {
              checkOllamaStatus();
              fetchAvailableModels();
            }}
            title="Refresh models from Ollama"
          >
            <FaSyncAlt />
          </button>
        </label>
        {ollamaStatus === 'disconnected' ? (
          <div className="error-message">
            <FaExclamationCircle /> Ollama is disconnected. Please check the Ollama container.
            <button 
              className="btn-secondary btn-small" 
              onClick={() => {
                checkOllamaStatus();
                fetchAvailableModels();
              }}
              style={{ marginLeft: '10px' }}
            >
              <FaSyncAlt /> Retry Connection
            </button>
          </div>
        ) : modelInfo.length === 0 ? (
          <div className="loading-models">
            <FaSpinner className="spin" /> Detecting models from Ollama...
          </div>
        ) : (
          <select
            value={agentConfig.model}
            onChange={e => setAgentConfig(prev => ({ ...prev, model: e.target.value }))}
          >
            {modelInfo.map(model => (
              <option key={model.id} value={model.id}>
                {model.name} {model.size && `(${model.size})`}
              </option>
            ))}
          </select>
        )}
      </div>

      <div className="config-row">
        <div className="config-section">
          <label>Temperature: {agentConfig.temperature}</label>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={agentConfig.temperature}
            onChange={e => setAgentConfig(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
          />
          <div className="temperature-info">
            <p className="temperature-description">
              LLM temperature controls text generation randomness, acting like a creativity dial: 
              <strong> low values (near 0)</strong> yield predictable, factual text (e.g., summarization), 
              while <strong>high values (above 1)</strong> produce diverse, creative, sometimes unexpected results (e.g., brainstorming), 
              influencing how the model samples from potential next words.
            </p>
          </div>
        </div>
        <div className="config-section">
          <label>Max Tokens: {agentConfig.maxTokens}</label>
          <input
            type="range"
            min="100"
            max="4000"
            step="100"
            value={agentConfig.maxTokens}
            onChange={e => setAgentConfig(prev => ({ ...prev, maxTokens: parseInt(e.target.value) }))}
          />
        </div>
      </div>

      <div className="config-section">
        <label>System Prompt</label>
        <textarea
          value={agentConfig.systemPrompt}
          onChange={e => setAgentConfig(prev => ({ ...prev, systemPrompt: e.target.value }))}
          placeholder="Enter system prompt..."
          rows={4}
        />
      </div>

      <div className="config-section">
        <label>Tools</label>
        <div className="tools-grid">
          {AVAILABLE_TOOLS.map(tool => (
            <label key={tool.id} className="tool-checkbox">
              <input
                type="checkbox"
                checked={agentConfig.tools.includes(tool.id)}
                onChange={e => {
                  if (e.target.checked) {
                    setAgentConfig(prev => ({ ...prev, tools: [...prev.tools, tool.id] }));
                  } else {
                    setAgentConfig(prev => ({ ...prev, tools: prev.tools.filter(t => t !== tool.id) }));
                  }
                }}
              />
              <span className="tool-name">{tool.name}</span>
              <span className="tool-category">{tool.category}</span>
            </label>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
        <button 
          className="btn-primary btn-large"
          onClick={initializeAgent}
          disabled={isLoading || ollamaStatus === 'checking'}
        >
          {isLoading ? <><FaSpinner className="spin" /> Initializing...</> : <><FaPlay /> Initialize Agent</>}
        </button>
        <button 
          className="btn-secondary btn-large"
          onClick={initializeMultipleAgents}
          disabled={isLoading || ollamaStatus === 'checking'}
        >
          <FaNetworkWired /> Initialize Multiple Test Agents
        </button>
      </div>
    </div>
  );

  // Render test panel
  const renderTestPanel = () => (
    <div className="test-panel">
      <div className="agent-status-bar">
        <div className="agent-info">
          <FaRobot />
          <span className="agent-name">{agentConfig.name}</span>
          <span className={`status-badge ${agentInstance?.status}`}>
            {agentInstance?.status}
          </span>
        </div>
        <div className="agent-metrics">
          <span><FaBrain /> {Math.round(agentInstance?.metrics.tokensUsed || 0)} tokens</span>
          <span><FaTools /> {agentInstance?.metrics.toolCalls || 0} tool calls</span>
          <span><TimerIcon size={16} /> {Math.round((agentInstance?.metrics.responseTime || 0) / 1000)}s avg</span>
        </div>
        <button className="btn-danger btn-small" onClick={stopAgent}>
          <FaStop /> Stop Agent
        </button>
      </div>

      <div className="conversation-container">
        <div className="messages">
          {conversation.length === 0 && (
            <div className="empty-state">
              <FaRobot size={48} />
              <h3>Agent Ready</h3>
              <p>Send a message to start interacting with the agent</p>
            </div>
          )}
          {conversation.map(msg => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="message-header">
                <span className="role">{msg.role}</span>
                <span className="timestamp">{new Date(msg.timestamp).toLocaleTimeString()}</span>
              </div>
              <div className="message-content">
                {msg.content}
                {msg.streaming && <span className="cursor">▊</span>}
              </div>
              {msg.toolCalls && msg.toolCalls.length > 0 && (
                <div className="tool-calls">
                  {msg.toolCalls.map(tc => (
                    <div key={tc.toolId} className={`tool-call ${tc.status}`}>
                      <FaTools /> {tc.toolName}
                      {tc.status === 'completed' && <FaCheckCircle className="success" />}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {liveToolCalls.length > 0 && (
          <div className="live-tools">
            <h4><FaTools /> Active Tools</h4>
            {liveToolCalls.map(tc => (
              <div key={tc.toolId} className={`live-tool ${tc.status}`}>
                <span className="tool-name">{tc.toolName}</span>
                {tc.status === 'executing' && <FaSpinner className="spin" />}
                {tc.status === 'completed' && <FaCheckCircle className="success" />}
                {tc.status === 'failed' && <FaExclamationCircle className="error" />}
              </div>
            ))}
          </div>
        )}

        <div className="input-container">
          <input
            type="text"
            value={userInput}
            onChange={e => setUserInput(e.target.value)}
            onKeyPress={e => e.key === 'Enter' && !isStreaming && sendMessage()}
            placeholder="Type a message..."
            disabled={isStreaming}
          />
          <button 
            className="btn-primary"
            onClick={sendMessage}
            disabled={isStreaming || !userInput.trim()}
          >
            {isStreaming ? <FaSpinner className="spin" /> : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );

  // Render monitor panel
  const renderMonitorPanel = () => (
    <div className="monitor-panel">
      <div className="monitor-header">
        <h3><FaChartLine /> Agent Monitor</h3>
        <button className="btn-secondary btn-small" onClick={() => setEvents([])}>
          Clear Events
        </button>
      </div>

      <div className="monitor-grid">
        <div className="monitor-section events">
          <h4><FaTerminal /> Event Log</h4>
          <div className="event-list">
            {events.slice().reverse().map(event => (
              <div key={event.eventId} className={`event ${event.eventType}`}>
                <span className="event-time">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </span>
                <span className="event-type">{event.eventType}</span>
                <span className="event-data">
                  {JSON.stringify(event.data).substring(0, 100)}
                </span>
              </div>
            ))}
            {events.length === 0 && (
              <div className="empty-events">No events yet</div>
            )}
          </div>
        </div>

        <div className="monitor-section metrics">
          <h4><FaChartLine /> Metrics</h4>
          <div className="metrics-grid">
            <div className="metric-card">
              <span className="metric-label">Status</span>
              <span className={`metric-value status-${agentInstance?.status || 'idle'}`}>
                {agentInstance?.status || 'Not Started'}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Tokens Used</span>
              <span className="metric-value">
                {Math.round(agentInstance?.metrics.tokensUsed || 0)}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Tool Calls</span>
              <span className="metric-value">
                {agentInstance?.metrics.toolCalls || 0}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Messages</span>
              <span className="metric-value">{conversation.length}</span>
            </div>
          </div>
        </div>

        <div className="monitor-section state">
          <h4><FaNetworkWired /> Agent State</h4>
          <pre className="state-json">
            {JSON.stringify({
              instance: agentInstance,
              config: agentConfig,
              conversationLength: conversation.length,
            }, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );

  return (
    <div className="agent-playground">
      <div className="playground-header">
        <h2><FaRobot /> Agent Playground</h2>
        <div className="header-status">
          <span className={`ollama-status ${ollamaStatus}`}>
            {ollamaStatus === 'connected' && <><FaCheckCircle /> Ollama Connected</>}
            {ollamaStatus === 'disconnected' && <><FaExclamationCircle /> Ollama Disconnected</>}
            {ollamaStatus === 'checking' && <><FaSpinner className="spin" /> Checking...</>}
          </span>
          <button className="btn-icon" onClick={checkOllamaStatus}>
            <FaSyncAlt />
          </button>
        </div>
      </div>

      <div className="playground-tabs">
        <button
          className={`tab ${activeTab === 'configure' ? 'active' : ''}`}
          onClick={() => setActiveTab('configure')}
        >
          <FaCog /> Configure
        </button>
        <button
          className={`tab ${activeTab === 'test' ? 'active' : ''}`}
          onClick={() => setActiveTab('test')}
          disabled={!agentInstance}
        >
          <FaPlay /> Test
        </button>
        <button
          className={`tab ${activeTab === 'monitor' ? 'active' : ''}`}
          onClick={() => setActiveTab('monitor')}
        >
          <FaChartLine /> Monitor
        </button>
      </div>

      <div className="playground-content">
        {activeTab === 'configure' && renderConfigPanel()}
        {activeTab === 'test' && renderTestPanel()}
        {activeTab === 'monitor' && renderMonitorPanel()}
      </div>
    </div>
  );
}

