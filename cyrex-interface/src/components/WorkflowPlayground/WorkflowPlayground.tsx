/**
 * LangGraph Workflow Playground Component
 * 
 * Features:
 * 1. Test LangGraph multi-agent workflows
 * 2. Visualize workflow execution in real-time
 * 3. Test different workflow types (standard, lease, contract, fraud)
 * 4. View agent history and state transitions
 * 5. Monitor workflow metrics
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  FaProjectDiagram,
  FaPlay,
  FaStop,
  FaSpinner,
  FaCog,
  FaCheckCircle,
  FaExclamationCircle,
  FaCode,
  FaFileContract,
  FaFileInvoice,
  FaShieldAlt,
  FaNetworkWired,
  FaChartLine,
  FaHistory,
  FaEye,
  FaDownload,
  FaSyncAlt,
  FaRobot,
  FaTasks,
  FaLightbulb,
  FaCheckSquare,
} from 'react-icons/fa';
import './WorkflowPlayground.css';

// Types
interface WorkflowConfig {
  workflowType: 'standard' | 'lease' | 'contract' | 'fraud' | 'custom';
  taskDescription: string;
  context: Record<string, any>;
  sessionId?: string;
  userId?: string;
}

interface WorkflowState {
  workflow_id: string;
  workflow_type: string;
  task_type: string;
  current_agent: string;
  task_description: string;
  plan?: string;
  code?: string;
  quality_check?: string;
  result?: any;
  agent_history: Array<{
    agent: string;
    role: string;
    response?: string;
    result?: string;
    confidence?: number;
    timestamp: string;
  }>;
  tool_calls: Array<any>;
  errors: string[];
  messages: Array<any>;
  metadata: Record<string, any>;
}

interface AgentNode {
  id: string;
  label: string;
  type: string;
  status: 'pending' | 'active' | 'completed' | 'error' | 'skipped';
  startTime?: string;
  endTime?: string;
  duration?: number;
  result?: any;
}

const API_BASE = import.meta.env.VITE_CYREX_BASE_URL || 'http://localhost:8000';

// Workflow type templates
const WORKFLOW_TEMPLATES = {
  standard: {
    name: 'Standard Workflow',
    icon: <FaTasks />,
    description: 'Task → Plan → Code → Quality',
    defaultTask: 'Create a Python function to calculate fibonacci numbers',
  },
  lease: {
    name: 'Lease Abstraction',
    icon: <FaFileContract />,
    description: 'Process lease documents and extract structured data',
    defaultTask: 'Process this lease document and extract key terms',
  },
  contract: {
    name: 'Contract Intelligence',
    icon: <FaFileInvoice />,
    description: 'Analyze contracts, track clause evolution, build dependency graphs',
    defaultTask: 'Process this contract and identify all clauses and obligations',
  },
  fraud: {
    name: 'Vendor Fraud Detection',
    icon: <FaShieldAlt />,
    description: 'Detect vendor fraud, analyze invoices, benchmark pricing',
    defaultTask: 'Analyze this invoice for potential fraud indicators',
  },
  custom: {
    name: 'Custom Workflow',
    icon: <FaNetworkWired />,
    description: 'Custom workflow with manual configuration',
    defaultTask: '',
  },
};

export function WorkflowPlayground() {
  // State
  const [activeTab, setActiveTab] = useState<'configure' | 'execute' | 'visualize' | 'monitor'>('configure');
  const [workflowConfig, setWorkflowConfig] = useState<WorkflowConfig>({
    workflowType: 'standard',
    taskDescription: '',
    context: {},
  });
  const [workflowState, setWorkflowState] = useState<WorkflowState | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionHistory, setExecutionHistory] = useState<WorkflowState[]>([]);
  const [agentNodes, setAgentNodes] = useState<AgentNode[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [streamingMessages, setStreamingMessages] = useState<string[]>([]);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [streamingMessages, workflowState]);

  // Load template when workflow type changes
  useEffect(() => {
    const template = WORKFLOW_TEMPLATES[workflowConfig.workflowType];
    if (template && !workflowConfig.taskDescription) {
      setWorkflowConfig(prev => ({
        ...prev,
        taskDescription: template.defaultTask,
      }));
    }
  }, [workflowConfig.workflowType]);

  // Execute workflow
  const executeWorkflow = async () => {
    if (!workflowConfig.taskDescription.trim()) {
      alert('Please provide a task description');
      return;
    }

    setIsExecuting(true);
    setStreamingMessages([]);
    setWorkflowState(null);
    setAgentNodes([]);

    try {
      const workflowId = `workflow_${Date.now()}`;
      
      const response = await fetch(`${API_BASE}/api/workflow/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_description: workflowConfig.taskDescription,
          workflow_type: workflowConfig.workflowType,
          context: workflowConfig.context,
          session_id: workflowConfig.sessionId,
          user_id: workflowConfig.userId,
          workflow_id: workflowId,
          stream: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle streaming response
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.trim()) continue;
            
            try {
              const data = JSON.parse(line);
              
              if (data.type === 'state_update') {
                setWorkflowState(data.state);
                updateAgentNodes(data.state);
              } else if (data.type === 'message') {
                setStreamingMessages(prev => [...prev, data.content]);
              } else if (data.type === 'agent_start') {
                updateNodeStatus(data.agent, 'active', data);
              } else if (data.type === 'agent_complete') {
                updateNodeStatus(data.agent, 'completed', data);
              } else if (data.type === 'error') {
                updateNodeStatus(data.agent || 'unknown', 'error', data);
              } else if (data.type === 'done') {
                // Final state
                if (data.final_state) {
                  setWorkflowState(data.final_state);
                  updateAgentNodes(data.final_state);
                  setExecutionHistory(prev => [...prev, data.final_state]);
                }
              }
            } catch (e) {
              // Non-JSON line, treat as message
              if (line.trim()) {
                setStreamingMessages(prev => [...prev, line]);
              }
            }
          }
        }
      }

      setActiveTab('visualize');
    } catch (error: any) {
      console.error('Workflow execution error:', error);
      alert(`Error executing workflow: ${error.message}`);
    } finally {
      setIsExecuting(false);
    }
  };

  // Stop workflow
  const stopWorkflow = async () => {
    if (workflowState) {
      try {
        await fetch(`${API_BASE}/api/workflow/${workflowState.workflow_id}/stop`, {
          method: 'POST',
        });
      } catch (error) {
        console.error('Error stopping workflow:', error);
      }
    }
    setIsExecuting(false);
    setStreamingMessages([]);
  };

  // Update agent nodes from workflow state
  const updateAgentNodes = (state: WorkflowState) => {
    const nodes: AgentNode[] = [];
    
    // Task router
    nodes.push({
      id: 'task_router',
      label: 'Task Router',
      type: 'router',
      status: state.current_agent === 'task_router' ? 'active' : 
              state.workflow_type ? 'completed' : 'pending',
    });

    // Standard workflow nodes
    if (state.workflow_type === 'standard') {
      nodes.push(
        {
          id: 'task_agent',
          label: 'Task Decomposer',
          type: 'agent',
          status: state.current_agent === 'task_agent' ? 'active' :
                  state.metadata?.task_breakdown ? 'completed' : 'pending',
        },
        {
          id: 'plan_agent',
          label: 'Planning Agent',
          type: 'agent',
          status: state.current_agent === 'plan_agent' ? 'active' :
                  state.plan ? 'completed' : state.metadata?.task_breakdown ? 'pending' : 'skipped',
        },
        {
          id: 'code_agent',
          label: 'Code Generator',
          type: 'agent',
          status: state.current_agent === 'code_agent' ? 'active' :
                  state.code ? 'completed' : state.plan ? 'pending' : 'skipped',
        },
        {
          id: 'qa_agent',
          label: 'Quality Assurance',
          type: 'agent',
          status: state.current_agent === 'qa_agent' ? 'active' :
                  state.quality_check ? 'completed' : state.code ? 'pending' : 'skipped',
        }
      );
    }

    // Specialized workflow nodes
    if (state.workflow_type === 'lease') {
      nodes.push({
        id: 'lease_processor',
        label: 'Lease Processor',
        type: 'processor',
        status: state.current_agent === 'lease_processor' ? 'active' :
                state.result ? 'completed' : 'pending',
        result: state.result,
      });
    }

    if (state.workflow_type === 'contract') {
      nodes.push({
        id: 'contract_processor',
        label: 'Contract Processor',
        type: 'processor',
        status: state.current_agent === 'contract_processor' ? 'active' :
                state.result ? 'completed' : 'pending',
        result: state.result,
      });
    }

    if (state.workflow_type === 'fraud') {
      nodes.push({
        id: 'fraud_agent',
        label: 'Fraud Detector',
        type: 'agent',
        status: state.current_agent === 'fraud_agent' ? 'active' :
                state.result ? 'completed' : 'pending',
        result: state.result,
      });
    }

    setAgentNodes(nodes);
  };

  // Update node status
  const updateNodeStatus = (agentId: string, status: AgentNode['status'], data?: any) => {
    setAgentNodes(prev => prev.map(node => {
      if (node.id === agentId) {
        return {
          ...node,
          status,
          startTime: status === 'active' ? new Date().toISOString() : node.startTime,
          endTime: status === 'completed' || status === 'error' ? new Date().toISOString() : node.endTime,
          result: data?.result || node.result,
        };
      }
      return node;
    }));
  };

  // Render configuration panel
  const renderConfigPanel = () => (
    <div className="workflow-config-panel">
      <h3><FaCog /> Workflow Configuration</h3>
      
      <div className="config-section">
        <label>Workflow Type</label>
        <div className="workflow-type-grid">
          {Object.entries(WORKFLOW_TEMPLATES).map(([key, template]) => (
            <button
              key={key}
              className={`workflow-type-btn ${workflowConfig.workflowType === key ? 'active' : ''}`}
              onClick={() => setWorkflowConfig(prev => ({ ...prev, workflowType: key as any }))}
            >
              <span className="icon">{template.icon}</span>
              <span className="name">{template.name}</span>
              <span className="description">{template.description}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="config-section">
        <label>Task Description</label>
        <textarea
          value={workflowConfig.taskDescription}
          onChange={e => setWorkflowConfig(prev => ({ ...prev, taskDescription: e.target.value }))}
          placeholder="Enter task description..."
          rows={6}
        />
      </div>

      {workflowConfig.workflowType === 'lease' && (
        <div className="config-section">
          <label>Lease Context</label>
          <textarea
            value={JSON.stringify(workflowConfig.context, null, 2)}
            onChange={e => {
              try {
                const context = JSON.parse(e.target.value);
                setWorkflowConfig(prev => ({ ...prev, context }));
              } catch {
                // Invalid JSON, ignore
              }
            }}
            placeholder='{"document_text": "...", "document_url": "...", "lease_id": "..."}'
            rows={4}
          />
        </div>
      )}

      {workflowConfig.workflowType === 'contract' && (
        <div className="config-section">
          <label>Contract Context</label>
          <textarea
            value={JSON.stringify(workflowConfig.context, null, 2)}
            onChange={e => {
              try {
                const context = JSON.parse(e.target.value);
                setWorkflowConfig(prev => ({ ...prev, context }));
              } catch {
                // Invalid JSON, ignore
              }
            }}
            placeholder='{"document_text": "...", "document_url": "...", "contract_id": "..."}'
            rows={4}
          />
        </div>
      )}

      <div className="config-row">
        <div className="config-section">
          <label>Session ID (Optional)</label>
          <input
            type="text"
            value={workflowConfig.sessionId || ''}
            onChange={e => setWorkflowConfig(prev => ({ ...prev, sessionId: e.target.value || undefined }))}
            placeholder="session_123"
          />
        </div>
        <div className="config-section">
          <label>User ID (Optional)</label>
          <input
            type="text"
            value={workflowConfig.userId || ''}
            onChange={e => setWorkflowConfig(prev => ({ ...prev, userId: e.target.value || undefined }))}
            placeholder="user_123"
          />
        </div>
      </div>

      <button
        className="btn-primary btn-large"
        onClick={executeWorkflow}
        disabled={isExecuting || !workflowConfig.taskDescription.trim()}
      >
        {isExecuting ? (
          <>
            <FaSpinner className="spin" /> Executing Workflow...
          </>
        ) : (
          <>
            <FaPlay /> Execute Workflow
          </>
        )}
      </button>
    </div>
  );

  // Render execution panel
  const renderExecutePanel = () => (
    <div className="workflow-execute-panel">
      <div className="workflow-status-bar">
        <div className="workflow-info">
          <FaProjectDiagram />
          <span className="workflow-type">
            {WORKFLOW_TEMPLATES[workflowConfig.workflowType as keyof typeof WORKFLOW_TEMPLATES]?.name}
          </span>
          {workflowState && (
            <>
              <span className={`status-badge ${isExecuting ? 'executing' : 'completed'}`}>
                {isExecuting ? 'Executing' : 'Completed'}
              </span>
              <span className="workflow-id">{workflowState.workflow_id}</span>
            </>
          )}
        </div>
        {isExecuting && (
          <button className="btn-danger btn-small" onClick={stopWorkflow}>
            <FaStop /> Stop
          </button>
        )}
      </div>

      <div className="execution-container">
        <div className="execution-messages">
          {streamingMessages.length === 0 && !workflowState && (
            <div className="empty-state">
              <FaProjectDiagram size={48} />
              <h3>Ready to Execute</h3>
              <p>Configure workflow and click "Execute Workflow" to start</p>
            </div>
          )}
          
          {streamingMessages.map((msg, idx) => (
            <div key={idx} className="execution-message">
              {msg}
            </div>
          ))}
          
          {workflowState && (
            <div className="workflow-state-summary">
              <h4>Workflow Summary</h4>
              <div className="summary-grid">
                <div className="summary-item">
                  <span className="label">Workflow Type:</span>
                  <span className="value">{workflowState.workflow_type}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Task Type:</span>
                  <span className="value">{workflowState.task_type || 'N/A'}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Current Agent:</span>
                  <span className="value">{workflowState.current_agent || 'N/A'}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Agents Used:</span>
                  <span className="value">{workflowState.agent_history.length}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Tool Calls:</span>
                  <span className="value">{workflowState.tool_calls.length}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Errors:</span>
                  <span className={`value ${workflowState.errors.length > 0 ? 'error' : ''}`}>
                    {workflowState.errors.length}
                  </span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>
    </div>
  );

  // Render visualization panel
  const renderVisualizePanel = () => (
    <div className="workflow-visualize-panel">
      <div className="visualization-header">
        <h3><FaNetworkWired /> Workflow Visualization</h3>
        {workflowState && (
          <button
            className="btn-secondary btn-small"
            onClick={() => {
              const data = JSON.stringify(workflowState, null, 2);
              const blob = new Blob([data], { type: 'application/json' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `workflow_${workflowState.workflow_id}.json`;
              a.click();
            }}
          >
            <FaDownload /> Export State
          </button>
        )}
      </div>

      {agentNodes.length === 0 && !workflowState && (
        <div className="empty-state">
          <FaNetworkWired size={48} />
          <h3>No Workflow Executed</h3>
          <p>Execute a workflow to see the visualization</p>
        </div>
      )}

      {agentNodes.length > 0 && (
        <div className="workflow-graph">
          <div className="agent-nodes">
            {agentNodes.map((node, idx) => (
              <div
                key={node.id}
                className={`agent-node ${node.status} ${selectedNode === node.id ? 'selected' : ''}`}
                onClick={() => setSelectedNode(selectedNode === node.id ? null : node.id)}
              >
                <div className="node-icon">
                  {node.type === 'router' && <FaNetworkWired />}
                  {node.type === 'agent' && <FaRobot />}
                  {node.type === 'processor' && <FaFileContract />}
                </div>
                <div className="node-label">{node.label}</div>
                <div className={`node-status ${node.status}`}>
                  {node.status === 'active' && <FaSpinner className="spin" />}
                  {node.status === 'completed' && <FaCheckCircle />}
                  {node.status === 'error' && <FaExclamationCircle />}
                  {node.status === 'skipped' && <span>⏭️</span>}
                  {node.status === 'pending' && <span>⏳</span>}
                </div>
                {idx < agentNodes.length - 1 && (
                  <div className="node-arrow">↓</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {selectedNode && workflowState && (
        <div className="node-details">
          <h4>Node Details: {agentNodes.find(n => n.id === selectedNode)?.label}</h4>
          <div className="details-content">
            {(() => {
              const node = agentNodes.find(n => n.id === selectedNode);
              const history = workflowState.agent_history.find(h => h.agent === selectedNode);
              
              return (
                <>
                  <div className="detail-item">
                    <span className="label">Status:</span>
                    <span className={`value status-${node?.status}`}>{node?.status}</span>
                  </div>
                  {history && (
                    <>
                      <div className="detail-item">
                        <span className="label">Role:</span>
                        <span className="value">{history.role}</span>
                      </div>
                      {history.response && (
                        <div className="detail-item">
                          <span className="label">Response:</span>
                          <div className="value response-preview">{history.response.substring(0, 200)}...</div>
                        </div>
                      )}
                      {history.confidence && (
                        <div className="detail-item">
                          <span className="label">Confidence:</span>
                          <span className="value">{(history.confidence * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      <div className="detail-item">
                        <span className="label">Timestamp:</span>
                        <span className="value">{new Date(history.timestamp).toLocaleString()}</span>
                      </div>
                    </>
                  )}
                  {node?.result && (
                    <div className="detail-item">
                      <span className="label">Result:</span>
                      <pre className="value result-json">
                        {JSON.stringify(node.result, null, 2)}
                      </pre>
                    </div>
                  )}
                </>
              );
            })()}
          </div>
        </div>
      )}

      {workflowState && (
        <div className="workflow-results">
          <h4>Workflow Results</h4>
          {workflowState.result && (
            <div className="result-section">
              <h5>Final Result</h5>
              <pre className="result-json">
                {JSON.stringify(workflowState.result, null, 2)}
              </pre>
            </div>
          )}
          {workflowState.plan && (
            <div className="result-section">
              <h5>Plan</h5>
              <div className="result-text">{workflowState.plan}</div>
            </div>
          )}
          {workflowState.code && (
            <div className="result-section">
              <h5>Generated Code</h5>
              <pre className="result-code">{workflowState.code}</pre>
            </div>
          )}
          {workflowState.quality_check && (
            <div className="result-section">
              <h5>Quality Check</h5>
              <div className="result-text">{workflowState.quality_check}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );

  // Render monitor panel
  const renderMonitorPanel = () => (
    <div className="workflow-monitor-panel">
      <div className="monitor-header">
        <h3><FaChartLine /> Workflow Monitor</h3>
        <button
          className="btn-secondary btn-small"
          onClick={() => setExecutionHistory([])}
        >
          Clear History
        </button>
      </div>

      <div className="monitor-grid">
        <div className="monitor-section metrics">
          <h4><FaChartLine /> Metrics</h4>
          <div className="metrics-grid">
            <div className="metric-card">
              <span className="metric-label">Workflow Type</span>
              <span className="metric-value">
                {workflowState?.workflow_type || 'N/A'}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Agents Used</span>
              <span className="metric-value">
                {workflowState?.agent_history.length || 0}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Tool Calls</span>
              <span className="metric-value">
                {workflowState?.tool_calls.length || 0}
              </span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Errors</span>
              <span className={`metric-value ${(workflowState?.errors.length || 0) > 0 ? 'error' : ''}`}>
                {workflowState?.errors.length || 0}
              </span>
            </div>
          </div>
        </div>

        <div className="monitor-section history">
          <h4><FaHistory /> Agent History</h4>
          <div className="history-list">
            {workflowState?.agent_history.map((entry, idx) => (
              <div key={idx} className="history-item">
                <div className="history-header">
                  <span className="agent-name">{entry.agent}</span>
                  <span className="agent-role">{entry.role}</span>
                  <span className="history-time">
                    {new Date(entry.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                {entry.response && (
                  <div className="history-response">
                    {entry.response.substring(0, 150)}...
                  </div>
                )}
                {entry.confidence && (
                  <div className="history-confidence">
                    Confidence: {(entry.confidence * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            ))}
            {(!workflowState || workflowState.agent_history.length === 0) && (
              <div className="empty-history">No agent history yet</div>
            )}
          </div>
        </div>

        <div className="monitor-section state">
          <h4><FaEye /> Workflow State</h4>
          <pre className="state-json">
            {JSON.stringify(workflowState, null, 2)}
          </pre>
        </div>

        <div className="monitor-section errors">
          <h4><FaExclamationCircle /> Errors</h4>
          <div className="errors-list">
            {workflowState?.errors.map((error, idx) => (
              <div key={idx} className="error-item">
                {error}
              </div>
            ))}
            {(!workflowState || workflowState.errors.length === 0) && (
              <div className="no-errors">No errors</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="workflow-playground">
      <div className="playground-header">
        <h2><FaProjectDiagram /> LangGraph Workflow Playground</h2>
        <div className="header-actions">
          {workflowState && (
            <span className="workflow-id-badge">
              ID: {workflowState.workflow_id}
            </span>
          )}
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
          className={`tab ${activeTab === 'execute' ? 'active' : ''}`}
          onClick={() => setActiveTab('execute')}
        >
          <FaPlay /> Execute
        </button>
        <button
          className={`tab ${activeTab === 'visualize' ? 'active' : ''}`}
          onClick={() => setActiveTab('visualize')}
        >
          <FaNetworkWired /> Visualize
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
        {activeTab === 'execute' && renderExecutePanel()}
        {activeTab === 'visualize' && renderVisualizePanel()}
        {activeTab === 'monitor' && renderMonitorPanel()}
      </div>
    </div>
  );
}

