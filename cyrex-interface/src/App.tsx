import { useCallback, useMemo, useState } from 'react';

type ChatMessage = {
  role: 'user' | 'assistant' | 'system';
  content: string;
  meta?: Record<string, unknown>;
};

const defaultUserProfile = {
  role: 'software_engineer',
  momentum: 420,
  level: 14,
  active_boosts: [],
  recent_activities: ['completed_task', 'code_reviewed']
};

const defaultProjectContext = {
  project: 'cyrex-interface',
  repo: 'diri-cyrex',
  tags: ['testing', 'langchain', 'workflow']
};

const pretty = (data: unknown) => JSON.stringify(data, null, 2);

const defaultBaseUrl = import.meta.env.VITE_CYREX_BASE_URL ?? 'http://localhost:8000';

export default function App() {
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl);
  const [apiKey, setApiKey] = useState('');
  const [routeCommand, setRouteCommand] = useState('Generate a quick summary of the auth logs.');
  const [routeResult, setRouteResult] = useState('');
  const [abilityCommand, setAbilityCommand] = useState('Plan a TypeScript refactor for the realtime gateway.');
  const [abilityResult, setAbilityResult] = useState('');
  const [workflowPayload, setWorkflowPayload] = useState(
    pretty({
      momentum: 420,
      current_level: 14,
      task_completion_rate: 0.82,
      daily_streak: 9,
      recent_efficiency: 0.78
    })
  );
  const [workflowResult, setWorkflowResult] = useState('');
  const [knowledgeQuery, setKnowledgeQuery] = useState('focus boost strategies');
  const [knowledgeResult, setKnowledgeResult] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
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
    async (path: string, payload: Record<string, unknown>) => {
      setLoadingKey(path);
      setError(null);
      try {
        const res = await fetch(`${baseUrl}${path}`, {
          method: 'POST',
          headers,
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`${res.status} ${res.statusText}: ${text}`);
        }
        const json = await res.json();
        return json;
      } catch (err: any) {
        setError(err?.message ?? 'Unknown error');
        throw err;
      } finally {
        setLoadingKey(null);
      }
    },
    [baseUrl, headers]
  );

  const runRouteCommand = async () => {
    const payload = {
      command: routeCommand,
      user_role: defaultUserProfile.role,
      context: {
        source: 'cyrex-interface',
        timestamp: new Date().toISOString()
      },
      min_confidence: 0.5,
      top_k: 3
    };
    const result = await callEndpoint('/agent/intelligence/route-command', payload);
    setRouteResult(pretty(result));
  };

  const runAbilityGenerator = async () => {
    const payload = {
      user_id: 'cyrex-interface',
      user_command: abilityCommand,
      user_profile: defaultUserProfile,
      project_context: defaultProjectContext
    };
    const result = await callEndpoint('/agent/intelligence/generate-ability', payload);
    setAbilityResult(pretty(result));
  };

  const runWorkflowRecommendation = async () => {
    const payload = {
      user_data: JSON.parse(workflowPayload)
    };
    const result = await callEndpoint('/agent/intelligence/recommend-action', payload);
    setWorkflowResult(pretty(result));
  };

  const runKnowledgeQuery = async () => {
    const payload = {
      query: knowledgeQuery,
      knowledge_bases: ['user_patterns', 'ability_templates', 'project_context'],
      top_k: 5
    };
    const result = await callEndpoint('/agent/intelligence/knowledge/query', payload);
    setKnowledgeResult(pretty(result));
  };

  const handleChatSend = async () => {
    if (!chatInput.trim()) return;
    const newMessage: ChatMessage = {
      role: 'user',
      content: chatInput.trim()
    };
    setChatHistory((prev) => [...prev, newMessage]);
    setChatInput('');
    try {
      const payload = {
        user_id: 'visual-chat',
        user_command: newMessage.content,
        user_profile: defaultUserProfile,
        project_context: defaultProjectContext,
        chat_history: chatHistory
          .concat(newMessage)
          .slice(-6)
          .map((entry) => ({
            role: entry.role,
            content: entry.content
          }))
      };
      const response = await callEndpoint('/agent/intelligence/generate-ability', payload);
      const content =
        response?.data?.description ??
        response?.data?.ability_name ??
        'No response returned. Check logs for details.';
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content,
        meta: response?.data
      };
      setChatHistory((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      setChatHistory((prev) => [
        ...prev,
        {
          role: 'system',
          content: `⚠️ ${err?.message ?? 'Failed to reach Cyrex'}`
        }
      ]);
    }
  };

  const disabled = (key: string) => loadingKey === key;

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Cyrex Interface</h1>
        <p>Manually exercise the intelligence API endpoints with a friendly UI.</p>
        <span className="build-tag">Build: {__BUILD_TIME__}</span>
      </header>

      <section className="panel">
        <h2>Connection</h2>
        <div className="field-grid">
          <label>
            Base URL
            <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="http://localhost:8000" />
          </label>
          <label>
            API Key (x-api-key)
            <input value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="change-me" />
          </label>
        </div>
        {error && <p className="error">Last error: {error}</p>}
      </section>

      <section className="panel">
        <h2>Chat Console</h2>
        <div className="chat-box">
          {chatHistory.length === 0 && <p className="empty-state">No messages yet. Ask Cyrex to plan something.</p>}
          {chatHistory.map((msg, idx) => (
            <div key={`${msg.role}-${idx}`} className={`chat-line ${msg.role}`}>
              <strong>{msg.role.toUpperCase()}</strong>
              <p>{msg.content}</p>
              {msg.meta && (
                <details>
                  <summary>Details</summary>
                  <pre>{pretty(msg.meta)}</pre>
                </details>
              )}
            </div>
          ))}
        </div>
        <div className="chat-input">
          <input
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder="“Help me design a focused debug sprint.”"
            onKeyDown={(e) => e.key === 'Enter' && handleChatSend()}
          />
          <button onClick={handleChatSend} disabled={disabled('/agent/intelligence/generate-ability')}>
            Send
          </button>
        </div>
      </section>

      <div className="grid">
        <section className="panel">
          <h3>Command Routing</h3>
          <textarea value={routeCommand} onChange={(e) => setRouteCommand(e.target.value)} />
          <button onClick={runRouteCommand} disabled={disabled('/agent/intelligence/route-command')}>
            Call /route-command
          </button>
          <pre>{routeResult || 'No response yet.'}</pre>
        </section>

        <section className="panel">
          <h3>Ability Generator</h3>
          <textarea value={abilityCommand} onChange={(e) => setAbilityCommand(e.target.value)} />
          <button onClick={runAbilityGenerator} disabled={disabled('/agent/intelligence/generate-ability')}>
            Call /generate-ability
          </button>
          <pre>{abilityResult || 'No response yet.'}</pre>
        </section>

        <section className="panel">
          <h3>Workflow Optimizer</h3>
          <textarea value={workflowPayload} onChange={(e) => setWorkflowPayload(e.target.value)} />
          <button onClick={runWorkflowRecommendation} disabled={disabled('/agent/intelligence/recommend-action')}>
            Call /recommend-action
          </button>
          <pre>{workflowResult || 'No response yet.'}</pre>
        </section>

        <section className="panel">
          <h3>Knowledge Retrieval</h3>
          <input value={knowledgeQuery} onChange={(e) => setKnowledgeQuery(e.target.value)} />
          <button onClick={runKnowledgeQuery} disabled={disabled('/agent/intelligence/knowledge/query')}>
            Call /knowledge/query
          </button>
          <pre>{knowledgeResult || 'No response yet.'}</pre>
        </section>
      </div>

      <section className="panel">
        <h2>Test & Lint Commands</h2>
        <ul>
          <li>
            <code>pytest</code> – run server-side tests (from <code>diri-cyrex</code>)
          </li>
          <li>
            <code>mypy app</code> – static type checks for the FastAPI service
          </li>
          <li>
            <code>npm run lint</code> – lint this interface UI
          </li>
        </ul>
      </section>
    </div>
  );
}

