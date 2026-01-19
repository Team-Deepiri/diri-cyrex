/**
 * LinkedIn-style Messages Widget
 * Can be opened from anywhere in the interface
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FaComments, FaTimes, FaPlus, FaUsers, FaRobot, FaChevronDown, FaChevronUp } from 'react-icons/fa';
import './MessagesWidget.css';

const API_BASE = import.meta.env.VITE_CYREX_BASE_URL || 'http://localhost:8000';

interface AgentChat {
  instanceId: string;
  agentId: string;
  name: string;
  lastMessage?: string;
  lastMessageTime?: string;
  unreadCount?: number;
}

interface GroupChat {
  groupChatId: string;
  name: string;
  agentCount: number;
  lastMessage?: string;
  lastMessageTime?: string;
}

interface MessagesWidgetProps {
  isOpen: boolean;
  onClose: () => void;
}

interface AgentType {
  id: string;
  name: string;
  description: string;
  template_key: string;
  temperature?: number;
  max_tokens?: number;
  tools?: string[];
}

export function MessagesWidget({ isOpen, onClose }: MessagesWidgetProps) {
  const [activeView, setActiveView] = useState<'list' | 'chat' | 'group-chat'>('list');
  const [agentChats, setAgentChats] = useState<AgentChat[]>([]);
  const [groupChats, setGroupChats] = useState<GroupChat[]>([]);
  const [selectedChat, setSelectedChat] = useState<AgentChat | null>(null);
  const [selectedGroupChat, setSelectedGroupChat] = useState<GroupChat | null>(null);
  const [messages, setMessages] = useState<any[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [agentTypes, setAgentTypes] = useState<AgentType[]>([]);
  const [selectedAgentType, setSelectedAgentType] = useState<string>('conversational');
  const [newAgentName, setNewAgentName] = useState('');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('llama3:8b');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch agent chats
  useEffect(() => {
    if (isOpen && activeView === 'list') {
      fetchAgentChats();
      fetchGroupChats();
      fetchAgentTypes();
      fetchModels();
    }
  }, [isOpen, activeView]);

  const fetchAgentTypes = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/agent/agent-types`);
      if (response.ok) {
        const data = await response.json();
        setAgentTypes(data.agent_types || []);
      }
    } catch (error) {
      console.error('Failed to fetch agent types:', error);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/agent/models`);
      if (response.ok) {
        const data = await response.json();
        setAvailableModels(data.model_names || ['llama3:8b']);
        if (data.model_names && data.model_names.length > 0) {
          setSelectedModel(data.model_names[0]);
        }
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const fetchAgentChats = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/agent/instances`);
      if (response.ok) {
        const instances = await response.json();
        setAgentChats(instances.map((inst: any) => ({
          instanceId: inst.instance_id,
          agentId: inst.agent_id,
          name: inst.name || `Agent ${inst.agent_id.slice(0, 8)}`,
          lastMessage: 'Click to start chatting',
        })));
      }
    } catch (error) {
      console.error('Failed to fetch agent chats:', error);
    }
  };

  const fetchGroupChats = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/agent/group-chat/list`);
      if (response.ok) {
        const chats = await response.json();
        setGroupChats(chats);
      }
    } catch (error) {
      console.error('Failed to fetch group chats:', error);
    }
  };

  const openChat = async (chat: AgentChat) => {
    setSelectedChat(chat);
    setSelectedGroupChat(null);
    setActiveView('chat');
    // Don't clear messages immediately - load history first to avoid flicker
    
    // Load conversation history
    try {
      const response = await fetch(`${API_BASE}/api/agent/${chat.instanceId}/conversation`);
      if (response.ok) {
        const data = await response.json();
        const history = data.messages || [];
        // Map history messages properly, ensuring content is a string
        const mappedMessages = history.map((msg: any, index: number) => ({
          id: msg.message_id || `msg-hist-${index}-${Date.now()}`,
          role: msg.role || 'user',
          content: String(msg.content || ''),
          timestamp: msg.timestamp || msg.created_at || new Date().toISOString(),
          streaming: false,
        }));
        setMessages(mappedMessages);
      } else {
        // If no history, start with empty array
        setMessages([]);
      }
    } catch (error) {
      console.error('Failed to load conversation history:', error);
      // On error, start with empty array
      setMessages([]);
    }
  };

  const openGroupChat = (groupChat: GroupChat) => {
    setSelectedGroupChat(groupChat);
    setSelectedChat(null);
    setActiveView('group-chat');
    setMessages([]);
  };

  const createAgent = async () => {
    if (!newAgentName.trim()) {
      alert('Please enter an agent name');
      return;
    }

    setIsLoading(true);
    try {
      const agentType = agentTypes.find(t => t.id === selectedAgentType);
      const response = await fetch(`${API_BASE}/api/agent/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newAgentName,
          agent_type: selectedAgentType,
          model: selectedModel,
          temperature: agentType?.temperature || 0.7,
          max_tokens: agentType?.max_tokens || 2000,
          tools: agentType?.tools || [],
        }),
      });

      if (response.ok) {
        const data = await response.json();
        // Refresh agent chats
        await fetchAgentChats();
        // Close dialog and open the new chat
        setShowCreateDialog(false);
        setNewAgentName('');
        const newChat: AgentChat = {
          instanceId: data.instance_id,
          agentId: data.agent_id,
          name: data.name || newAgentName,
        };
        await openChat(newChat);
      } else {
        const errorText = await response.text();
        alert(`Failed to create agent: ${errorText}`);
      }
    } catch (error) {
      console.error('Failed to create agent:', error);
      alert('Failed to create agent. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    setIsLoading(true);
    const userMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');

    try {
      if (selectedChat) {
        // Send to single agent
        const response = await fetch(`${API_BASE}/api/agent/invoke`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            instance_id: selectedChat.instanceId,
            input: inputMessage,
            stream: true,
          }),
        });

        if (response.ok) {
          const reader = response.body?.getReader();
          const decoder = new TextDecoder();
          const messageId = `msg-${Date.now()}-response`;
          const contentRef = { current: '' }; // Use ref to track content across closures

          const assistantMessage = {
            id: messageId,
            role: 'assistant',
            content: '',
            timestamp: new Date().toISOString(),
            streaming: true,
          };
          setMessages(prev => [...prev, assistantMessage]);

          if (reader) {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              // Decode chunk - handle incomplete UTF-8 sequences
              let chunk = '';
              try {
                chunk = decoder.decode(value, { stream: true });
              } catch (e) {
                console.warn('Decode error, trying with flush:', e);
                chunk = decoder.decode(value, { stream: false });
              }
              
              const lines = chunk.split('\n').filter(line => line.trim());

              for (const line of lines) {
                try {
                  const data = JSON.parse(line);
                  if (data.type === 'token' && data.content) {
                    // Accumulate content in ref
                    contentRef.current += data.content;
                    // Use functional update with message ID to ensure we update the correct message
                    setMessages(prev => {
                      const updated = prev.map(msg => {
                        if (msg.id === messageId) {
                          return { 
                            ...msg, 
                            content: contentRef.current,
                            streaming: true // Keep streaming true until done
                          };
                        }
                        return msg;
                      });
                      return updated;
                    });
                  } else if (data.type === 'done') {
                    // Mark streaming as complete
                    setMessages(prev => {
                      const updated = prev.map(msg => {
                        if (msg.id === messageId) {
                          return { ...msg, streaming: false };
                        }
                        return msg;
                      });
                      return updated;
                    });
                  } else if (data.type === 'error') {
                    setMessages(prev => {
                      const updated = prev.map(msg => {
                        if (msg.id === messageId) {
                          return { ...msg, content: data.content || 'Error occurred', streaming: false, isError: true };
                        }
                        return msg;
                      });
                      return updated;
                    });
                  }
                } catch (e) {
                  // Non-JSON line, ignore
                  console.debug('Failed to parse line:', line.substring(0, 100), e);
                }
              }
            }
            
            // Flush decoder and ensure streaming is marked as complete
            try {
              decoder.decode(new Uint8Array(), { stream: false });
            } catch (e) {
              // Ignore flush errors
            }
            
            // Final update to ensure streaming is marked as complete
            setMessages(prev => {
              const updated = prev.map(msg => {
                if (msg.id === messageId) {
                  return { ...msg, content: contentRef.current, streaming: false };
                }
                return msg;
              });
              return updated;
            });
          }
        }
      } else if (selectedGroupChat) {
        // Send to group chat
        const response = await fetch(`${API_BASE}/api/agent/group-chat/${selectedGroupChat.groupChatId}/message`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            group_chat_id: selectedGroupChat.groupChatId,
            message: inputMessage,
            stream: true,
          }),
        });

        if (response.ok) {
          const reader = response.body?.getReader();
          const decoder = new TextDecoder();

          if (reader) {
            let currentAgentId: string | null = null;
            let agentResponses: Record<string, string> = {};

            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              const chunk = decoder.decode(value);
              const lines = chunk.split('\n').filter(line => line.trim());

              for (const line of lines) {
                try {
                  const data = JSON.parse(line);
                  if (data.type === 'agent_start') {
                    currentAgentId = data.agent_id;
                    agentResponses[data.agent_id] = '';
                  } else if (data.type === 'token' && currentAgentId) {
                    agentResponses[currentAgentId] += data.content;
                    // Update messages with agent responses
                    setMessages(prev => {
                      const updated = [...prev];
                      // Find or create agent message
                      let agentMsgIdx = updated.findIndex(
                        m => m.role === 'assistant' && m.agentId === currentAgentId
                      );
                      if (agentMsgIdx === -1) {
                        updated.push({
                          id: `msg-${Date.now()}-${currentAgentId}`,
                          role: 'assistant',
                          agentId: currentAgentId,
                          agentName: data.agent_name,
                          content: agentResponses[currentAgentId],
                          timestamp: new Date().toISOString(),
                        });
                      } else {
                        updated[agentMsgIdx] = {
                          ...updated[agentMsgIdx],
                          content: agentResponses[currentAgentId],
                        };
                      }
                      return updated;
                    });
                  }
                } catch {
                  // Non-JSON line, ignore
                }
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages(prev => [...prev, {
        id: `msg-${Date.now()}-error`,
        role: 'system',
        content: 'Failed to send message. Please try again.',
        timestamp: new Date().toISOString(),
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (!isOpen) return null;

  return (
    <div className="messages-widget-overlay" onClick={onClose}>
      <div className="messages-widget" onClick={(e) => e.stopPropagation()}>
        <div className="messages-widget-header">
          <h3><FaComments /> Messages</h3>
          <button className="btn-icon" onClick={onClose}>
            <FaTimes />
          </button>
        </div>

        {activeView === 'list' && (
          <div className="messages-widget-content">
            <div className="messages-widget-actions">
              <button className="btn-primary btn-small" onClick={() => setShowCreateDialog(true)}>
                <FaPlus /> New Agent
              </button>
              <button className="btn-secondary btn-small" onClick={() => {
                // Create group chat
                const name = prompt('Enter group chat name:');
                if (name) {
                  // This would open a dialog to select agents
                  alert('Group chat creation - select agents from playground');
                }
              }}>
                <FaUsers /> New Group
              </button>
            </div>

            {showCreateDialog && (
              <div className="modal-overlay" onClick={() => setShowCreateDialog(false)}>
                <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                  <div className="modal-header">
                    <h3>Create New Agent</h3>
                    <button className="btn-icon" onClick={() => setShowCreateDialog(false)}>
                      <FaTimes />
                    </button>
                  </div>
                  <div className="modal-body">
                    <div className="form-group">
                      <label>Agent Name</label>
                      <input
                        type="text"
                        value={newAgentName}
                        onChange={(e) => setNewAgentName(e.target.value)}
                        placeholder="Enter agent name..."
                        className="form-input"
                      />
                    </div>
                    <div className="form-group">
                      <label>Agent Type</label>
                      <select
                        value={selectedAgentType}
                        onChange={(e) => setSelectedAgentType(e.target.value)}
                        className="form-select"
                      >
                        {agentTypes.map(type => (
                          <option key={type.id} value={type.id}>
                            {type.name} - {type.description}
                          </option>
                        ))}
                      </select>
                      {agentTypes.find(t => t.id === selectedAgentType) && (
                        <p className="form-help">
                          {agentTypes.find(t => t.id === selectedAgentType)?.description}
                        </p>
                      )}
                    </div>
                    <div className="form-group">
                      <label>Model</label>
                      <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="form-select"
                      >
                        {availableModels.map(model => (
                          <option key={model} value={model}>{model}</option>
                        ))}
                      </select>
                    </div>
                    {agentTypes.find(t => t.id === selectedAgentType)?.tools && 
                     agentTypes.find(t => t.id === selectedAgentType)!.tools!.length > 0 && (
                      <div className="form-group">
                        <label>Available Tools</label>
                        <div className="tools-list">
                          {agentTypes.find(t => t.id === selectedAgentType)!.tools!.map(tool => (
                            <span key={tool} className="tool-badge">{tool}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="modal-footer">
                    <button className="btn-secondary" onClick={() => setShowCreateDialog(false)}>
                      Cancel
                    </button>
                    <button 
                      className="btn-primary" 
                      onClick={createAgent}
                      disabled={isLoading || !newAgentName.trim()}
                    >
                      {isLoading ? 'Creating...' : 'Create Agent'}
                    </button>
                  </div>
                </div>
              </div>
            )}

            <div className="messages-list">
              <h4><FaRobot /> Agent Chats</h4>
              {agentChats.length === 0 ? (
                <div className="empty-state">
                  <p>No agent chats yet. Start a chat from the Agent Playground.</p>
                </div>
              ) : (
                agentChats.map(chat => (
                  <div
                    key={chat.instanceId}
                    className="message-item"
                    onClick={() => openChat(chat)}
                  >
                    <div className="message-item-avatar">
                      <FaRobot />
                    </div>
                    <div className="message-item-content">
                      <div className="message-item-name">{chat.name}</div>
                      <div className="message-item-preview">{chat.lastMessage}</div>
                    </div>
                    {chat.unreadCount && chat.unreadCount > 0 && (
                      <div className="message-item-badge">{chat.unreadCount}</div>
                    )}
                  </div>
                ))
              )}

              <h4 style={{ marginTop: '20px' }}><FaUsers /> Group Chats</h4>
              {groupChats.length === 0 ? (
                <div className="empty-state">
                  <p>No group chats yet. Create one from the Agent Playground.</p>
                </div>
              ) : (
                groupChats.map(chat => (
                  <div
                    key={chat.groupChatId}
                    className="message-item"
                    onClick={() => openGroupChat(chat)}
                  >
                    <div className="message-item-avatar">
                      <FaUsers />
                    </div>
                    <div className="message-item-content">
                      <div className="message-item-name">{chat.name}</div>
                      <div className="message-item-preview">{chat.agentCount} agents</div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {(activeView === 'chat' || activeView === 'group-chat') && (
          <div className="messages-widget-content chat-view">
            <div className="chat-header">
              <button className="btn-icon" onClick={() => setActiveView('list')}>
                <FaChevronUp />
              </button>
              <div className="chat-title">
                {selectedChat?.name || selectedGroupChat?.name}
                {selectedGroupChat && (
                  <span className="chat-subtitle">{selectedGroupChat.agentCount} agents</span>
                )}
              </div>
            </div>

            <div className="chat-messages">
              {messages.length === 0 ? (
                <div className="empty-state">
                  <p>Start a conversation...</p>
                </div>
              ) : (
                messages.map(msg => (
                  <div key={msg.id} className={`chat-message ${msg.role}`}>
                    {msg.role === 'assistant' && msg.agentName && (
                      <div className="chat-message-agent">{msg.agentName}</div>
                    )}
                    <div className="chat-message-content">{msg.content}</div>
                    {msg.streaming && <span className="cursor">▊</span>}
                  </div>
                ))
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-container">
              <input
                type="text"
                value={inputMessage}
                onChange={e => setInputMessage(e.target.value)}
                onKeyPress={e => e.key === 'Enter' && !isLoading && sendMessage()}
                placeholder="Type a message..."
                disabled={isLoading}
              />
              <button
                className="btn-primary"
                onClick={sendMessage}
                disabled={isLoading || !inputMessage.trim()}
              >
                Send
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


