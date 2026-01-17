/**
 * LinkedIn-style Messages Widget
 * Can be opened from anywhere in the interface
 */

import React, { useState, useEffect, useRef } from 'react';
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

export function MessagesWidget({ isOpen, onClose }: MessagesWidgetProps) {
  const [activeView, setActiveView] = useState<'list' | 'chat' | 'group-chat'>('list');
  const [agentChats, setAgentChats] = useState<AgentChat[]>([]);
  const [groupChats, setGroupChats] = useState<GroupChat[]>([]);
  const [selectedChat, setSelectedChat] = useState<AgentChat | null>(null);
  const [selectedGroupChat, setSelectedGroupChat] = useState<GroupChat | null>(null);
  const [messages, setMessages] = useState<any[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch agent chats
  useEffect(() => {
    if (isOpen && activeView === 'list') {
      fetchAgentChats();
      fetchGroupChats();
    }
  }, [isOpen, activeView]);

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

  const openChat = (chat: AgentChat) => {
    setSelectedChat(chat);
    setSelectedGroupChat(null);
    setActiveView('chat');
    setMessages([]);
  };

  const openGroupChat = (groupChat: GroupChat) => {
    setSelectedGroupChat(groupChat);
    setSelectedChat(null);
    setActiveView('group-chat');
    setMessages([]);
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
          let fullContent = '';

          const assistantMessage = {
            id: `msg-${Date.now()}-response`,
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

              const chunk = decoder.decode(value);
              const lines = chunk.split('\n').filter(line => line.trim());

              for (const line of lines) {
                try {
                  const data = JSON.parse(line);
                  if (data.type === 'token') {
                    fullContent += data.content;
                    setMessages(prev => {
                      const updated = [...prev];
                      const lastIdx = updated.length - 1;
                      if (updated[lastIdx]?.streaming) {
                        updated[lastIdx] = { ...updated[lastIdx], content: fullContent, streaming: false };
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
              <button className="btn-primary btn-small" onClick={() => {
                // Open agent playground or create new chat
                window.location.hash = '#agent-playground';
                onClose();
              }}>
                <FaPlus /> New Chat
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

