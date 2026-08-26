/**
 * HTTP client for deepiri-messaging-service.
 * Used when messaging is up; cyrex-interface falls back to Cyrex direct otherwise.
 */

import {
  LOCAL_USER_ID,
  MESSAGING_SERVICE_URL,
  saveRoomMapping,
} from './platformConfig';

export type MessagingDeliveryMode = 'platform' | 'direct' | 'unknown';

export interface MessagingChatRoom {
  id: string;
  type: string;
  name?: string;
  agentInstanceId?: string | null;
  createdAt?: string;
}

export interface MessagingMessage {
  id: string;
  chatRoomId: string;
  senderId?: string;
  senderType?: 'USER' | 'AGENT' | 'SYSTEM' | string;
  agentInstanceId?: string | null;
  content: string;
  messageType?: string;
  createdAt?: string;
  metadata?: Record<string, unknown>;
}

let cachedAvailable: boolean | null = null;
let lastProbeAt = 0;
const PROBE_TTL_MS = 15_000;

function headers(): HeadersInit {
  return {
    'Content-Type': 'application/json',
    'x-user-id': LOCAL_USER_ID,
  };
}

export async function probeMessagingAvailable(force = false): Promise<boolean> {
  const now = Date.now();
  if (!force && cachedAvailable !== null && now - lastProbeAt < PROBE_TTL_MS) {
    return cachedAvailable;
  }
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 1500);
    const res = await fetch(`${MESSAGING_SERVICE_URL}/health`, { signal: ctrl.signal });
    clearTimeout(t);
    cachedAvailable = res.ok;
  } catch {
    cachedAvailable = false;
  }
  lastProbeAt = now;
  return cachedAvailable;
}

export function getCachedMessagingAvailable(): boolean | null {
  return cachedAvailable;
}

export async function listAgentChats(): Promise<MessagingChatRoom[]> {
  const res = await fetch(`${MESSAGING_SERVICE_URL}/api/v1/chats?type=AGENT&take=100`, {
    headers: headers(),
  });
  if (!res.ok) throw new Error(`list chats failed: ${res.status}`);
  const body = await res.json();
  return (body.data || []) as MessagingChatRoom[];
}

export async function createAgentChatRoom(
  agentInstanceId: string,
  name: string,
): Promise<MessagingChatRoom> {
  const res = await fetch(`${MESSAGING_SERVICE_URL}/api/v1/chats`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({
      type: 'AGENT',
      name,
      agentInstanceId,
      userIds: [LOCAL_USER_ID],
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`create chat failed: ${res.status} ${text}`);
  }
  const body = await res.json();
  const room = body.data as MessagingChatRoom;
  saveRoomMapping(agentInstanceId, room.id);
  return room;
}

/** Find existing room for instance, or create one. */
export async function ensureAgentChatRoom(
  agentInstanceId: string,
  name: string,
  knownRoomId?: string,
): Promise<string> {
  if (knownRoomId) {
    saveRoomMapping(agentInstanceId, knownRoomId);
    return knownRoomId;
  }
  const rooms = await listAgentChats();
  const existing = rooms.find((r) => r.agentInstanceId === agentInstanceId);
  if (existing?.id) {
    saveRoomMapping(agentInstanceId, existing.id);
    return existing.id;
  }
  const created = await createAgentChatRoom(agentInstanceId, name);
  return created.id;
}

export async function listMessages(chatRoomId: string, limit = 100): Promise<MessagingMessage[]> {
  // Mounted at /api/v1/:chatRoomId/messages (messageRoutes under /)
  const res = await fetch(
    `${MESSAGING_SERVICE_URL}/api/v1/${chatRoomId}/messages?limit=${limit}&offset=0`,
    { headers: headers() },
  );
  if (!res.ok) throw new Error(`list messages failed: ${res.status}`);
  const body = await res.json();
  return (body.data || []) as MessagingMessage[];
}

export async function sendUserMessage(
  chatRoomId: string,
  content: string,
): Promise<MessagingMessage> {
  const res = await fetch(`${MESSAGING_SERVICE_URL}/api/v1/${chatRoomId}/messages`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ content, messageType: 'TEXT' }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`send message failed: ${res.status} ${text}`);
  }
  const body = await res.json();
  return body.data as MessagingMessage;
}

export function messagingToUiMessage(msg: MessagingMessage, index = 0) {
  const role =
    msg.senderType === 'AGENT' || msg.senderType === 'SYSTEM' ? 'assistant' : 'user';
  return {
    id: msg.id || `msg-plat-${index}`,
    role,
    content: String(msg.content || ''),
    timestamp: msg.createdAt || new Date().toISOString(),
    streaming: false,
    source: 'messaging' as const,
  };
}
