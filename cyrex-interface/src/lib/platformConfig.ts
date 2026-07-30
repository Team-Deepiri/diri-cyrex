/** Platform URLs + local user identity for cyrex-interface. */

export const CYREX_BASE_URL =
  import.meta.env.VITE_CYREX_BASE_URL || 'http://localhost:8000';

/** Messaging HTTP — compose maps host 5010. */
export const MESSAGING_SERVICE_URL =
  import.meta.env.VITE_MESSAGING_SERVICE_URL || 'http://localhost:5010';

/** Realtime Gateway Socket.IO — compose maps host 5008. */
export const REALTIME_GATEWAY_URL =
  import.meta.env.VITE_REALTIME_GATEWAY_URL || 'http://localhost:5008';

/** Synthetic user for local interface (messaging requires x-user-id). */
export const LOCAL_USER_ID =
  import.meta.env.VITE_LOCAL_USER_ID || '00000000-0000-4000-8000-000000000001';

const ROOM_MAP_KEY = 'cyrex-interface:agent-chat-rooms';

export function loadRoomMap(): Record<string, string> {
  try {
    const raw = localStorage.getItem(ROOM_MAP_KEY);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch {
    return {};
  }
}

export function saveRoomMapping(instanceId: string, chatRoomId: string): void {
  const map = loadRoomMap();
  map[instanceId] = chatRoomId;
  localStorage.setItem(ROOM_MAP_KEY, JSON.stringify(map));
}

export function getStoredRoomId(instanceId: string): string | undefined {
  return loadRoomMap()[instanceId];
}
