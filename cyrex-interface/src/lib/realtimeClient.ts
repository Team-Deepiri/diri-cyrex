/**
 * Socket.IO client for deepiri-realtime-gateway.
 * Listens for messaging `platform-event` / message:new fan-out.
 */

import { io, Socket } from 'socket.io-client';
import { LOCAL_USER_ID, REALTIME_GATEWAY_URL } from './platformConfig';

export type PlatformEventHandler = (event: {
  event?: string;
  source?: string;
  data?: Record<string, unknown>;
  timestamp?: string;
}) => void;

let socket: Socket | null = null;
let connectPromise: Promise<Socket | null> | null = null;

export async function connectRealtime(): Promise<Socket | null> {
  if (socket?.connected) return socket;
  if (connectPromise) return connectPromise;

  connectPromise = new Promise((resolve) => {
    try {
      const s = io(REALTIME_GATEWAY_URL, {
        transports: ['websocket', 'polling'],
        reconnection: true,
        timeout: 4000,
      });

      const failTimer = setTimeout(() => {
        if (!s.connected) {
          s.close();
          socket = null;
          connectPromise = null;
          resolve(null);
        }
      }, 4500);

      s.on('connect', () => {
        clearTimeout(failTimer);
        socket = s;
        s.emit('join_user_room', LOCAL_USER_ID);
        connectPromise = null;
        resolve(s);
      });

      s.on('connect_error', () => {
        // keep trying via reconnection; initial probe can still time out
      });
    } catch {
      connectPromise = null;
      resolve(null);
    }
  });

  return connectPromise;
}

export function disconnectRealtime(): void {
  if (socket) {
    socket.close();
    socket = null;
  }
  connectPromise = null;
}

export async function subscribePlatformEvents(
  handler: PlatformEventHandler,
): Promise<() => void> {
  const s = await connectRealtime();
  if (!s) {
    return () => undefined;
  }

  const listener = (payload: unknown) => {
    if (payload && typeof payload === 'object') {
      handler(payload as Parameters<PlatformEventHandler>[0]);
    }
  };

  s.on('platform-event', listener);
  return () => {
    s.off('platform-event', listener);
  };
}

export function isRealtimeConnected(): boolean {
  return Boolean(socket?.connected);
}
