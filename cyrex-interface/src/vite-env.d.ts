/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_CYREX_BASE_URL?: string;
  readonly VITE_MESSAGING_SERVICE_URL?: string;
  readonly VITE_REALTIME_GATEWAY_URL?: string;
  readonly VITE_SPEECH_URL?: string;
  readonly VITE_LIVEKIT_URL?: string;
  readonly VITE_LOCAL_USER_ID?: string;
  readonly VITE_SYNAPSE_URL?: string;
  readonly VITE_PORT?: string;
  readonly VITE_HMR_HOST?: string;
  readonly VITE_HMR_PORT?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
