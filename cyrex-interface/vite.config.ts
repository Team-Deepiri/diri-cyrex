import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  const port = Number(env.VITE_PORT ?? 5175);
  const hmrHost = env.VITE_HMR_HOST || 'localhost';
  const hmrPort = Number(env.VITE_HMR_PORT || port);

  return {
    plugins: [
      react({
        // Enable Fast Refresh (HMR for React)
        fastRefresh: true
      })
    ],
    server: {
      port,
      host: '0.0.0.0',
      strictPort: false,
      hmr: {
        host: hmrHost,
        port: hmrPort,
        protocol: 'ws',
        clientPort: hmrPort
      },
      watch: {
        usePolling: true,
        interval: 1000,
        ignored: ['**/node_modules/**', '**/.git/**']
      },
      // Enable CORS for HMR
      cors: true
    },
    define: {
      __BUILD_TIME__: JSON.stringify(new Date().toISOString())
    }
  };
});

