import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [react()],
    server: {
      port: Number(env.VITE_PORT ?? 5175),
      host: '0.0.0.0'
    },
    define: {
      __BUILD_TIME__: JSON.stringify(new Date().toISOString())
    }
  };
});

