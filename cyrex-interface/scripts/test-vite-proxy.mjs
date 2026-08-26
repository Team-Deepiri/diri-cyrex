import http from 'node:http';
import { spawn } from 'node:child_process';

const upstreamPort = 8123;
const vitePort = 5199;

const upstream = http.createServer((request, response) => {
  response.setHeader('Content-Type', 'application/json');
  response.end(JSON.stringify({ path: request.url, proxied: true }));
});

const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';

function waitFor(url, timeoutMs = 30000) {
  const started = Date.now();

  return new Promise((resolve, reject) => {
    const attempt = async () => {
      try {
        const response = await fetch(url);
        resolve(response);
      } catch (error) {
        if (Date.now() - started >= timeoutMs) {
          reject(error);
          return;
        }
        setTimeout(attempt, 250);
      }
    };
    attempt();
  });
}

await new Promise((resolve) => upstream.listen(upstreamPort, '127.0.0.1', resolve));

const vite = spawn(
  npmCommand,
  ['run', 'dev', '--', '--host', '127.0.0.1', '--port', String(vitePort)],
  {
    cwd: process.cwd(),
    env: {
      ...process.env,
      VITE_CYREX_API_URL: `http://127.0.0.1:${upstreamPort}`,
    },
    shell: process.platform === 'win32',
    stdio: 'ignore',
  },
);

try {
  const response = await waitFor(
    `http://127.0.0.1:${vitePort}/api/v1/proxy-test`,
  );
  const body = await response.json();

  if (!response.ok || body.path !== '/api/v1/proxy-test' || body.proxied !== true) {
    throw new Error(`Unexpected proxy response: ${JSON.stringify(body)}`);
  }

  console.log('Vite proxy passed:', JSON.stringify(body));
} finally {
  vite.kill('SIGTERM');
  upstream.close();
}
