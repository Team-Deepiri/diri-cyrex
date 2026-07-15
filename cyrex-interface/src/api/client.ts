// Shared API client wrapper.

const BASE_API_URL = '/api/v1';

export async function apiRequest<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(`${BASE_API_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${res.statusText}`);
  return res.json() as Promise<T>;
}

export async function apiUpload<T>(
  path: string,
  formData: FormData,
): Promise<T> {
  const res = await fetch(`${BASE_API_URL}${path}`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload error ${res.status}: ${res.statusText}`);
  return res.json() as Promise<T>;
}