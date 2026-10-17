import { DinoObservation } from '../game/types';

export interface PredictResponse {
  id: number;
  model: string;
  action: 'jump' | 'duck' | 'run';
  confidence: number;
  probabilities?: Record<'jump' | 'duck' | 'run' | string, number>;
  inference_ms: number;
}

const STORAGE_KEY = 'layadino_backend_url';

export function getBackendUrl(): string {
  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return saved.trim().replace(/\/+$/, '');
  }
  return (import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000').trim().replace(/\/+$/, '');
}

export function setBackendUrl(url: string): string {
  const clean = url.trim().replace(/\/+$/, '');
  if (typeof window !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, clean);
  }
  return clean;
}

export async function checkBackendHealth(targetUrl?: string): Promise<boolean> {
  const base = (targetUrl || getBackendUrl()).trim().replace(/\/+$/, '');
  try {
    const res = await fetch(`${base}/health`, { method: 'GET', signal: AbortSignal.timeout(2000) });
    const data = await res.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
}

export async function requestPrediction(
  model: 'laya',
  state: DinoObservation
): Promise<PredictResponse> {
  const base = getBackendUrl();
  const res = await fetch(`${base}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: Date.now(),
      model,
      state,
    }),
  });

  if (!res.ok) {
    throw new Error(`Inference request failed: ${res.statusText}`);
  }

  return res.json();
}
