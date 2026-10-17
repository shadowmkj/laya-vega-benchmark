export interface MoveOptionProbabilities {
  jump: number;
  duck: number;
  run: number;
}

export interface MoveHistoryItem {
  id: string;
  timestamp: number;
  timeString: string;
  driver: 'laya' | 'human';
  fromAction: string;
  action: 'JUMP' | 'DUCK' | 'RUN';
  confidence: number;
  probabilities: MoveOptionProbabilities;
  obstacleType: string;
  obstacleAltitude: string;
  distance: number;
  serverLatencyMs: number;
  roundTripMs: number;
}

export interface TelemetryMetrics {
  lastLatencyMs: number;
  lastRttMs?: number;
  lastConfidence: number;
  lastAction: string;
  lastProbabilities?: MoveOptionProbabilities;
  p50: number;
  p90: number;
  p99: number;
  jitter: number;
  deadlineMisses: number;
  history: number[];
}

export class TelemetryTracker {
  private samples: number[] = [];
  public deadlineMisses = 0;
  public lastLatencyMs = 0;
  public lastRttMs = 0;
  public lastConfidence = 0;
  public lastAction = 'RUN';
  public lastProbabilities?: MoveOptionProbabilities;

  public record(
    latencyMs: number,
    confidence: number,
    action: string,
    probabilities?: MoveOptionProbabilities,
    rttMs?: number
  ) {
    this.lastLatencyMs = latencyMs;
    this.lastRttMs = rttMs ?? latencyMs;
    this.lastConfidence = confidence;
    this.lastAction = action.toUpperCase();
    if (probabilities) {
      this.lastProbabilities = probabilities;
    }

    this.samples.push(latencyMs);
    if (this.samples.length > 500) {
      this.samples.shift();
    }

    if (latencyMs > 45.0) {
      this.deadlineMisses += 1;
    }
  }

  public getMetrics(): TelemetryMetrics {
    return {
      lastLatencyMs: this.lastLatencyMs,
      lastRttMs: this.lastRttMs,
      lastConfidence: this.lastConfidence,
      lastAction: this.lastAction,
      lastProbabilities: this.lastProbabilities,
      p50: this.percentile(0.50),
      p90: this.percentile(0.90),
      p99: this.percentile(0.99),
      jitter: this.jitter(),
      deadlineMisses: this.deadlineMisses,
      history: this.samples.slice(-35),
    };
  }

  private percentile(p: number): number {
    if (this.samples.length === 0) return 0;
    const sorted = [...this.samples].sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * p) - 1));
    return Math.round(sorted[idx] * 10) / 10;
  }

  private jitter(): number {
    if (this.samples.length < 2) return 0;
    const mean = this.samples.reduce((a, b) => a + b, 0) / this.samples.length;
    const variance =
      this.samples.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / this.samples.length;
    return Math.round(Math.sqrt(variance) * 10) / 10;
  }
}
