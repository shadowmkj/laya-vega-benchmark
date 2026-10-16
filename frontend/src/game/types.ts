export type DriverMode = 'human' | 'laya' | 'vega';
export type ExperimentRound = 'raw_async' | 'fixed_45ms_hold';

export interface Obstacle {
  id: number;
  x: number;
  width: number;
  height: number;
  type: 'small' | 'large' | 'cluster';
}

export interface DinoObservation {
  obstacle_id: number;
  distance_to_cactus: number;
  cactus_width: number;
  cactus_height: number;
  dino_is_grounded: boolean;
  speed: number;
  latency_budget_ms: number;
}

export interface GameState {
  dinoY: number;
  dinoVy: number;
  isGrounded: boolean;
  score: number;
  highScore: number;
  speed: number;
  gameOver: boolean;
  obstacles: Obstacle[];
  ticks: number;
}
