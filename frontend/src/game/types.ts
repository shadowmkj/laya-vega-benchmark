export type DriverMode = 'human' | 'laya';
export type ExperimentRound = 'raw_async' | 'fixed_45ms_hold';

export interface Obstacle {
  id: number;
  x: number;
  yOffset: number;
  width: number;
  height: number;
  type: 'small' | 'large' | 'cluster' | 'pterodactyl';
  altitude?: 'ground' | 'mid' | 'high';
}

export interface DinoObservation {
  obstacle_id: number;
  distance_to_cactus: number;
  cactus_width: number;
  cactus_height: number;
  obstacle_type: 'cactus' | 'pterodactyl';
  obstacle_altitude: 'ground' | 'mid' | 'high';
  dino_is_grounded: boolean;
  dino_is_ducking: boolean;
  speed: number;
  latency_budget_ms: number;
}

export interface GameState {
  dinoY: number;
  dinoVy: number;
  isGrounded: boolean;
  isDucking: boolean;
  score: number;
  highScore: number;
  speed: number;
  gameOver: boolean;
  obstacles: Obstacle[];
  ticks: number;
}
