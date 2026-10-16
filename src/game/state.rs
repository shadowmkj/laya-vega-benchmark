use serde::{Deserialize, Serialize};
use super::physics::Dino;
use super::obstacle::{check_collision, FlightAltitude, Obstacle, ObstacleSpawner, ObstacleType};

pub const DINO_X: f32 = 6.0;
pub const DINO_WIDTH: f32 = 4.0;
pub const DINO_HEIGHT: f32 = 3.0;
pub const TRACK_SPAWN_X: f32 = 90.0;
pub const BASE_SPEED: f32 = 1.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DinoObservation {
    pub distance_to_cactus: f32,
    pub cactus_width: u16,
    pub cactus_height: u16,
    pub obstacle_type: String,     // "cactus" or "pterodactyl"
    pub obstacle_altitude: String, // "ground", "mid", "high"
    pub dino_is_grounded: bool,
    pub dino_is_ducking: bool,
    pub speed: f32,
    pub latency_budget_ms: f32,
}

#[derive(Debug, Clone)]
pub struct GameState {
    pub dino: Dino,
    pub obstacles: Vec<Obstacle>,
    pub spawner: ObstacleSpawner,
    pub speed: f32,
    pub score: u32,
    pub high_score: u32,
    pub ticks: u64,
    pub game_over: bool,
    pub seed: u64,
}

impl GameState {
    pub fn new(seed: u64) -> Self {
        Self {
            dino: Dino::new(),
            obstacles: Vec::new(),
            spawner: ObstacleSpawner::new(seed),
            speed: BASE_SPEED,
            score: 0,
            high_score: 0,
            ticks: 0,
            game_over: false,
            seed,
        }
    }

    pub fn jump(&mut self) -> bool {
        if !self.game_over {
            self.dino.jump()
        } else {
            false
        }
    }

    pub fn duck(&mut self) -> bool {
        if !self.game_over {
            self.dino.duck()
        } else {
            false
        }
    }

    pub fn tick(&mut self) {
        if self.game_over {
            return;
        }

        self.ticks += 1;
        self.score += 1;
        if self.score > self.high_score {
            self.high_score = self.score;
        }

        // Gentle speed progression
        self.speed = BASE_SPEED + (self.score as f32 / 2000.0).min(0.8);

        self.dino.tick(Dino::GRAVITY);
        self.spawner.tick(self.speed, &mut self.obstacles, TRACK_SPAWN_X, self.score);

        if check_collision(&self.dino, DINO_X, DINO_WIDTH, DINO_HEIGHT, &self.obstacles) {
            self.game_over = true;
        }
    }

    pub fn reset(&mut self) {
        self.dino = Dino::new();
        self.obstacles.clear();
        self.spawner.reset(self.seed + self.ticks);
        self.speed = BASE_SPEED;
        self.score = 0;
        self.game_over = false;
    }

    pub fn get_observation(&self, latency_budget_ms: f32) -> DinoObservation {
        // Find closest obstacle in front of Dino's body
        let closest = self
            .obstacles
            .iter()
            .filter(|obs| obs.x + (obs.width as f32) > DINO_X + 2.0)
            .min_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

        let (mut dist, w, h, obs_type, obs_alt) = if let Some(obs) = closest {
            // Distance from Dino's front body (DINO_X + 6.0) to obstacle left edge
            let d = (obs.x - (DINO_X + 6.0)).max(0.0);
            let (t_str, alt_str) = match obs.obstacle_type {
                ObstacleType::SmallCactus | ObstacleType::LargeCactus | ObstacleType::Cluster => {
                    ("cactus", "ground")
                }
                ObstacleType::Pterodactyl(FlightAltitude::Low) => ("pterodactyl", "ground"),
                ObstacleType::Pterodactyl(FlightAltitude::Mid) => ("pterodactyl", "mid"),
                ObstacleType::Pterodactyl(FlightAltitude::High) => ("pterodactyl", "high"),
            };
            (d, obs.width, obs.height, t_str.to_string(), alt_str.to_string())
        } else {
            (999.0, 0, 0, "none".to_string(), "ground".to_string())
        };

        if latency_budget_ms > 0.0 && dist < 900.0 {
            // Speed is cells/frame at 30 FPS -> cells/second = speed * 30.0 -> cells/ms = speed * 0.03
            let cells_per_ms = self.speed * 0.03;
            let projected_movement = cells_per_ms * latency_budget_ms;
            dist = (dist - projected_movement).max(0.0);
        }

        DinoObservation {
            distance_to_cactus: (dist * 10.0).round() / 10.0,
            cactus_width: w,
            cactus_height: h,
            obstacle_type: obs_type,
            obstacle_altitude: obs_alt,
            dino_is_grounded: self.dino.is_grounded,
            dino_is_ducking: self.dino.is_ducking,
            speed: (self.speed * 100.0).round() / 100.0,
            latency_budget_ms,
        }
    }
}
