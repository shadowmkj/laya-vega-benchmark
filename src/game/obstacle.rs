use serde::{Deserialize, Serialize};
use super::physics::Dino;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlightAltitude {
    Low,   // Near ground (requires jump)
    Mid,   // At head level (requires duck)
    High,  // Overhead (safe to run under)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObstacleType {
    SmallCactus,
    LargeCactus,
    Cluster,
    Pterodactyl(FlightAltitude),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Obstacle {
    pub x: f32,
    pub y_offset: f32,
    pub width: u16,
    pub height: u16,
    pub obstacle_type: ObstacleType,
}

impl Obstacle {
    pub fn new(x: f32, obstacle_type: ObstacleType) -> Self {
        let (width, height, y_offset) = match obstacle_type {
            ObstacleType::SmallCactus => (3, 2, 0.0),
            ObstacleType::LargeCactus => (3, 3, 0.0),
            ObstacleType::Cluster => (7, 2, 0.0),
            ObstacleType::Pterodactyl(FlightAltitude::Low) => (6, 2, 0.0),
            ObstacleType::Pterodactyl(FlightAltitude::Mid) => (6, 2, 1.8),
            ObstacleType::Pterodactyl(FlightAltitude::High) => (6, 2, 3.8),
        };
        Self {
            x,
            y_offset,
            width,
            height,
            obstacle_type,
        }
    }

    pub fn step(&mut self, speed: f32) {
        self.x -= speed;
    }
}

#[derive(Debug, Clone)]
pub struct ObstacleSpawner {
    pub min_gap: f32,
    pub next_spawn_dist: f32,
    rng: fastrand::Rng,
}

impl ObstacleSpawner {
    pub fn new(seed: u64) -> Self {
        Self {
            min_gap: 55.0,
            next_spawn_dist: 60.0,
            rng: fastrand::Rng::with_seed(seed),
        }
    }

    pub fn tick(&mut self, speed: f32, obstacles: &mut Vec<Obstacle>, spawn_x: f32, score: u32) {
        for obs in obstacles.iter_mut() {
            obs.step(speed);
        }

        // Remove obstacles that have rolled off-screen
        obstacles.retain(|obs| obs.x + (obs.width as f32) > 0.0);

        self.next_spawn_dist -= speed;
        if self.next_spawn_dist <= 0.0 {
            // Once score >= 100, introduce flying Pterodactyls into the spawn pool
            let kind = if score >= 100 && self.rng.u8(0..10) < 4 {
                let alt_roll = self.rng.u8(0..3);
                match alt_roll {
                    0 => ObstacleType::Pterodactyl(FlightAltitude::Mid), // Primary duck challenge
                    1 => ObstacleType::Pterodactyl(FlightAltitude::Low), // Jump challenge
                    _ => ObstacleType::Pterodactyl(FlightAltitude::High), // Fly-over
                }
            } else {
                let cactus_roll = self.rng.u8(0..3);
                match cactus_roll {
                    0 => ObstacleType::SmallCactus,
                    1 => ObstacleType::LargeCactus,
                    _ => ObstacleType::Cluster,
                }
            };

            obstacles.push(Obstacle::new(spawn_x, kind));

            // Set next gap with generous spacing so player has room to react and recover
            let gap_variance = self.rng.f32() * 30.0;
            self.next_spawn_dist = self.min_gap + (speed * 12.0) + gap_variance;
        }
    }

    pub fn reset(&mut self, seed: u64) {
        self.rng = fastrand::Rng::with_seed(seed);
        self.next_spawn_dist = 60.0;
    }
}

/// Forgiving collision detection handling jumping, ducking, and airborne obstacles.
pub fn check_collision(
    dino: &Dino,
    dino_x: f32,
    _dino_w: f32,
    _dino_h: f32,
    obstacles: &[Obstacle],
) -> bool {
    let dino_left = dino_x + 2.0;
    let dino_right = if dino.is_ducking { dino_x + 7.5 } else { dino_x + 6.5 };
    let dino_bottom = dino.y;
    // Ducking lowers Dino head clearance significantly from 3.2 down to 1.5
    let dino_top = dino.y + if dino.is_ducking { 1.5 } else { 3.2 };

    for obs in obstacles {
        let obs_left = obs.x + 0.4;
        let obs_right = obs.x + (obs.width as f32) - 0.4;
        let obs_bottom = obs.y_offset;
        let obs_top = obs.y_offset + (obs.height as f32) - 0.3;

        let horizontal_overlap = dino_right > obs_left && dino_left < obs_right;
        let vertical_collision = dino_top > obs_bottom && dino_bottom < obs_top;

        if horizontal_overlap && vertical_collision {
            return true;
        }
    }

    false
}
