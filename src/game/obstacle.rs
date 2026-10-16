use super::physics::Dino;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObstacleType {
    SmallCactus,
    LargeCactus,
    Cluster,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Obstacle {
    pub x: f32,
    pub width: u16,
    pub height: u16,
    pub obstacle_type: ObstacleType,
}

impl Obstacle {
    pub fn new(x: f32, obstacle_type: ObstacleType) -> Self {
        let (width, height) = match obstacle_type {
            ObstacleType::SmallCactus => (3, 2),
            ObstacleType::LargeCactus => (3, 3),
            ObstacleType::Cluster => (7, 2),
        };
        Self {
            x,
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

    pub fn tick(&mut self, speed: f32, obstacles: &mut Vec<Obstacle>, spawn_x: f32) {
        for obs in obstacles.iter_mut() {
            obs.step(speed);
        }

        // Remove obstacles that have rolled off-screen
        obstacles.retain(|obs| obs.x + (obs.width as f32) > 0.0);

        self.next_spawn_dist -= speed;
        if self.next_spawn_dist <= 0.0 {
            let roll = self.rng.u8(0..3);
            let kind = match roll {
                0 => ObstacleType::SmallCactus,
                1 => ObstacleType::LargeCactus,
                _ => ObstacleType::Cluster,
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

/// Forgiving collision detection with generous foot and snout leeway.
pub fn check_collision(
    dino: &Dino,
    dino_x: f32,
    _dino_w: f32,
    _dino_h: f32,
    obstacles: &[Obstacle],
) -> bool {
    // Dino main torso & legs: columns 2.0 to 6.5 relative to DINO_X (ignore tail 0..2 and snout tip 6.5..9)
    let dino_left = dino_x + 2.0;
    let dino_right = dino_x + 6.5;
    let dino_bottom = dino.y;

    for obs in obstacles {
        let obs_left = obs.x + 0.4;
        let obs_right = obs.x + (obs.width as f32) - 0.4;
        // Generous vertical clearance: clear if foot is within 0.3 units of top
        let obs_top = (obs.height as f32) - 0.3;

        let horizontal_overlap = dino_right > obs_left && dino_left < obs_right;
        let vertical_collision = dino_bottom < obs_top;

        if horizontal_overlap && vertical_collision {
            return true;
        }
    }

    false
}
