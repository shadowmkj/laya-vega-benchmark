#[derive(Debug, Clone, PartialEq)]
pub struct Dino {
    pub y: f32,
    pub vy: f32,
    pub is_grounded: bool,
}

impl Dino {
    // Graceful, floaty Chrome Dino jump with ~19 frames of airtime
    pub const JUMP_VELOCITY: f32 = 2.1;
    pub const GRAVITY: f32 = -0.22;

    pub fn new() -> Self {
        Self {
            y: 0.0,
            vy: 0.0,
            is_grounded: true,
        }
    }

    pub fn jump(&mut self) -> bool {
        if self.is_grounded {
            self.vy = Self::JUMP_VELOCITY;
            self.is_grounded = false;
            true
        } else {
            false
        }
    }

    pub fn tick(&mut self, gravity: f32) {
        if !self.is_grounded {
            self.y += self.vy;
            self.vy += gravity;

            if self.y <= 0.0 {
                self.y = 0.0;
                self.vy = 0.0;
                self.is_grounded = true;
            }
        }
    }
}

impl Default for Dino {
    fn default() -> Self {
        Self::new()
    }
}
