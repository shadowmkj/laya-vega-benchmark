#[derive(Debug, Clone, PartialEq)]
pub struct Dino {
    pub y: f32,
    pub vy: f32,
    pub is_grounded: bool,
    pub is_ducking: bool,
    pub duck_ticks_remaining: u16,
}

impl Dino {
    // Graceful, floaty Chrome Dino jump with ~19 frames of airtime
    pub const JUMP_VELOCITY: f32 = 2.1;
    pub const GRAVITY: f32 = -0.22;
    pub const DEFAULT_DUCK_TICKS: u16 = 24; // ~0.8s duck duration for smooth terminal repeat

    pub fn new() -> Self {
        Self {
            y: 0.0,
            vy: 0.0,
            is_grounded: true,
            is_ducking: false,
            duck_ticks_remaining: 0,
        }
    }

    pub fn jump(&mut self) -> bool {
        if self.is_grounded {
            self.is_ducking = false;
            self.duck_ticks_remaining = 0;
            self.vy = Self::JUMP_VELOCITY;
            self.is_grounded = false;
            true
        } else {
            false
        }
    }

    pub fn duck(&mut self) -> bool {
        if self.is_grounded {
            self.is_ducking = true;
            self.duck_ticks_remaining = Self::DEFAULT_DUCK_TICKS;
            true
        } else {
            // Fast fall when ducking in mid-air
            self.vy += Self::GRAVITY * 1.5;
            true
        }
    }

    pub fn stand(&mut self) {
        self.is_ducking = false;
        self.duck_ticks_remaining = 0;
    }

    pub fn tick(&mut self, gravity: f32) {
        if self.duck_ticks_remaining > 0 {
            self.duck_ticks_remaining -= 1;
            if self.duck_ticks_remaining == 0 {
                self.is_ducking = false;
            }
        }

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
