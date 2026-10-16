# Lag-Compensated Terminal Chrome Dino Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 30 FPS terminal Chrome Dino game in Rust with Ratatui and an asynchronous Python inference worker running Laya and Vega decision models, benchmarking latency jitter and a 45ms fixed horizon lag compensation.

**Architecture:** A decoupled multi-process architecture where the Rust game engine runs a strict 30 FPS physics and Ratatui rendering loop in the main thread, communicating with a persistent Python model worker via line-delimited JSON IPC pipes. The Rust engine supports both Human keyboard control and asynchronous AI agents under two experimental modes: Round 1 (Raw Async execution with jitter) and Round 2 (Fixed 45ms Horizon Hold with lag-projected state).

**Tech Stack:** Rust (edition 2021), `ratatui`, `crossterm`, `serde`, `serde_json`, `fastrand`, Python 3.12, `laya`, `vegaml`.

**Spec:** [`docs/superpowers/specs/2026-10-16-lag-compensated-dino-design.md`](file:///Users/milan/Documents/Desk/others/layadino/docs/superpowers/specs/2026-10-16-lag-compensated-dino-design.md)

## Global Constraints

- Rust toolchain: cargo 1.99.0+ / rustc 1.99.0+, edition 2021.
- Game loop timing: Strict 30 FPS ($33.\overline{3}\text{ ms}$ per tick) without blocking on IPC.
- Terminal UI: Built with `ratatui` (0.28+) and `crossterm` (0.28+).
- Python environment: Python >= 3.12 with `laya>=0.3.6` and `vegaml>=0.9.0`.
- IPC: Standard I/O anonymous pipes with line-delimited JSON strings.
- Target hold deadline: 45.0 ms for Round 2 lag compensation.

---

### Task 1: Rust Project Scaffolding & Core Physics

**Files:**
- Create: `Cargo.toml`
- Create: `src/game/physics.rs`
- Create: `src/game/mod.rs`
- Create: `tests/test_physics.rs`

**Interfaces:**
- Produces:
  - `Dino`: Struct with `y: f32`, `vy: f32`, `is_grounded: bool`.
  - `Dino::new() -> Self`
  - `Dino::jump(&mut self) -> bool` (returns true if jump initiated, false if already airborne)
  - `Dino::tick(&mut self, gravity: f32)`

- [ ] **Step 1: Write Cargo.toml with dependencies**

Create `Cargo.toml`:
```toml
[package]
name = "layadino"
version = "0.1.0"
edition = "2021"

[dependencies]
ratatui = "0.28"
crossterm = "0.28"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
fastrand = "2.0"

[[bin]]
name = "layadino"
path = "src/main.rs"
```

- [ ] **Step 2: Write failing unit test for Dino jump physics**

Create `tests/test_physics.rs`:
```rust
use layadino::game::physics::Dino;

#[test]
fn test_dino_starts_grounded() {
    let dino = Dino::new();
    assert_eq!(dino.y, 0.0);
    assert_eq!(dino.vy, 0.0);
    assert!(dino.is_grounded);
}

#[test]
fn test_dino_jump_and_gravity_arc() {
    let mut dino = Dino::new();
    let jumped = dino.jump();
    assert!(jumped);
    assert!(!dino.is_grounded);
    assert!(dino.vy > 0.0);

    // Second jump while airborne should fail
    assert!(!dino.jump());

    // Advance 10 ticks with gravity -1.2
    let mut reached_apex = false;
    for _ in 0..15 {
        dino.tick(-1.2);
        if dino.vy <= 0.0 {
            reached_apex = true;
        }
        if dino.is_grounded {
            break;
        }
    }
    assert!(reached_apex);
    assert!(dino.is_grounded);
    assert_eq!(dino.y, 0.0);
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --test test_physics`
Expected: FAIL (cannot find module `layadino::game::physics`)

- [ ] **Step 4: Implement Dino physics**

Create `src/lib.rs`:
```rust
pub mod game;
```

Create `src/game/mod.rs`:
```rust
pub mod physics;
pub mod obstacle;
pub mod state;
```

Create `src/game/physics.rs`:
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct Dino {
    pub y: f32,
    pub vy: f32,
    pub is_grounded: bool,
}

impl Dino {
    pub const JUMP_VELOCITY: f32 = 4.2;
    pub const GRAVITY: f32 = -1.2;

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
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `cargo test --test test_physics`
Expected: PASS (2 tests pass)

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml src/lib.rs src/game/mod.rs src/game/physics.rs tests/test_physics.rs
git commit -m "feat(engine): implement dino jump physics and unit tests"
```

---

### Task 2: Obstacles & Collision Detection

**Files:**
- Create: `src/game/obstacle.rs`
- Create: `tests/test_obstacles.rs`

**Interfaces:**
- Consumes: `Dino` from `src/game/physics.rs`
- Produces:
  - `ObstacleType`: `SmallCactus`, `LargeCactus`, `Cluster`
  - `Obstacle`: `x: f32`, `width: u16`, `height: u16`, `obstacle_type: ObstacleType`
  - `ObstacleSpawner`: `fn tick(&mut self, dt_speed: f32, obstacles: &mut Vec<Obstacle>, track_width: f32)`
  - `check_collision(dino: &Dino, dino_x: f32, dino_w: f32, dino_h: f32, obstacles: &[Obstacle]) -> bool`

- [ ] **Step 1: Write failing unit test for obstacles and collision detection**

Create `tests/test_obstacles.rs`:
```rust
use layadino::game::physics::Dino;
use layadino::game::obstacle::{Obstacle, ObstacleType, check_collision};

#[test]
fn test_collision_when_grounded_and_overlapping() {
    let dino = Dino::new(); // y = 0.0
    let obstacles = vec![Obstacle {
        x: 6.0,
        width: 2,
        height: 2,
        obstacle_type: ObstacleType::SmallCactus,
    }];

    // Dino is at x=6, w=4, h=3
    assert!(check_collision(&dino, 6.0, 4.0, 3.0, &obstacles));
}

#[test]
fn test_no_collision_when_dino_jumped_over_obstacle() {
    let mut dino = Dino::new();
    dino.y = 5.0; // High in the air
    dino.is_grounded = false;

    let obstacles = vec![Obstacle {
        x: 6.0,
        width: 2,
        height: 2,
        obstacle_type: ObstacleType::SmallCactus,
    }];

    // Height of cactus is 2, dino foot is at y=5.0 -> safe
    assert!(!check_collision(&dino, 6.0, 4.0, 3.0, &obstacles));
}

#[test]
fn test_obstacle_movement() {
    let mut obs = Obstacle {
        x: 20.0,
        width: 2,
        height: 2,
        obstacle_type: ObstacleType::SmallCactus,
    };
    obs.step(1.5);
    assert_eq!(obs.x, 18.5);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test test_obstacles`
Expected: FAIL (cannot find module `layadino::game::obstacle`)

- [ ] **Step 3: Implement obstacle logic and collision checking**

Create `src/game/obstacle.rs`:
```rust
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
            ObstacleType::SmallCactus => (2, 2),
            ObstacleType::LargeCactus => (3, 3),
            ObstacleType::Cluster => (5, 2),
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

pub struct ObstacleSpawner {
    pub min_gap: f32,
    pub next_spawn_dist: f32,
    rng: fastrand::Rng,
}

impl ObstacleSpawner {
    pub fn new(seed: u64) -> Self {
        Self {
            min_gap: 30.0,
            next_spawn_dist: 40.0,
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

            // Set next gap (scaled dynamically by speed + random jitter)
            let gap_variance = self.rng.f32() * 25.0;
            self.next_spawn_dist = self.min_gap + (speed * 10.0) + gap_variance;
        }
    }

    pub fn reset(&mut self, seed: u64) {
        self.rng = fastrand::Rng::with_seed(seed);
        self.next_spawn_dist = 40.0;
    }
}

pub fn check_collision(
    dino: &Dino,
    dino_x: f32,
    dino_w: f32,
    _dino_h: f32,
    obstacles: &[Obstacle],
) -> bool {
    let dino_left = dino_x + 0.5;
    let dino_right = dino_x + dino_w - 0.5;
    let dino_bottom = dino.y;

    for obs in obstacles {
        let obs_left = obs.x;
        let obs_right = obs.x + (obs.width as f32);
        let obs_top = obs.height as f32;

        let horizontal_overlap = dino_right > obs_left && dino_left < obs_right;
        let vertical_collision = dino_bottom < obs_top;

        if horizontal_overlap && vertical_collision {
            return true;
        }
    }

    false
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test test_obstacles`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/game/obstacle.rs tests/test_obstacles.rs
git commit -m "feat(engine): implement obstacles, spawner, and collision detection"
```

---

### Task 3: Game State, Serialization & Observation Model

**Files:**
- Create: `src/game/state.rs`
- Create: `tests/test_state.rs`

**Interfaces:**
- Consumes: `Dino` from `physics.rs`, `Obstacle` from `obstacle.rs`
- Produces:
  - `DinoObservation`: Serializable struct with `distance_to_cactus`, `cactus_width`, `cactus_height`, `dino_is_grounded`, `speed`, `latency_budget_ms`.
  - `GameState`: Full game manager (`tick`, `jump`, `reset`, `get_observation`).

- [ ] **Step 1: Write failing unit test for GameState and observation projection**

Create `tests/test_state.rs`:
```rust
use layadino::game::state::GameState;

#[test]
fn test_game_state_step_and_score() {
    let mut state = GameState::new(42);
    assert_eq!(state.score, 0);
    assert!(!state.game_over);

    state.tick();
    assert_eq!(state.score, 1);
}

#[test]
fn test_observation_projection_with_latency_budget() {
    let mut state = GameState::new(42);
    state.obstacles.push(layadino::game::obstacle::Obstacle::new(
        25.0,
        layadino::game::obstacle::ObstacleType::SmallCactus,
    ));

    // Current distance = 25.0 - 6.0 = 19.0
    let obs_raw = state.get_observation(0.0);
    assert!((obs_raw.distance_to_cactus - 19.0).abs() < 0.1);

    // Projected with 45ms latency budget at speed 1.0 cell/frame (30 frames/sec = 30 cells/sec = 0.03 cells/ms)
    // Distance in 45ms should be 19.0 - (1.0 * 30.0 * 0.045) = 19.0 - 1.35 = 17.65
    let obs_projected = state.get_observation(45.0);
    assert!(obs_projected.distance_to_cactus < obs_raw.distance_to_cactus);
    assert_eq!(obs_projected.latency_budget_ms, 45.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test test_state`
Expected: FAIL

- [ ] **Step 3: Implement GameState and DinoObservation**

Create `src/game/state.rs`:
```rust
use serde::{Deserialize, Serialize};
use super::physics::Dino;
use super::obstacle::{check_collision, Obstacle, ObstacleSpawner};

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
    pub dino_is_grounded: bool,
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

    pub fn tick(&mut self) {
        if self.game_over {
            return;
        }

        self.ticks += 1;
        self.score += 1;
        if self.score > self.high_score {
            self.high_score = self.score;
        }

        // Slowly increase speed up to max 2.2
        self.speed = BASE_SPEED + (self.score as f32 / 1500.0).min(1.2);

        self.dino.tick(Dino::GRAVITY);
        self.spawner.tick(self.speed, &mut self.obstacles, TRACK_SPAWN_X);

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
        // Find closest obstacle in front of Dino
        let closest = self
            .obstacles
            .iter()
            .filter(|obs| obs.x + (obs.width as f32) > DINO_X)
            .min_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

        let (mut dist, w, h) = if let Some(obs) = closest {
            let d = (obs.x - DINO_X).max(0.0);
            (d, obs.width, obs.height)
        } else {
            (999.0, 0, 0)
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
            dino_is_grounded: self.dino.is_grounded,
            speed: (self.speed * 100.0).round() / 100.0,
            latency_budget_ms,
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test test_state`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/game/state.rs tests/test_state.rs
git commit -m "feat(engine): add GameState and DinoObservation projection"
```

---

### Task 4: Ratatui Sprites & Terminal Rendering UI

**Files:**
- Create: `src/tui/sprites.rs`
- Create: `src/tui/ui.rs`
- Create: `src/tui/mod.rs`

**Interfaces:**
- Consumes: `GameState` from `src/game/state.rs`
- Produces:
  - `render_ui<B: Backend>(f: &mut Frame, state: &GameState, stats: &TelemetryStats)`

- [ ] **Step 1: Implement ASCII Sprites**

Create `src/tui/mod.rs`:
```rust
pub mod sprites;
pub mod ui;
```

Create `src/tui/sprites.rs`:
```rust
pub const DINO_RUN_1: [&str; 4] = [
    "   .---. ",
    "  / _ _ \\",
    "==|  __ /",
    "  / / \\  ",
];

pub const DINO_RUN_2: [&str; 4] = [
    "   .---. ",
    "  / _ _ \\",
    "==|  __ /",
    "   \\ \\/  ",
];

pub const DINO_JUMP: [&str; 4] = [
    "   .---. ",
    "  / _ _ \\",
    "==|  __ /",
    "  (____) ",
];

pub const DINO_DEAD: [&str; 4] = [
    "   .---. ",
    "  / x x \\",
    "==|  __ /",
    "   /  \\  ",
];

pub const CACTUS_SMALL: [&str; 2] = [
    " _|_ ",
    "| | |",
];

pub const CACTUS_LARGE: [&str; 3] = [
    "  |  ",
    " _|_ ",
    "| | |",
];

pub const CACTUS_CLUSTER: [&str; 2] = [
    " _|_  _|_ ",
    "| | || | |",
];
```

- [ ] **Step 2: Implement UI layout and widgets**

Create `src/tui/ui.rs`:
```rust
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Sparkline, Wrap},
    Frame,
};

use crate::game::obstacle::ObstacleType;
use crate::game::state::{GameState, DINO_X};
use crate::tui::sprites::*;

pub struct TelemetryStats {
    pub controller_name: String,
    pub mode_name: String,
    pub last_action: String,
    pub last_latency_ms: f32,
    pub p50_ms: f32,
    pub p90_ms: f32,
    pub p99_ms: f32,
    pub jitter_ms: f32,
    pub deadline_misses: u32,
    pub confidence: f32,
    pub latency_history: Vec<u64>,
}

pub fn render_ui(frame: &mut Frame, state: &GameState, stats: &TelemetryStats) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header stats
            Constraint::Min(12),    // Game canvas
            Constraint::Length(10), // Bottom telemetry & controls
        ])
        .split(frame.area());

    // 1. Header
    render_header(frame, chunks[0], state, stats);

    // 2. Game Canvas
    render_canvas(frame, chunks[1], state);

    // 3. Bottom Panels
    let bottom_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[2]);

    render_agent_panel(frame, bottom_chunks[0], stats);
    render_latency_panel(frame, bottom_chunks[1], stats);
}

fn render_header(frame: &mut Frame, area: Rect, state: &GameState, stats: &TelemetryStats) {
    let header_text = vec![Line::from(vec![
        Span::styled(" SCORE: ", Style::default().fg(Color::Yellow)),
        Span::styled(format!("{:05} ", state.score), Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(" HI: ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{:05} ", state.high_score), Style::default()),
        Span::styled(" SPEED: ", Style::default().fg(Color::Cyan)),
        Span::styled(format!("{:.2} ", state.speed), Style::default()),
        Span::styled(" DRIVER: ", Style::default().fg(Color::Green)),
        Span::styled(
            format!("[{}] ", stats.controller_name),
            Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("[{}]", stats.mode_name),
            Style::default().fg(Color::Magenta),
        ),
    ])];

    let header_widget = Paragraph::new(header_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" CHROME DINO LAG-COMPENSATED BENCHMARK "),
    );
    frame.render_widget(header_widget, area);
}

fn render_canvas(frame: &mut Frame, area: Rect, state: &GameState) {
    let block = Block::default().borders(Borders::ALL).title(" RUNWAY ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height < 6 {
        return;
    }

    let ground_y = inner.bottom() - 2;
    let track_width = inner.width as usize;

    // Draw scrolling ground
    let mut ground_str = String::with_capacity(track_width);
    for x in 0..track_width {
        let cycle = (x + (state.ticks as usize)) % 13;
        if cycle == 0 {
            ground_str.push('.');
        } else if cycle == 7 {
            ground_str.push('_');
        } else {
            ground_str.push('=');
        }
    }
    frame.render_widget(
        Paragraph::new(ground_str).style(Style::default().fg(Color::DarkGray)),
        Rect::new(inner.left(), ground_y, inner.width, 1),
    );

    // Draw Dino
    let dino_sprite = if state.game_over {
        &DINO_DEAD
    } else if !state.dino.is_grounded {
        &DINO_JUMP
    } else if (state.ticks / 4) % 2 == 0 {
        &DINO_RUN_1
    } else {
        &DINO_RUN_2
    };

    let dino_screen_x = inner.left() + (DINO_X as u16);
    let dino_screen_y = ground_y.saturating_sub(4 + (state.dino.y as u16));

    for (i, line) in dino_sprite.iter().enumerate() {
        let y = dino_screen_y + (i as u16);
        if y < ground_y {
            frame.render_widget(
                Paragraph::new(*line).style(Style::default().fg(Color::Green)),
                Rect::new(dino_screen_x, y, 9, 1),
            );
        }
    }

    // Draw Obstacles
    for obs in &state.obstacles {
        let obs_screen_x = inner.left() + (obs.x as u16);
        if obs_screen_x >= inner.right() {
            continue;
        }

        let sprite: &[&str] = match obs.obstacle_type {
            ObstacleType::SmallCactus => &CACTUS_SMALL,
            ObstacleType::LargeCactus => &CACTUS_LARGE,
            ObstacleType::Cluster => &CACTUS_CLUSTER,
        };

        let obs_y = ground_y.saturating_sub(obs.height);
        for (i, line) in sprite.iter().enumerate() {
            let y = obs_y + (i as u16);
            if y < ground_y && obs_screen_x < inner.right() {
                let max_len = (inner.right() - obs_screen_x) as usize;
                let visible_slice = if line.len() > max_len {
                    &line[..max_len]
                } else {
                    line
                };
                frame.render_widget(
                    Paragraph::new(visible_slice).style(Style::default().fg(Color::Red)),
                    Rect::new(obs_screen_x, y, visible_slice.len() as u16, 1),
                );
            }
        }
    }

    // Game Over Banner
    if state.game_over {
        let banner = Paragraph::new("=== GAME OVER (Press [R] to Restart) ===")
            .style(Style::default().fg(Color::LightRed).add_modifier(Modifier::BOLD));
        let banner_area = Rect::new(
            inner.left() + (inner.width.saturating_sub(40) / 2),
            inner.top() + (inner.height / 3),
            40,
            1,
        );
        frame.render_widget(banner, banner_area);
    }
}

fn render_agent_panel(frame: &mut Frame, area: Rect, stats: &TelemetryStats) {
    let text = vec![
        Line::from(vec![
            Span::styled("Last Action: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&stats.last_action, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" (Conf: {:.1}%)", stats.confidence * 100.0), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("Inference: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.1} ms", stats.last_latency_ms), Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[Space/Up]", Style::default().fg(Color::Yellow)),
            Span::raw(" Jump   "),
            Span::styled("[m/Tab]", Style::default().fg(Color::Yellow)),
            Span::raw(" Switch Model"),
        ]),
        Line::from(vec![
            Span::styled("[h]", Style::default().fg(Color::Yellow)),
            Span::raw(" Toggle Round 1/2   "),
            Span::styled("[r]", Style::default().fg(Color::Yellow)),
            Span::raw(" Reset   "),
            Span::styled("[q]", Style::default().fg(Color::Yellow)),
            Span::raw(" Quit"),
        ]),
    ];

    let widget = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title(" AGENT TELEMETRY & CONTROLS "))
        .wrap(Wrap { trim: false });
    frame.render_widget(widget, area);
}

fn render_latency_panel(frame: &mut Frame, area: Rect, stats: &TelemetryStats) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(3)])
        .split(area);

    let stats_text = vec![Line::from(vec![
        Span::styled(format!("p50: {:.1}ms ", stats.p50_ms), Style::default().fg(Color::Green)),
        Span::styled(format!("p90: {:.1}ms ", stats.p90_ms), Style::default().fg(Color::Yellow)),
        Span::styled(format!("p99: {:.1}ms ", stats.p99_ms), Style::default().fg(Color::Red)),
        Span::styled(format!("Sigma: {:.1}ms ", stats.jitter_ms), Style::default().fg(Color::Cyan)),
        Span::styled(format!("Misses: {}", stats.deadline_misses), Style::default().fg(Color::LightRed)),
    ])];

    let stats_widget = Paragraph::new(stats_text).block(
        Block::default().borders(Borders::ALL).title(" JITTER & LATENCY STATS "),
    );
    frame.render_widget(stats_widget, chunks[0]);

    let sparkline = Sparkline::default()
        .block(Block::default().borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM))
        .data(&stats.latency_history)
        .max(60)
        .style(Style::default().fg(Color::LightCyan));
    frame.render_widget(sparkline, chunks[1]);
}
```

- [ ] **Step 3: Update `src/lib.rs` and verify clean compilation**

Add `pub mod tui;` to `src/lib.rs`.
Run: `cargo check`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/tui/sprites.rs src/tui/ui.rs src/tui/mod.rs src/lib.rs
git commit -m "feat(tui): implement Ratatui ASCII canvas, HUD, and latency sparkline"
```

---

### Task 5: Python Model Inference Worker

**Files:**
- Create: `python/worker.py`
- Create: `python/pyproject.toml`
- Create: `python/test_worker.py`

**Interfaces:**
- Produces:
  - CLI script `python/worker.py` reading stdin line-by-line JSON requests:
    `{"id": u64, "model": "laya"|"vega", "mode": "raw"|"fixed_horizon", "state": DinoObservation}`
  - Emits JSON to stdout:
    `{"id": u64, "action": "jump"|"run", "confidence": f32, "inference_ms": f32, "model": str}`

- [ ] **Step 1: Create Python configuration and worker script**

Create `python/pyproject.toml`:
```toml
[project]
name = "layadino-worker"
version = "0.1.0"
description = "Inference worker for Laya and Vega decision models"
requires-python = ">=3.12"
dependencies = [
    "laya>=0.3.6",
    "vegaml>=0.9.0",
]
```

Create `python/worker.py`:
```python
#!/usr/bin/env python3
import json
import os
import sys
import time

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Models lazily initialized
laya_model = None
vega_model = None

LAYA_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": "Chrome Dino game. Pick JUMP or RUN to survive.",
        "criteria": {
            "jump": "Cactus is approaching and Dino is grounded. Must jump over obstacle now.",
            "run": "Obstacle is far away, already cleared, or Dino is airborne.",
        },
    }
}

VEGA_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": "Avoid cactus obstacle in Chrome Dino.",
        "criteria": {
            "jump": "Approaching cactus within jump trigger distance. Leap immediately.",
            "run": "Clear runway or safe distance. Keep running on ground.",
        },
    }
}


def get_laya():
    global laya_model
    if laya_model is None:
        import laya
        laya_model = laya.load("convaiinnovations/laya")
    return laya_model


def get_vega():
    global vega_model
    if vega_model is None:
        import vegaml
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        vega_model = vegaml.load(device=device)
    return vega_model


def decide_laya(state: dict) -> tuple[str, float]:
    model = get_laya()
    # State format for Laya
    formatted_state = {
        "cactus_dist": f"{state.get('distance_to_cactus', 999.0):.1f} cells",
        "grounded": "yes" if state.get("dino_is_grounded", True) else "no",
        "speed": f"{state.get('speed', 1.0):.2f}",
        "width": state.get("cactus_width", 2),
    }
    res = model.predict(state=formatted_state, questions=LAYA_QUESTION)
    action_res = res.get("action", {})
    action = action_res.get("value", "run")
    confidence = float(action_res.get("confidence", 0.9))
    return action, confidence


def decide_vega(state: dict) -> tuple[str, float]:
    model = get_vega()
    formatted_state = (
        f"Dino approaching cactus at distance {state.get('distance_to_cactus', 999.0):.1f}. "
        f"Width: {state.get('cactus_width', 2)}. Grounded: {state.get('dino_is_grounded', True)}."
    )
    res = model.decide(formatted_state, VEGA_QUESTION)
    action_res = res.get("action", {})
    action = action_res.get("value", "run")
    confidence = float(action_res.get("confidence", 0.9))
    return action, confidence


def main():
    # Signal readiness to Rust parent process
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id", 0)
            model_name = req.get("model", "laya").lower()
            state = req.get("state", {})

            t0 = time.perf_counter()
            if model_name == "vega":
                action, conf = decide_vega(state)
            else:
                action, conf = decide_laya(state)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            resp = {
                "id": req_id,
                "model": model_name,
                "action": action,
                "confidence": conf,
                "inference_ms": round(elapsed_ms, 2),
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {
                "id": req.get("id", 0) if "req" in locals() else 0,
                "error": str(e),
                "action": "run",
                "confidence": 0.0,
                "inference_ms": 0.0,
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write test script to verify worker pipe execution**

Create `python/test_worker.py`:
```python
import json
import subprocess
import sys

def test_worker_pipe():
    python_bin = "../laya-decision-brain/backend/.venv/bin/python"
    proc = subprocess.Popen(
        [python_bin, "python/worker.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # 1. Check ready
    ready_line = proc.stdout.readline()
    assert "ready" in ready_line, f"Unexpected ready: {ready_line}"

    # 2. Send request
    req = {
        "id": 1,
        "model": "laya",
        "state": {
            "distance_to_cactus": 14.0,
            "cactus_width": 2,
            "cactus_height": 2,
            "dino_is_grounded": True,
            "speed": 1.0,
            "latency_budget_ms": 0.0,
        },
    }
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()

    resp_line = proc.stdout.readline()
    resp = json.loads(resp_line)
    assert resp["id"] == 1
    assert "action" in resp
    assert resp["inference_ms"] > 0
    print(f"Laya inference response: {resp}")

    proc.terminate()
    print("Python worker verified successfully!")

if __name__ == "__main__":
    test_worker_pipe()
```

- [ ] **Step 3: Run the test script and verify worker communication**

Run: `../laya-decision-brain/backend/.venv/bin/python python/test_worker.py`
Expected: PASS ("Python worker verified successfully!")

- [ ] **Step 4: Commit**

```bash
git add python/worker.py python/pyproject.toml python/test_worker.py
git commit -m "feat(worker): implement standalone Python worker for Laya and Vega inference"
```

---

### Task 6: Rust Asynchronous IPC Client & Agent Driver

**Files:**
- Create: `src/ipc/client.rs`
- Create: `src/ipc/mod.rs`
- Create: `src/agent/telemetry.rs`
- Create: `src/agent/mod.rs`
- Create: `tests/test_telemetry.rs`

**Interfaces:**
- Produces:
  - `IpcWorker`: Spawns and manages child process, non-blocking `send_request`, non-blocking `try_recv_response`.
  - `Telemetry`: Tracks latency history, p50, p90, p99, and jitter standard deviation.
  - `DriverMode`: `Human`, `Laya`, `Vega`
  - `ExperimentRound`: `Round1RawAsync`, `Round2Fixed45msHold`

- [ ] **Step 1: Write failing unit test for Telemetry metrics**

Create `tests/test_telemetry.rs`:
```rust
use layadino::agent::telemetry::Telemetry;

#[test]
fn test_telemetry_percentiles_and_jitter() {
    let mut tel = Telemetry::new();
    for ms in [10.0, 20.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0] {
        tel.record(ms);
    }

    assert_eq!(tel.sample_count(), 10);
    assert!((tel.p50() - 40.0).abs() < 5.0);
    assert!(tel.p99() > 80.0);
    assert!(tel.jitter() > 0.0);
}

#[test]
fn test_deadline_misses_count() {
    let mut tel = Telemetry::new();
    tel.record(30.0);
    tel.record(48.0); // Miss (> 45.0)
    assert_eq!(tel.deadline_misses, 1);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test test_telemetry`
Expected: FAIL

- [ ] **Step 3: Implement Telemetry**

Create `src/agent/telemetry.rs`:
```rust
#[derive(Debug, Clone, Default)]
pub struct Telemetry {
    pub samples: Vec<f32>,
    pub deadline_misses: u32,
    pub last_latency_ms: f32,
    pub last_confidence: f32,
    pub last_action: String,
}

impl Telemetry {
    pub const DEADLINE_MS: f32 = 45.0;

    pub fn new() -> Self {
        Self {
            samples: Vec::with_capacity(1000),
            deadline_misses: 0,
            last_latency_ms: 0.0,
            last_confidence: 0.0,
            last_action: "RUN".to_string(),
        }
    }

    pub fn record(&mut self, latency_ms: f32) {
        self.last_latency_ms = latency_ms;
        self.samples.push(latency_ms);
        if latency_ms > Self::DEADLINE_MS {
            self.deadline_misses += 1;
        }
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn p50(&self) -> f32 {
        self.percentile(0.50)
    }

    pub fn p90(&self) -> f32 {
        self.percentile(0.90)
    }

    pub fn p99(&self) -> f32 {
        self.percentile(0.99)
    }

    pub fn percentile(&self, p: f32) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((sorted.len() as f32 * p).ceil() as usize).saturating_sub(1);
        sorted[idx.min(sorted.len() - 1)]
    }

    pub fn jitter(&self) -> f32 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mean = self.samples.iter().sum::<f32>() / self.samples.len() as f32;
        let variance = self.samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>()
            / self.samples.len() as f32;
        variance.sqrt()
    }

    pub fn recent_sparkline_data(&self, max_points: usize) -> Vec<u64> {
        let n = self.samples.len();
        let start = n.saturating_sub(max_points);
        self.samples[start..]
            .iter()
            .map(|&s| s.round() as u64)
            .collect()
    }
}
```

- [ ] **Step 4: Implement IPC Worker client**

Create `src/ipc/mod.rs`:
```rust
pub mod client;
```

Create `src/ipc/client.rs`:
```rust
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use serde::{Deserialize, Serialize};
use crate::game::state::DinoObservation;

#[derive(Debug, Clone, Serialize)]
pub struct ModelRequest {
    pub id: u64,
    pub model: String,
    pub mode: String,
    pub state: DinoObservation,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelResponse {
    pub id: u64,
    #[serde(default)]
    pub model: String,
    pub action: String,
    pub confidence: f32,
    pub inference_ms: f32,
    #[serde(default)]
    pub error: Option<String>,
}

pub struct IpcWorker {
    _child: Child,
    req_tx: Sender<ModelRequest>,
    resp_rx: Receiver<ModelResponse>,
    next_id: u64,
}

impl IpcWorker {
    pub fn spawn(python_bin: &str, script_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut child = Command::new(python_bin)
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let mut stdin = child.stdin.take().expect("Failed to open child stdin");
        let stdout = child.stdout.take().expect("Failed to open child stdout");

        let (req_tx, req_rx) = channel::<ModelRequest>();
        let (resp_tx, resp_rx) = channel::<ModelResponse>();

        // Stdin writing thread
        thread::spawn(move || {
            while let Ok(req) = req_rx.recv() {
                if let Ok(line) = serde_json::to_string(&req) {
                    if writeln!(stdin, "{}", line).is_err() || stdin.flush().is_err() {
                        break;
                    }
                }
            }
        });

        // Stdout reading thread
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                if line.contains("\"ready\"") {
                    continue;
                }
                if let Ok(resp) = serde_json::from_str::<ModelResponse>(&line) {
                    let _ = resp_tx.send(resp);
                }
            }
        });

        Ok(Self {
            _child: child,
            req_tx,
            resp_rx,
            next_id: 1,
        })
    }

    pub fn send_request(
        &mut self,
        model: &str,
        mode: &str,
        state: DinoObservation,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let req = ModelRequest {
            id,
            model: model.to_string(),
            mode: mode.to_string(),
            state,
        };
        let _ = self.req_tx.send(req);
        id
    }

    pub fn try_recv(&self) -> Option<ModelResponse> {
        self.resp_rx.try_recv().ok()
    }
}
```

- [ ] **Step 5: Implement Driver Agent and lag compensation scheduler**

Create `src/agent/mod.rs`:
```rust
pub mod telemetry;

use std::time::Instant;
use crate::agent::telemetry::Telemetry;
use crate::game::state::{DinoObservation, GameState};
use crate::ipc::client::{IpcWorker, ModelResponse};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverMode {
    Human,
    Laya,
    Vega,
}

impl DriverMode {
    pub fn name(&self) -> &'static str {
        match self {
            DriverMode::Human => "HUMAN",
            DriverMode::Laya => "LAYA (0.8B)",
            DriverMode::Vega => "VEGA (0.8B)",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            DriverMode::Human => DriverMode::Laya,
            DriverMode::Laya => DriverMode::Vega,
            DriverMode::Vega => DriverMode::Human,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentRound {
    Round1RawAsync,
    Round2Fixed45msHold,
}

impl ExperimentRound {
    pub fn name(&self) -> &'static str {
        match self {
            ExperimentRound::Round1RawAsync => "Round 1 (Raw Async)",
            ExperimentRound::Round2Fixed45msHold => "Round 2 (Fixed 45ms Hold)",
        }
    }

    pub fn toggle(&self) -> Self {
        match self {
            ExperimentRound::Round1RawAsync => ExperimentRound::Round2Fixed45msHold,
            ExperimentRound::Round2Fixed45msHold => ExperimentRound::Round1RawAsync,
        }
    }
}

pub struct ScheduledAction {
    pub execute_at: Instant,
    pub action: String,
}

pub struct AgentManager {
    pub driver: DriverMode,
    pub round: ExperimentRound,
    pub worker: Option<IpcWorker>,
    pub telemetry_laya: Telemetry,
    pub telemetry_vega: Telemetry,
    in_flight_req: Option<(u64, Instant)>,
    scheduled_action: Option<ScheduledAction>,
}

impl AgentManager {
    pub fn new(worker: Option<IpcWorker>) -> Self {
        Self {
            driver: DriverMode::Human,
            round: ExperimentRound::Round1RawAsync,
            worker,
            telemetry_laya: Telemetry::new(),
            telemetry_vega: Telemetry::new(),
            in_flight_req: None,
            scheduled_action: None,
        }
    }

    pub fn active_telemetry(&self) -> &Telemetry {
        match self.driver {
            DriverMode::Vega => &self.telemetry_vega,
            _ => &self.telemetry_laya,
        }
    }

    pub fn active_telemetry_mut(&mut self) -> &mut Telemetry {
        match self.driver {
            DriverMode::Vega => &mut self.telemetry_vega,
            _ => &mut self.telemetry_laya,
        }
    }

    pub fn update(&mut self, state: &GameState) -> Option<String> {
        let now = Instant::now();

        // 1. Check if a scheduled action is due (Round 2 Hold)
        if let Some(ref sched) = self.scheduled_action {
            if now >= sched.execute_at {
                let action = sched.action.clone();
                self.scheduled_action = None;
                return Some(action);
            }
        }

        // 2. Poll for IPC worker responses
        let maybe_resp = if let Some(ref worker) = self.worker {
            worker.try_recv()
        } else {
            None
        };

        if let Some(resp) = maybe_resp {
            let latency = resp.inference_ms;
            let conf = resp.confidence;
            let action = resp.action.to_uppercase();

            let tel = self.active_telemetry_mut();
            tel.record(latency);
            tel.last_confidence = conf;
            tel.last_action = action.clone();

            let req_info = self.in_flight_req.take();

            match self.round {
                ExperimentRound::Round1RawAsync => {
                    // Apply immediately!
                    if action == "JUMP" {
                        return Some(action);
                    }
                }
                ExperimentRound::Round2Fixed45msHold => {
                    if action == "JUMP" {
                        let dispatch_time = req_info.map(|(_, t)| t).unwrap_or(now);
                        let execute_at = dispatch_time + std::time::Duration::from_millis(45);
                        if now >= execute_at {
                            return Some(action);
                        } else {
                            self.scheduled_action = Some(ScheduledAction {
                                execute_at,
                                action,
                            });
                        }
                    }
                }
            }
        }

        // 3. Dispatch new request if AI driver is active and not currently waiting on in-flight request
        if self.driver != DriverMode::Human && self.in_flight_req.is_none() && !state.game_over {
            let budget_ms = match self.round {
                ExperimentRound::Round1RawAsync => 0.0,
                ExperimentRound::Round2Fixed45msHold => 45.0,
            };
            let obs = state.get_observation(budget_ms);

            // Only trigger inference if obstacle is within decision horizon (distance <= 35)
            if obs.distance_to_cactus <= 35.0 {
                let model_str = match self.driver {
                    DriverMode::Vega => "vega",
                    _ => "laya",
                };
                let mode_str = match self.round {
                    ExperimentRound::Round1RawAsync => "raw",
                    ExperimentRound::Round2Fixed45msHold => "fixed_horizon",
                };

                if let Some(ref mut worker) = self.worker {
                    let id = worker.send_request(model_str, mode_str, obs);
                    self.in_flight_req = Some((id, now));
                }
            }
        }

        None
    }
}
```

- [ ] **Step 6: Update `src/lib.rs` and verify tests**

Add `pub mod ipc;` and `pub mod agent;` to `src/lib.rs`.
Run: `cargo test --test test_telemetry`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/agent/telemetry.rs src/ipc/client.rs src/ipc/mod.rs src/agent/mod.rs tests/test_telemetry.rs src/lib.rs
git commit -m "feat(ipc): implement asynchronous worker IPC client and agent lag compensation"
```

---

### Task 7: Main Game Loop & Interactive CLI Application

**Files:**
- Create: `src/main.rs`

**Interfaces:**
- Connects crossterm event loop, strict 30 FPS timer, `GameState`, `AgentManager`, and `ratatui` UI rendering.

- [ ] **Step 1: Implement main loop**

Create `src/main.rs`:
```rust
use std::io;
use std::time::{Duration, Instant};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use layadino::agent::{AgentManager, DriverMode};
use layadino::game::state::GameState;
use layadino::ipc::client::IpcWorker;
use layadino::tui::ui::{render_ui, TelemetryStats};

const TICK_DURATION: Duration = Duration::from_micros(33_333); // 30 FPS = 33.333 ms

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Try to start Python IPC worker
    let python_bin = if std::path::Path::new("../laya-decision-brain/backend/.venv/bin/python").exists() {
        "../laya-decision-brain/backend/.venv/bin/python"
    } else if std::path::Path::new("../vega/.venv/bin/python").exists() {
        "../vega/.venv/bin/python"
    } else {
        "python3"
    };

    let worker = match IpcWorker::spawn(python_bin, "python/worker.py") {
        Ok(w) => Some(w),
        Err(e) => {
            eprintln!("Warning: could not spawn python worker: {e}. AI autopilot will be disabled.");
            None
        }
    };

    // 2. Setup Terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // 3. Initialize Game & Agent Manager
    let mut game_state = GameState::new(1337);
    let mut agent_mgr = AgentManager::new(worker);

    let mut last_tick = Instant::now();

    // 4. Main Loop
    loop {
        // Poll user input (hotkeys & human jump)
        if event::poll(Duration::from_millis(0))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('r') => game_state.reset(),
                        KeyCode::Char('m') | KeyCode::Tab => {
                            agent_mgr.driver = agent_mgr.driver.next();
                        }
                        KeyCode::Char('h') => {
                            agent_mgr.round = agent_mgr.round.toggle();
                        }
                        KeyCode::Char(' ') | KeyCode::Up => {
                            if agent_mgr.driver == DriverMode::Human {
                                game_state.jump();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Agent updates
        if let Some(action) = agent_mgr.update(&game_state) {
            if action == "JUMP" {
                game_state.jump();
            }
        }

        // Physics tick (Strict 30 FPS)
        let now = Instant::now();
        if now.duration_since(last_tick) >= TICK_DURATION {
            game_state.tick();
            last_tick = now;
        }

        // Prepare telemetry stats for UI
        let active_tel = agent_mgr.active_telemetry();
        let stats = TelemetryStats {
            controller_name: agent_mgr.driver.name().to_string(),
            mode_name: agent_mgr.round.name().to_string(),
            last_action: active_tel.last_action.clone(),
            last_latency_ms: active_tel.last_latency_ms,
            p50_ms: active_tel.p50(),
            p90_ms: active_tel.p90(),
            p99_ms: active_tel.p99(),
            jitter_ms: active_tel.jitter(),
            deadline_misses: active_tel.deadline_misses,
            confidence: active_tel.last_confidence,
            latency_history: active_tel.recent_sparkline_data(30),
        };

        // Render Frame
        terminal.draw(|f| render_ui(f, &game_state, &stats))?;

        // Precision frame sleep / yield
        let elapsed = now.elapsed();
        if elapsed < TICK_DURATION {
            std::thread::sleep(TICK_DURATION - elapsed);
        }
    }

    // Restore Terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
```

- [ ] **Step 2: Build the project and verify release compilation**

Run: `cargo build`
Expected: PASS with 0 errors

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat(app): wire up main 30 FPS game loop and terminal lifecycle"
```

---

### Task 8: End-to-End Verification & Benchmark Validation

**Files:**
- Create: `tests/test_e2e_integration.rs`

- [ ] **Step 1: Write an end-to-end simulation test**

Create `tests/test_e2e_integration.rs`:
```rust
use layadino::game::state::GameState;

#[test]
fn test_simulated_gameplay_survives_cacti() {
    let mut state = GameState::new(42);

    // Simulate 300 frames (~10 seconds)
    for _ in 0..300 {
        let obs = state.get_observation(0.0);
        // Basic heuristic: jump if cactus is close (dist <= 15) and grounded
        if obs.distance_to_cactus <= 15.0 && obs.dino_is_grounded {
            state.jump();
        }
        state.tick();
    }

    // Should survive 300 frames and have a positive score
    assert!(!state.game_over);
    assert_eq!(state.score, 300);
}
```

- [ ] **Step 2: Run all tests in the workspace**

Run: `cargo test`
Expected: ALL tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_integration.rs
git commit -m "test: add end-to-end simulated gameplay test"
```
