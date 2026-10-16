use std::io;
use std::panic;
use std::path::Path;
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

const TICK_DURATION: Duration = Duration::from_micros(33_333); // Strict 30 FPS (33.333 ms)

fn setup_panic_hook() {
    let original_hook = panic::take_hook();
    panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(panic_info);
    }));
}

fn detect_python() -> &'static str {
    if Path::new(".venv/bin/python").exists() {
        ".venv/bin/python"
    } else if Path::new("../laya-decision-brain/backend/.venv/bin/python").exists() {
        "../laya-decision-brain/backend/.venv/bin/python"
    } else if Path::new("../vega/.venv/bin/python").exists() {
        "../vega/.venv/bin/python"
    } else {
        "python3"
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_panic_hook();

    let python_bin = detect_python();
    let worker = match IpcWorker::spawn(python_bin, "python/worker.py") {
        Ok(w) => Some(w),
        Err(e) => {
            eprintln!("Warning: could not spawn python worker: {e}. Model autopilot disabled.");
            None
        }
    };

    // Terminal Initialization
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut game_state = GameState::new(1337);
    let mut agent_mgr = AgentManager::new(worker);

    let mut current_time = Instant::now();
    let mut accumulator = Duration::ZERO;

    loop {
        let frame_start = Instant::now();
        let frame_time = frame_start
            .duration_since(current_time)
            .min(Duration::from_millis(100)); // clamp to avoid spiral
        current_time = frame_start;
        accumulator += frame_time;

        // 1. Non-blocking Key Event Poll (drain all pending keypresses)
        while event::poll(Duration::from_millis(0))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            // Clean exit
                            disable_raw_mode()?;
                            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                            terminal.show_cursor()?;
                            return Ok(());
                        }
                        KeyCode::Char('r') => game_state.reset(),
                        KeyCode::Char('m') | KeyCode::Tab => {
                            agent_mgr.driver = agent_mgr.driver.next();
                        }
                        KeyCode::Char('h') => {
                            agent_mgr.round = agent_mgr.round.toggle();
                        }
                        KeyCode::Char(' ') | KeyCode::Up => {
                            if game_state.game_over {
                                game_state.reset();
                            } else if agent_mgr.driver == DriverMode::Human {
                                game_state.jump();
                            }
                        }
                        KeyCode::Down | KeyCode::Char('s') => {
                            if !game_state.game_over && agent_mgr.driver == DriverMode::Human {
                                game_state.duck();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // 2. Fixed Timestep Physics Simulation
        let mut did_tick = false;
        while accumulator >= TICK_DURATION {
            if let Some(action) = agent_mgr.update(&game_state) {
                if action == "JUMP" {
                    game_state.jump();
                } else if action == "DUCK" {
                    game_state.duck();
                }
            }

            game_state.tick();
            accumulator -= TICK_DURATION;
            did_tick = true;
        }

        // 3. Render on tick
        if did_tick {
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

            terminal.draw(|f| render_ui(f, &game_state, &stats))?;
        }

        // 4. High-Precision Frame Sleep + Micro-Yield for Sub-Millisecond Pacing
        let elapsed = frame_start.elapsed();
        if elapsed < TICK_DURATION {
            let remaining = TICK_DURATION - elapsed;
            if remaining > Duration::from_millis(3) {
                std::thread::sleep(remaining - Duration::from_millis(3));
            }
            while frame_start.elapsed() < TICK_DURATION {
                std::thread::yield_now();
            }
        }
    }
}
