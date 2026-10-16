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
