use layadino::game::state::GameState;

#[test]
fn test_simulated_gameplay_survives_cacti() {
    let mut state = GameState::new(42);

    for frame in 0..400 {
        let obs = state.get_observation(0.0);
        // Generous jump window: 2 to 10 cells ahead
        if obs.distance_to_cactus <= 10.0 && obs.distance_to_cactus >= 2.0 && obs.dino_is_grounded {
            state.jump();
        }
        state.tick();
        if state.game_over {
            panic!("Dino hit an obstacle at frame {frame}, score {}", state.score);
        }
    }

    assert!(!state.game_over);
    assert_eq!(state.score, 400);
}

#[test]
fn test_lag_compensated_decision_horizon() {
    let mut state = GameState::new(42);
    state.obstacles.push(layadino::game::obstacle::Obstacle::new(
        22.0,
        layadino::game::obstacle::ObstacleType::SmallCactus,
    ));

    let raw = state.get_observation(0.0);
    let compensated = state.get_observation(45.0);

    assert!(compensated.distance_to_cactus < raw.distance_to_cactus);
    assert_eq!(compensated.latency_budget_ms, 45.0);
}
