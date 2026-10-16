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

    // Distance from Dino front body (12.0) to cactus (25.0) = 13.0
    let obs_raw = state.get_observation(0.0);
    assert!((obs_raw.distance_to_cactus - 13.0).abs() < 0.1);

    // Projected with 45ms latency budget
    let obs_projected = state.get_observation(45.0);
    assert!(obs_projected.distance_to_cactus < obs_raw.distance_to_cactus);
    assert_eq!(obs_projected.latency_budget_ms, 45.0);
}
