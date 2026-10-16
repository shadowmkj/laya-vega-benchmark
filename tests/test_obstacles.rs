use layadino::game::physics::Dino;
use layadino::game::obstacle::{check_collision, FlightAltitude, Obstacle, ObstacleType};

#[test]
fn test_collision_when_grounded_and_overlapping() {
    let dino = Dino::new(); // y = 0.0
    let obstacles = vec![Obstacle::new(7.0, ObstacleType::SmallCactus)];

    // Dino is at x=6, torso at [8.0, 12.5], cactus at [7.0, 10.0] -> overlap!
    assert!(check_collision(&dino, 6.0, 4.0, 3.0, &obstacles));
}

#[test]
fn test_no_collision_when_dino_jumped_over_obstacle() {
    let mut dino = Dino::new();
    dino.y = 5.0; // High in the air
    dino.is_grounded = false;

    let obstacles = vec![Obstacle::new(7.0, ObstacleType::SmallCactus)];

    // Height of cactus is 2, dino foot is at y=5.0 -> safe
    assert!(!check_collision(&dino, 6.0, 4.0, 3.0, &obstacles));
}

#[test]
fn test_pterodactyl_mid_altitude_duck_under_safe() {
    let mut dino = Dino::new();
    dino.duck();
    assert!(dino.is_ducking);

    let obstacles = vec![Obstacle::new(7.0, ObstacleType::Pterodactyl(FlightAltitude::Mid))];

    // Ducking Dino top is 1.5, mid bird bottom is 1.8 -> safe underneath!
    assert!(!check_collision(&dino, 6.0, 4.0, 3.0, &obstacles));
}

#[test]
fn test_pterodactyl_mid_altitude_standing_collides() {
    let dino = Dino::new(); // Standing
    assert!(!dino.is_ducking);

    let obstacles = vec![Obstacle::new(7.0, ObstacleType::Pterodactyl(FlightAltitude::Mid))];

    // Standing Dino top is 3.2, collides with mid bird (1.8..3.5)
    assert!(check_collision(&dino, 6.0, 4.0, 3.0, &obstacles));
}

#[test]
fn test_pterodactyl_low_altitude_requires_jump() {
    let mut dino = Dino::new();
    dino.duck();

    let obstacles = vec![Obstacle::new(7.0, ObstacleType::Pterodactyl(FlightAltitude::Low))];

    // Ducking Dino cannot clear low bird at ground level
    assert!(check_collision(&dino, 6.0, 4.0, 3.0, &obstacles));

    // But jumping clears it
    dino.stand();
    dino.y = 3.0;
    dino.is_grounded = false;
    assert!(!check_collision(&dino, 6.0, 4.0, 3.0, &obstacles));
}

#[test]
fn test_obstacle_movement() {
    let mut obs = Obstacle::new(20.0, ObstacleType::SmallCactus);
    obs.step(1.5);
    assert_eq!(obs.x, 18.5);
}
