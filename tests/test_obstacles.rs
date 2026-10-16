use layadino::game::physics::Dino;
use layadino::game::obstacle::{check_collision, Obstacle, ObstacleType};

#[test]
fn test_collision_when_grounded_and_overlapping() {
    let dino = Dino::new(); // y = 0.0
    let obstacles = vec![Obstacle {
        x: 7.0,
        width: 3,
        height: 2,
        obstacle_type: ObstacleType::SmallCactus,
    }];

    // Dino is at x=6, torso at [8.0, 12.5], cactus at [7.0, 10.0] -> overlap!
    assert!(check_collision(&dino, 6.0, 4.0, 3.0, &obstacles));
}

#[test]
fn test_no_collision_when_dino_jumped_over_obstacle() {
    let mut dino = Dino::new();
    dino.y = 5.0; // High in the air
    dino.is_grounded = false;

    let obstacles = vec![Obstacle {
        x: 7.0,
        width: 3,
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
        width: 3,
        height: 2,
        obstacle_type: ObstacleType::SmallCactus,
    };
    obs.step(1.5);
    assert_eq!(obs.x, 18.5);
}
