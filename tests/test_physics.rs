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

    // Advance ticks with Dino::GRAVITY
    let mut reached_apex = false;
    for _ in 0..25 {
        dino.tick(Dino::GRAVITY);
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

#[test]
fn test_dino_duck_and_timeout() {
    let mut dino = Dino::new();
    assert!(!dino.is_ducking);

    dino.duck();
    assert!(dino.is_ducking);
    assert_eq!(dino.duck_ticks_remaining, Dino::DEFAULT_DUCK_TICKS);

    // Ticking down
    for _ in 0..Dino::DEFAULT_DUCK_TICKS {
        dino.tick(Dino::GRAVITY);
    }
    assert!(!dino.is_ducking);
    assert_eq!(dino.duck_ticks_remaining, 0);
}

#[test]
fn test_dino_fast_fall_when_ducking() {
    let mut dino = Dino::new();
    dino.jump();
    let vy_before = dino.vy;

    dino.duck(); // In mid-air -> fast fall acceleration
    assert!(dino.vy < vy_before);
}

