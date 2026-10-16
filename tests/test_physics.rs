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
