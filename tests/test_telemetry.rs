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
