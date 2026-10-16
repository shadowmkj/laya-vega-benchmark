pub mod telemetry;

use std::time::Instant;
use crate::agent::telemetry::Telemetry;
use crate::game::state::GameState;
use crate::ipc::client::IpcWorker;

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
                    if action == "JUMP" || action == "DUCK" {
                        return Some(action);
                    }
                }
                ExperimentRound::Round2Fixed45msHold => {
                    if action == "JUMP" || action == "DUCK" {
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
