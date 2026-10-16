use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use serde::{Deserialize, Serialize};
use crate::game::state::DinoObservation;

#[derive(Debug, Clone, Serialize)]
pub struct ModelRequest {
    pub id: u64,
    pub model: String,
    pub mode: String,
    pub state: DinoObservation,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelResponse {
    pub id: u64,
    #[serde(default)]
    pub model: String,
    pub action: String,
    pub confidence: f32,
    pub inference_ms: f32,
    #[serde(default)]
    pub error: Option<String>,
}

pub struct IpcWorker {
    _child: Child,
    req_tx: Sender<ModelRequest>,
    resp_rx: Receiver<ModelResponse>,
    next_id: u64,
}

impl IpcWorker {
    pub fn spawn(python_bin: &str, script_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut child = Command::new(python_bin)
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let mut stdin = child.stdin.take().expect("Failed to open child stdin");
        let stdout = child.stdout.take().expect("Failed to open child stdout");

        let (req_tx, req_rx) = channel::<ModelRequest>();
        let (resp_tx, resp_rx) = channel::<ModelResponse>();

        // Stdin writing thread
        thread::spawn(move || {
            while let Ok(req) = req_rx.recv() {
                if let Ok(line) = serde_json::to_string(&req) {
                    if writeln!(stdin, "{}", line).is_err() || stdin.flush().is_err() {
                        break;
                    }
                }
            }
        });

        // Stdout reading thread
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                if line.contains("\"ready\"") {
                    continue;
                }
                if let Ok(resp) = serde_json::from_str::<ModelResponse>(&line) {
                    let _ = resp_tx.send(resp);
                }
            }
        });

        Ok(Self {
            _child: child,
            req_tx,
            resp_rx,
            next_id: 1,
        })
    }

    pub fn send_request(
        &mut self,
        model: &str,
        mode: &str,
        state: DinoObservation,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let req = ModelRequest {
            id,
            model: model.to_string(),
            mode: mode.to_string(),
            state,
        };
        let _ = self.req_tx.send(req);
        id
    }

    pub fn try_recv(&self) -> Option<ModelResponse> {
        self.resp_rx.try_recv().ok()
    }
}
