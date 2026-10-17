import React, { useEffect, useRef, useState } from 'react';
import { GameEngine } from './game/engine';
import { DriverMode, ExperimentRound, GameState } from './game/types';
import { checkBackendHealth, getBackendUrl, requestPrediction, setBackendUrl } from './telemetry/api';
import { MoveHistoryItem, MoveOptionProbabilities, TelemetryMetrics, TelemetryTracker } from './telemetry/stats';
import { Sparkline } from './telemetry/Sparkline';
import { Activity, Cpu, History, RotateCcw, Settings, ShieldAlert, Sparkles, Terminal, Trash2 } from 'lucide-react';

export const App: React.FC = () => {
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const engineRef = useRef<GameEngine | null>(null);

	const [driver, setDriver] = useState<DriverMode>('human');
	const [round, setRound] = useState<ExperimentRound>('raw_async');
	const [backendOnline, setBackendOnline] = useState<boolean>(false);
	const [backendUrl, setBackendUrlState] = useState<string>(getBackendUrl());
	const [isEditingUrl, setIsEditingUrl] = useState<boolean>(false);
	const [tempUrl, setTempUrl] = useState<string>(backendUrl);

	const handleSaveBackendUrl = (urlToSave: string) => {
		const clean = setBackendUrl(urlToSave);
		setBackendUrlState(clean);
		setIsEditingUrl(false);
		checkBackendHealth(clean).then((ok) => setBackendOnline(ok));
	};

	// Live game telemetry
	const [gameState, setGameState] = useState<GameState>({
		dinoY: 0,
		dinoVy: 0,
		isGrounded: true,
		isDucking: false,
		score: 0,
		highScore: 0,
		speed: 8.0,
		gameOver: false,
		obstacles: [],
		ticks: 0,
	});

	const trackerLayaRef = useRef(new TelemetryTracker());
	const trackerVegaRef = useRef(new TelemetryTracker());
	const [metrics, setMetrics] = useState<TelemetryMetrics>(trackerLayaRef.current.getMetrics());

	// Live move history & input transition tracking
	const [moveHistory, setMoveHistory] = useState<MoveHistoryItem[]>([]);
	const lastInputRef = useRef<string>('RUN');

	// In-flight request tracking, scheduled action queue & obstacle jump lockout
	const inFlightRef = useRef<boolean>(false);
	const scheduledTimerRef = useRef<number | null>(null);
	const jumpedObstaclesRef = useRef<Set<number>>(new Set());
	const lastSampleTimeRef = useRef<number>(0);

	const triggerReset = () => {
		jumpedObstaclesRef.current.clear();
		if (scheduledTimerRef.current) {
			clearTimeout(scheduledTimerRef.current);
			scheduledTimerRef.current = null;
		}
		inFlightRef.current = false;
		lastSampleTimeRef.current = 0;
		lastInputRef.current = 'RUN';
		engineRef.current?.reset();
	};

	// 1. Initialize GameEngine
	useEffect(() => {
		if (!canvasRef.current) return;
		const engine = new GameEngine(canvasRef.current);
		engineRef.current = engine;

		engine.onTick = (s) => {
			setGameState({ ...s });
			// Revert input tracking to RUN once Dino is grounded or finishes ducking
			if (lastInputRef.current === 'JUMP' && s.isGrounded) {
				lastInputRef.current = 'RUN';
			} else if (lastInputRef.current === 'DUCK' && !s.isDucking) {
				lastInputRef.current = 'RUN';
			}
			evaluateAutopilot(engine);
		};

		engine.start();

		// Check backend health
		checkBackendHealth().then((ok) => setBackendOnline(ok));
		const healthInterval = setInterval(() => {
			checkBackendHealth().then((ok) => setBackendOnline(ok));
		}, 5000);

		return () => {
			engine.stop();
			clearInterval(healthInterval);
			if (scheduledTimerRef.current) clearTimeout(scheduledTimerRef.current);
		};
	}, []);

	// 2. Autopilot Evaluation & Lag Compensation
	const evaluateAutopilot = async (engine: GameEngine) => {
		const currentDriver = driverRef.current;
		const currentRound = roundRef.current;

		if (currentDriver === 'human' || engine.state.gameOver) return;
		if (!engine.state.isGrounded) return; // Dino is airborne, cannot jump
		if (inFlightRef.current) return; // Wait for in-flight request to resolve

		const budgetMs = currentRound === 'fixed_45ms_hold' ? 45.0 : 0.0;
		const obs = engine.getObservation(budgetMs);

		// If this obstacle was already jumped over, do not jump again
		if (obs.obstacle_id && jumpedObstaclesRef.current.has(obs.obstacle_id)) {
			return;
		}

		// Prune older obstacle IDs to keep memory compact
		if (obs.obstacle_id > 10) {
			for (const id of jumpedObstaclesRef.current) {
				if (id < obs.obstacle_id - 5) {
					jumpedObstaclesRef.current.delete(id);
				}
			}
		}

		// High-Frequency Autopilot Decision Loop:
		// 1. Approaching obstacle (within 14 cells): continuous rapid-fire requests as fast as the model can respond!
		// 2. Clear track or distant obstacle: background telemetry heartbeat every ~350ms to keep RTT & sparkline alive.
		const isObstacleApproaching = obs.distance_to_cactus <= 14.0 && obs.distance_to_cactus > 0;
		const isHeartbeatDue = performance.now() - lastSampleTimeRef.current >= 350;

		if (isObstacleApproaching || isHeartbeatDue) {
			lastSampleTimeRef.current = performance.now();
			inFlightRef.current = true;
			const dispatchTime = performance.now();
			const currentObsId = obs.obstacle_id;

			try {
				const resp = await requestPrediction(currentDriver, obs);
				const tracker = currentDriver === 'vega' ? trackerVegaRef.current : trackerLayaRef.current;
				const clientRttMs = Math.round((performance.now() - dispatchTime) * 10) / 10;

				const probs: MoveOptionProbabilities = {
					jump: resp.probabilities?.jump ?? (resp.action === 'jump' ? resp.confidence : Math.max(0, (1 - resp.confidence) / 2)),
					duck: resp.probabilities?.duck ?? (resp.action === 'duck' ? resp.confidence : Math.max(0, (1 - resp.confidence) / 2)),
					run: resp.probabilities?.run ?? (resp.action === 'run' ? resp.confidence : Math.max(0, (1 - resp.confidence) / 2)),
				};

				tracker.record(resp.inference_ms, resp.confidence, resp.action, probs, clientRttMs);
				setMetrics(tracker.getMetrics());

				// Record move in history whenever input action changes or a new obstacle decision is evaluated
				const fromAction = lastInputRef.current;
				const toAction = resp.action.toUpperCase() as 'JUMP' | 'DUCK' | 'RUN';

				const historyEntry: MoveHistoryItem = {
					id: `${Date.now()}-${currentObsId ?? Math.random()}`,
					timestamp: Date.now(),
					timeString: new Date().toLocaleTimeString(undefined, {
						hour12: false,
						hour: '2-digit',
						minute: '2-digit',
						second: '2-digit',
					}) + '.' + String(Math.floor(performance.now() % 1000)).padStart(3, '0').slice(0, 2),
					driver: currentDriver,
					fromAction,
					action: toAction,
					confidence: resp.confidence,
					probabilities: probs,
					obstacleType: obs.obstacle_type,
					obstacleAltitude: obs.obstacle_altitude,
					distance: Math.round(obs.distance_to_cactus * 10) / 10,
					serverLatencyMs: resp.inference_ms,
					roundTripMs: clientRttMs,
				};

				setMoveHistory((prev) => [historyEntry, ...prev.slice(0, 49)]);
				lastInputRef.current = toAction;

				if (resp.action === 'jump') {
					// Execute jump when obstacle is within critical strike zone (<= 7.8 cells)
					if (engine.state.isGrounded && currentObsId && !jumpedObstaclesRef.current.has(currentObsId) && obs.distance_to_cactus <= 7.8) {
						if (currentRound === 'raw_async') {
							jumpedObstaclesRef.current.add(currentObsId);
							engine.jump();
						} else {
							// Round 2: Fixed 45ms Horizon Hold
							const elapsed = performance.now() - dispatchTime;
							const remainingHoldMs = Math.max(0, 45.0 - elapsed);

							if (remainingHoldMs === 0) {
								jumpedObstaclesRef.current.add(currentObsId);
								engine.jump();
							} else {
								if (scheduledTimerRef.current) clearTimeout(scheduledTimerRef.current);
								scheduledTimerRef.current = window.setTimeout(() => {
									if (engine.state.isGrounded && currentObsId && !jumpedObstaclesRef.current.has(currentObsId)) {
										jumpedObstaclesRef.current.add(currentObsId);
										engine.jump();
									}
								}, remainingHoldMs);
							}
						}
					}
				} else if (resp.action === 'duck') {
					// Execute duck when obstacle is within critical strike zone (<= 7.8 cells)
					if (engine.state.isGrounded && currentObsId && !jumpedObstaclesRef.current.has(currentObsId) && obs.distance_to_cactus <= 7.8) {
						if (currentRound === 'raw_async') {
							jumpedObstaclesRef.current.add(currentObsId);
							engine.duck(true);
						} else {
							const elapsed = performance.now() - dispatchTime;
							const remainingHoldMs = Math.max(0, 45.0 - elapsed);

							if (remainingHoldMs === 0) {
								jumpedObstaclesRef.current.add(currentObsId);
								engine.duck(true);
							} else {
								if (scheduledTimerRef.current) clearTimeout(scheduledTimerRef.current);
								scheduledTimerRef.current = window.setTimeout(() => {
									if (engine.state.isGrounded && currentObsId && !jumpedObstaclesRef.current.has(currentObsId)) {
										jumpedObstaclesRef.current.add(currentObsId);
										engine.duck(true);
									}
								}, remainingHoldMs);
							}
						}
					}
				} else if (resp.action === 'run') {
					// Only permanently mark handled if the obstacle is high overhead (safe to run under permanently).
					// Ground cacti and mid birds must NOT be locked out on distant 'run' so they can trigger when closer!
					if (currentObsId && obs.obstacle_type === 'pterodactyl' && obs.obstacle_altitude === 'high') {
						jumpedObstaclesRef.current.add(currentObsId);
					}
					// Only stand up if duck was not initiated as a timed maneuver
					if (engine.state.isDucking && !engine.isTimedDucking()) {
						engine.stand();
					}
				}
			} catch (err) {
				console.warn('Autopilot inference error:', err);
			} finally {
				inFlightRef.current = false;
			}
		}
	};

	// Keep refs current for the engine loop callback
	const driverRef = useRef(driver);
	driverRef.current = driver;
	const roundRef = useRef(round);
	roundRef.current = round;

	// 3. Keyboard Hotkeys
	useEffect(() => {
		const handleKeyDown = (e: KeyboardEvent) => {
			if (e.code === 'Space' || e.code === 'ArrowUp') {
				e.preventDefault();
				if (engineRef.current) {
					if (engineRef.current.state.gameOver) {
						triggerReset();
					} else if (driverRef.current === 'human') {
						engineRef.current.jump();
					}
				}
			} else if (e.code === 'ArrowDown' || e.key === 's') {
				e.preventDefault();
				if (engineRef.current && driverRef.current === 'human' && !engineRef.current.state.gameOver) {
					engineRef.current.duck(false); // continuous hold until keyup
				}
			} else if (e.key === 'm' || e.key === 'Tab') {
				e.preventDefault();
				setDriver((prev) => (prev === 'human' ? 'laya' : prev === 'laya' ? 'vega' : 'human'));
			} else if (e.key === 'h') {
				e.preventDefault();
				setRound((prev) => (prev === 'raw_async' ? 'fixed_45ms_hold' : 'raw_async'));
			} else if (e.key === 'r') {
				e.preventDefault();
				triggerReset();
			}
		};

		const handleKeyUp = (e: KeyboardEvent) => {
			if (e.code === 'ArrowDown' || e.key === 's') {
				if (engineRef.current && driverRef.current === 'human') {
					engineRef.current.stand();
				}
			}
		};

		const handleBlur = () => {
			if (engineRef.current && driverRef.current === 'human') {
				engineRef.current.stand();
			}
		};

		window.addEventListener('keydown', handleKeyDown);
		window.addEventListener('keyup', handleKeyUp);
		window.addEventListener('blur', handleBlur);
		return () => {
			window.removeEventListener('keydown', handleKeyDown);
			window.removeEventListener('keyup', handleKeyUp);
			window.removeEventListener('blur', handleBlur);
		};
	}, []);

	return (
		<div className="min-h-screen bg-[#07090e] text-zinc-100 flex flex-col items-center justify-center p-4 selection:bg-emerald-500/30">
			{/* Top Banner */}
			<div className="w-full max-w-7xl flex items-center justify-between pb-3 border-b border-zinc-800/80 mb-4 font-mono text-xs">
				<div className="flex items-center gap-2">
					<Terminal className="w-4 h-4 text-emerald-400" />
					<span className="font-bold tracking-wider text-emerald-400">LAYADINO // SYSTEM 1.0</span>
					<span className="text-zinc-600">|</span>
					<span className="text-zinc-400">Chrome Dino Typed Decision Model Benchmark</span>
				</div>
				<div className="flex items-center gap-2">
					{isEditingUrl ? (
						<form
							onSubmit={(e) => {
								e.preventDefault();
								handleSaveBackendUrl(tempUrl);
							}}
							className="flex items-center gap-1.5"
						>
							<input
								type="text"
								value={tempUrl}
								onChange={(e) => setTempUrl(e.target.value)}
								placeholder="http://192.168.1.xxx:8000"
								className="bg-zinc-900 border border-emerald-500/50 rounded px-2 py-0.5 text-[11px] text-emerald-300 font-mono outline-none w-48"
								autoFocus
							/>
							<button
								type="submit"
								className="bg-emerald-600 hover:bg-emerald-500 text-black px-2 py-0.5 rounded text-[11px] font-bold"
							>
								Save
							</button>
							<button
								type="button"
								onClick={() => {
									setTempUrl(backendUrl);
									setIsEditingUrl(false);
								}}
								className="bg-zinc-800 hover:bg-zinc-700 text-zinc-400 px-1.5 py-0.5 rounded text-[11px]"
							>
								✕
							</button>
						</form>
					) : (
						<div className="flex items-center gap-1.5">
							<span className="text-zinc-500">BACKEND:</span>
							<button
								onClick={() => {
									setTempUrl(backendUrl);
									setIsEditingUrl(true);
								}}
								title="Click to change backend host (e.g. RTX 3060 LAN IP)"
								className={`px-2 py-0.5 rounded text-[11px] font-semibold flex items-center gap-1.5 hover:opacity-90 transition cursor-pointer ${backendOnline
									? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
									: 'bg-red-500/10 text-red-400 border border-red-500/30'
									}`}
							>
								<span className={`w-1.5 h-1.5 rounded-full ${backendOnline ? 'bg-emerald-400' : 'bg-red-400'}`} />
								<span>{backendUrl.replace(/^https?:\/\//, '')}</span>
								<span className="text-[10px] text-zinc-500">{backendOnline ? '(ONLINE)' : '(OFFLINE)'}</span>
								<Settings className="w-3 h-3 text-zinc-400 ml-0.5" />
							</button>
						</div>
					)}
				</div>
			</div>

			{/* Main Container */}
			<div className="w-full max-w-7xl bg-zinc-950/90 rounded-lg terminal-border p-4 shadow-2xl relative overflow-hidden">
				{/* CRT Scanline Overlay */}
				<div className="absolute inset-0 crt-overlay pointer-events-none" />

				<div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-4 items-start">
					{/* Left Column: Game Viewport, Controls & Diagnostics */}
					<div className="flex flex-col space-y-3 min-w-0">
						{/* Header HUD Bar */}
						<div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-zinc-800 font-mono text-sm">
							<div className="flex items-center gap-4">
								<div>
									<span className="text-zinc-500 text-xs">SCORE: </span>
									<span className="text-amber-400 font-bold tracking-wider text-base">
										{String(gameState.score).padStart(5, '0')}
									</span>
								</div>
								<div>
									<span className="text-zinc-500 text-xs">HI: </span>
									<span className="text-zinc-400 font-semibold">
										{String(gameState.highScore).padStart(5, '0')}
									</span>
								</div>
								<div>
									<span className="text-zinc-500 text-xs">SPEED: </span>
									<span className="text-cyan-400 font-semibold">{gameState.speed.toFixed(2)}x</span>
								</div>
							</div>

							{/* Mode Badges */}
							<div className="flex items-center gap-2">
								<span className="px-2.5 py-1 rounded bg-zinc-900 border border-zinc-700 text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
									<Cpu className="w-3.5 h-3.5" />
									{driver === 'human'
										? 'DRIVER: [HUMAN]'
										: driver === 'laya'
											? 'DRIVER: [LAYA (0.8B)]'
											: 'DRIVER: [VEGA (0.8B)]'}
								</span>
								<span className="px-2.5 py-1 rounded bg-zinc-900 border border-zinc-700 text-xs font-semibold text-purple-400">
									{round === 'raw_async' ? 'ROUND 1: RAW ASYNC' : 'ROUND 2: FIXED 45ms HOLD'}
								</span>
							</div>
						</div>

						{/* 60 FPS Canvas Game Viewport */}
						<div
							className="relative w-full rounded border border-zinc-800/90 bg-[#0c0e14] overflow-hidden cursor-pointer"
							onClick={() => {
								if (engineRef.current) {
									if (engineRef.current.state.gameOver) {
										triggerReset();
									} else if (driver === 'human') {
										engineRef.current.jump();
									}
								}
							}}
						>
							<canvas
								ref={canvasRef}
								style={{ width: '100%', height: 'auto', display: 'block', aspectRatio: '800 / 240' }}
							/>
						</div>

						{/* Controls Toolbar */}
						<div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-zinc-800/80 font-mono text-xs">
							{/* Driver Switcher */}
							<div className="flex items-center gap-1.5">
								<span className="text-zinc-500 mr-1">CONTROLLER:</span>
								{(['human', 'laya', 'vega'] as DriverMode[]).map((m) => (
									<button
										key={m}
										onClick={() => setDriver(m)}
										className={`px-3 py-1.5 rounded transition font-semibold ${driver === m
											? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 shadow-sm'
											: 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
											}`}
									>
										{m.toUpperCase()}
									</button>
								))}
							</div>

							{/* Round Switcher */}
							<div className="flex items-center gap-1.5">
								<span className="text-zinc-500 mr-1">EXPERIMENT:</span>
								<button
									onClick={() => setRound('raw_async')}
									className={`px-3 py-1.5 rounded transition font-semibold ${round === 'raw_async'
										? 'bg-purple-500/20 text-purple-300 border border-purple-500/50'
										: 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
										}`}
								>
									1. Raw Async
								</button>
								<button
									onClick={() => setRound('fixed_45ms_hold')}
									className={`px-3 py-1.5 rounded transition font-semibold ${round === 'fixed_45ms_hold'
										? 'bg-purple-500/20 text-purple-300 border border-purple-500/50'
										: 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
										}`}
								>
									2. 45ms Horizon Hold
								</button>
							</div>

							{/* Restart button */}
							<button
								onClick={triggerReset}
								className="flex items-center gap-1 px-3 py-1.5 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-zinc-300 transition"
							>
								<RotateCcw className="w-3.5 h-3.5" />
								Reset [R]
							</button>
						</div>

						{/* Telemetry Dashboard Panels */}
						<div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-3 border-t border-zinc-800 font-mono text-xs">
							{/* Left: Model State & Decisions */}
							<div className="bg-zinc-900/60 p-3 rounded border border-zinc-800/80 flex flex-col justify-between">
								<div className="flex items-center justify-between pb-2 mb-2 border-b border-zinc-800/60">
									<span className="text-zinc-400 flex items-center gap-1.5 font-bold">
										<Sparkles className="w-3.5 h-3.5 text-amber-400" />
										MODEL DECISION ATTRIBUTION
									</span>
									<span className="text-zinc-500">
										{driver === 'human' ? 'MANUAL INPUT' : `${driver.toUpperCase()} AUTOPILOT`}
									</span>
								</div>

								<div className="space-y-1.5 text-zinc-300">
									<div className="flex justify-between">
										<span className="text-zinc-500">Last Action:</span>
										<span
											className={`font-bold ${metrics.lastAction === 'JUMP'
												? 'text-amber-400'
												: metrics.lastAction === 'DUCK'
													? 'text-cyan-400'
													: 'text-emerald-400'
												}`}
										>
											{metrics.lastAction}
										</span>
									</div>
									<div className="flex justify-between">
										<span className="text-zinc-500">Confidence:</span>
										<span className="text-cyan-400 font-semibold">
											{(metrics.lastConfidence * 100).toFixed(1)}%
											{metrics.lastProbabilities && (
												<span className="text-[10px] text-zinc-500 ml-1.5 font-normal">
													(J:{(metrics.lastProbabilities.jump * 100).toFixed(0)}% D:{(metrics.lastProbabilities.duck * 100).toFixed(0)}% R:{(metrics.lastProbabilities.run * 100).toFixed(0)}%)
												</span>
											)}
										</span>
									</div>
									<div className="flex justify-between">
										<span className="text-zinc-500">Latency:</span>
										<span className="text-white font-semibold">
											{metrics.lastLatencyMs.toFixed(1)} ms GPU
											{metrics.lastRttMs ? (
												<span className="text-zinc-400 font-normal ml-1">
													({metrics.lastRttMs.toFixed(1)} ms RTT)
												</span>
											) : null}
										</span>
									</div>
									<div className="flex justify-between">
										<span className="text-zinc-500">Compensated Budget:</span>
										<span className="text-purple-400">
											{round === 'fixed_45ms_hold' ? '45.0 ms target' : 'None (immediate)'}
										</span>
									</div>
								</div>

								<div className="pt-2 mt-2 border-t border-zinc-800/60 text-[11px] text-zinc-500 flex justify-between">
									<span>[Space] Jump / [↓/S] Duck</span>
									<span>[M] Swap Model</span>
									<span>[H] Toggle Mode</span>
								</div>
							</div>

							{/* Right: Latency & Jitter Percentiles */}
							<div className="bg-zinc-900/60 p-3 rounded border border-zinc-800/80">
								<div className="flex items-center justify-between pb-2 mb-2 border-b border-zinc-800/60">
									<span className="text-zinc-400 flex items-center gap-1.5 font-bold">
										<Activity className="w-3.5 h-3.5 text-cyan-400" />
										LATENCY JITTER & TIMELINE
									</span>
									<span className="text-red-400 flex items-center gap-1 text-[11px]">
										<ShieldAlert className="w-3 h-3" />
										Misses (&gt;45ms): {metrics.deadlineMisses}
									</span>
								</div>

								{/* Sparkline */}
								<div className="mb-2">
									<Sparkline data={metrics.history} threshold={45} width={360} height={52} />
								</div>

								{/* Percentile Grid */}
								<div className="grid grid-cols-4 gap-2 text-center text-zinc-300">
									<div className="bg-zinc-950 p-1.5 rounded border border-zinc-800">
										<div className="text-[10px] text-zinc-500">p50</div>
										<div className="font-bold text-emerald-400 text-xs">{metrics.p50.toFixed(1)}ms</div>
									</div>
									<div className="bg-zinc-950 p-1.5 rounded border border-zinc-800">
										<div className="text-[10px] text-zinc-500">p90</div>
										<div className="font-bold text-amber-400 text-xs">{metrics.p90.toFixed(1)}ms</div>
									</div>
									<div className="bg-zinc-950 p-1.5 rounded border border-zinc-800">
										<div className="text-[10px] text-zinc-500">p99</div>
										<div className="font-bold text-red-400 text-xs">{metrics.p99.toFixed(1)}ms</div>
									</div>
									<div className="bg-zinc-950 p-1.5 rounded border border-zinc-800">
										<div className="text-[10px] text-zinc-500">Jitter (σ)</div>
										<div className="font-bold text-cyan-400 text-xs">{metrics.jitter.toFixed(1)}ms</div>
									</div>
								</div>
							</div>
						</div>
					</div>

					{/* Right Column: Live Move & Decision History */}
					<div className="bg-zinc-900/50 rounded-lg border border-zinc-800/80 p-3.5 flex flex-col font-mono text-xs min-w-0">
						{/* History Header */}
						<div className="flex items-center justify-between pb-2.5 mb-3 border-b border-zinc-800/80">
							<div className="flex items-center gap-2">
								<History className="w-4 h-4 text-emerald-400" />
								<span className="font-bold text-zinc-200 tracking-wide">MOVE HISTORY</span>
								<span className="px-1.5 py-0.5 rounded bg-zinc-800 text-[10px] text-zinc-400 font-semibold">
									{moveHistory.length}
								</span>
							</div>
							{moveHistory.length > 0 && (
								<button
									onClick={() => setMoveHistory([])}
									title="Clear move history"
									className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-red-400 transition"
								>
									<Trash2 className="w-3 h-3" />
									<span>Clear</span>
								</button>
							)}
						</div>

						{/* Active / Latest Input Card */}
						<div className="mb-3.5 bg-zinc-950/80 p-3 rounded-lg border border-zinc-800">
							<div className="flex items-center justify-between mb-2 pb-1.5 border-b border-zinc-900">
								<span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">
									Active / Latest Move
								</span>
								{moveHistory.length > 0 && (
									<span className="text-[10px] text-zinc-500">{moveHistory[0].timeString}</span>
								)}
							</div>

							{moveHistory.length > 0 ? (
								(() => {
									const latest = moveHistory[0];
									return (
										<div className="space-y-2.5">
											{/* Action Badge & Transition */}
											<div className="flex items-center justify-between">
												<div className="flex items-center gap-2">
													<span
														className={`px-2.5 py-1 rounded text-xs font-bold border tracking-wider flex items-center gap-1 ${latest.action === 'JUMP'
															? 'bg-amber-500/20 text-amber-300 border-amber-500/50'
															: latest.action === 'DUCK'
																? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50'
																: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50'
															}`}
													>
														{latest.action === 'JUMP' ? '▲ JUMP' : latest.action === 'DUCK' ? '▼ DUCK' : '► RUN'}
													</span>
													<span className="text-[11px] text-zinc-400">
														{latest.fromAction} ➔ <span className="text-white font-semibold">{latest.action}</span>
													</span>
												</div>
												<span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400 font-semibold uppercase">
													{latest.driver}
												</span>
											</div>

											{/* Obstacle Context */}
											<div className="text-[11px] text-zinc-400 bg-zinc-900/60 px-2 py-1 rounded flex items-center justify-between">
												<span>Threat: <span className="text-zinc-200 capitalize">{latest.obstacleType} ({latest.obstacleAltitude})</span></span>
												<span>Distance: <span className="text-amber-400 font-semibold">{latest.distance}c</span></span>
											</div>

											{/* Option Confidences (All 3 Presented Choices) */}
											<div className="space-y-1.5 pt-1">
												<div className="text-[10px] text-zinc-500 uppercase font-semibold flex justify-between">
													<span>Option Confidences</span>
													<span>Score</span>
												</div>

												{/* JUMP Option */}
												<div className={`p-1.5 rounded transition ${latest.action === 'JUMP' ? 'bg-amber-950/30 border border-amber-500/30' : 'bg-zinc-900/40'}`}>
													<div className="flex justify-between items-center mb-1">
														<span className={`text-[11px] font-semibold ${latest.action === 'JUMP' ? 'text-amber-300' : 'text-zinc-400'}`}>
															JUMP
														</span>
														<span className={`font-mono text-[11px] font-bold ${latest.action === 'JUMP' ? 'text-amber-400' : 'text-zinc-500'}`}>
															{(latest.probabilities.jump * 100).toFixed(1)}%
														</span>
													</div>
													<div className="w-full bg-zinc-800/80 rounded-full h-1.5 overflow-hidden">
														<div
															className="h-full bg-amber-400 rounded-full transition-all duration-300"
															style={{ width: `${Math.min(100, Math.max(0, latest.probabilities.jump * 100))}%` }}
														/>
													</div>
												</div>

												{/* DUCK Option */}
												<div className={`p-1.5 rounded transition ${latest.action === 'DUCK' ? 'bg-cyan-950/30 border border-cyan-500/30' : 'bg-zinc-900/40'}`}>
													<div className="flex justify-between items-center mb-1">
														<span className={`text-[11px] font-semibold ${latest.action === 'DUCK' ? 'text-cyan-300' : 'text-zinc-400'}`}>
															DUCK
														</span>
														<span className={`font-mono text-[11px] font-bold ${latest.action === 'DUCK' ? 'text-cyan-400' : 'text-zinc-500'}`}>
															{(latest.probabilities.duck * 100).toFixed(1)}%
														</span>
													</div>
													<div className="w-full bg-zinc-800/80 rounded-full h-1.5 overflow-hidden">
														<div
															className="h-full bg-cyan-400 rounded-full transition-all duration-300"
															style={{ width: `${Math.min(100, Math.max(0, latest.probabilities.duck * 100))}%` }}
														/>
													</div>
												</div>

												{/* RUN Option */}
												<div className={`p-1.5 rounded transition ${latest.action === 'RUN' ? 'bg-emerald-950/30 border border-emerald-500/30' : 'bg-zinc-900/40'}`}>
													<div className="flex justify-between items-center mb-1">
														<span className={`text-[11px] font-semibold ${latest.action === 'RUN' ? 'text-emerald-300' : 'text-zinc-400'}`}>
															RUN
														</span>
														<span className={`font-mono text-[11px] font-bold ${latest.action === 'RUN' ? 'text-emerald-400' : 'text-zinc-500'}`}>
															{(latest.probabilities.run * 100).toFixed(1)}%
														</span>
													</div>
													<div className="w-full bg-zinc-800/80 rounded-full h-1.5 overflow-hidden">
														<div
															className="h-full bg-emerald-400 rounded-full transition-all duration-300"
															style={{ width: `${Math.min(100, Math.max(0, latest.probabilities.run * 100))}%` }}
														/>
													</div>
												</div>
											</div>

											{/* Latency Footer */}
											<div className="pt-1.5 border-t border-zinc-900 flex justify-between text-[10px] text-zinc-500">
												<span>GPU: <span className="text-white font-medium">{latest.serverLatencyMs.toFixed(1)}ms</span></span>
												<span>E2E RTT: <span className="text-cyan-400 font-medium">{latest.roundTripMs.toFixed(1)}ms</span></span>
											</div>
										</div>
									);
								})()
							) : (
								<div className="py-6 text-center text-zinc-600 space-y-1">
									<div className="text-xs">Awaiting AI decisions...</div>
									<div className="text-[11px] text-zinc-600">Switch to LAYA or VEGA autopilot to record move history and live option confidences.</div>
								</div>
							)}
						</div>

						{/* Past Moves Feed */}
						<div className="flex-1 flex flex-col min-h-0">
							<div className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold mb-2 flex justify-between">
								<span>Past Moves Log</span>
								<span>{Math.max(0, moveHistory.length - 1)} prior</span>
							</div>

							<div className="flex-1 overflow-y-auto max-h-[340px] space-y-2 pr-1 custom-scrollbar">
								{moveHistory.slice(1).map((m) => (
									<div key={m.id} className="p-2.5 rounded bg-zinc-950/70 border border-zinc-850 hover:border-zinc-700 transition">
										<div className="flex items-center justify-between mb-1">
											<div className="flex items-center gap-1.5">
												<span
													className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${m.action === 'JUMP'
														? 'bg-amber-500/20 text-amber-300'
														: m.action === 'DUCK'
															? 'bg-cyan-500/20 text-cyan-300'
															: 'bg-emerald-500/20 text-emerald-300'
														}`}
												>
													{m.action}
												</span>
												<span className="text-zinc-400 text-[11px]">
													{m.fromAction} ➔ {m.action}
												</span>
											</div>
											<span className="text-[10px] text-zinc-500 font-mono">{m.timeString}</span>
										</div>

										<div className="text-[10px] text-zinc-500 flex justify-between mb-1.5">
											<span className="capitalize">{m.obstacleType} ({m.obstacleAltitude}) @ {m.distance}c</span>
											<span>{m.serverLatencyMs.toFixed(0)}ms GPU • {m.roundTripMs.toFixed(0)}ms RTT</span>
										</div>

										{/* Proportional 3-way Confidence Distribution Bar */}
										<div className="space-y-1">
											<div className="w-full h-1.5 bg-zinc-850 rounded-full flex overflow-hidden">
												<div
													style={{ width: `${Math.min(100, Math.max(0, m.probabilities.jump * 100))}%` }}
													className="bg-amber-400/90 h-full"
													title={`Jump: ${(m.probabilities.jump * 100).toFixed(1)}%`}
												/>
												<div
													style={{ width: `${Math.min(100, Math.max(0, m.probabilities.duck * 100))}%` }}
													className="bg-cyan-400/90 h-full"
													title={`Duck: ${(m.probabilities.duck * 100).toFixed(1)}%`}
												/>
												<div
													style={{ width: `${Math.min(100, Math.max(0, m.probabilities.run * 100))}%` }}
													className="bg-emerald-400/90 h-full"
													title={`Run: ${(m.probabilities.run * 100).toFixed(1)}%`}
												/>
											</div>
											<div className="flex justify-between text-[9px] font-mono text-zinc-500">
												<span className={m.action === 'JUMP' ? 'text-amber-300 font-bold' : ''}>J:{(m.probabilities.jump * 100).toFixed(0)}%</span>
												<span className={m.action === 'DUCK' ? 'text-cyan-300 font-bold' : ''}>D:{(m.probabilities.duck * 100).toFixed(0)}%</span>
												<span className={m.action === 'RUN' ? 'text-emerald-300 font-bold' : ''}>R:{(m.probabilities.run * 100).toFixed(0)}%</span>
											</div>
										</div>
									</div>
								))}
								{moveHistory.length <= 1 && (
									<div className="text-center py-4 text-zinc-600 text-[11px] italic">
										{moveHistory.length === 0 ? 'No moves recorded yet.' : 'Older moves will appear here in chronological order.'}
									</div>
								)}
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
};
