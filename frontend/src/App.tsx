import React, { useEffect, useRef, useState } from 'react';
import { GameEngine } from './game/engine';
import { DriverMode, ExperimentRound, GameState } from './game/types';
import { checkBackendHealth, getBackendUrl, requestPrediction, setBackendUrl } from './telemetry/api';
import { TelemetryMetrics, TelemetryTracker } from './telemetry/stats';
import { Sparkline } from './telemetry/Sparkline';
import { Activity, Cpu, RotateCcw, Settings, ShieldAlert, Sparkles, Terminal } from 'lucide-react';

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

	// In-flight request tracking, scheduled action queue & obstacle jump lockout
	const inFlightRef = useRef<boolean>(false);
	const scheduledTimerRef = useRef<number | null>(null);
	const jumpedObstaclesRef = useRef<Set<number>>(new Set());

	const triggerReset = () => {
		jumpedObstaclesRef.current.clear();
		if (scheduledTimerRef.current) {
			clearTimeout(scheduledTimerRef.current);
			scheduledTimerRef.current = null;
		}
		inFlightRef.current = false;
		engineRef.current?.reset();
	};

	// 1. Initialize GameEngine
	useEffect(() => {
		if (!canvasRef.current) return;
		const engine = new GameEngine(canvasRef.current);
		engineRef.current = engine;

		engine.onTick = (s) => {
			setGameState({ ...s });
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

		// Only query when an obstacle is within trigger horizon (~8.5 cells) and Dino is grounded
		if (obs.distance_to_cactus <= 8.5 && obs.distance_to_cactus > 0) {
			inFlightRef.current = true;
			const dispatchTime = performance.now();
			const currentObsId = obs.obstacle_id;

			try {
				const resp = await requestPrediction(currentDriver, obs);
				const tracker = currentDriver === 'vega' ? trackerVegaRef.current : trackerLayaRef.current;

				tracker.record(resp.inference_ms, resp.confidence, resp.action);
				setMetrics(tracker.getMetrics());

				if (resp.action === 'jump') {
					// Confirm Dino is still grounded and has not already jumped this obstacle
					if (engine.state.isGrounded && currentObsId && !jumpedObstaclesRef.current.has(currentObsId)) {
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
					if (engine.state.isGrounded && currentObsId && !jumpedObstaclesRef.current.has(currentObsId)) {
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
			<div className="w-full max-w-4xl flex items-center justify-between pb-3 border-b border-zinc-800/80 mb-4 font-mono text-xs">
				<div className="flex items-center gap-2">
					<Terminal className="w-4 h-4 text-emerald-400" />
					<span className="font-bold tracking-wider text-emerald-400">LAYADINO // SYSTEM 2.0</span>
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
			<div className="w-full max-w-4xl bg-zinc-950/90 rounded-lg terminal-border p-4 shadow-2xl relative overflow-hidden">
				{/* CRT Scanline Overlay */}
				<div className="absolute inset-0 crt-overlay pointer-events-none" />

				{/* Header HUD Bar */}
				<div className="flex flex-wrap items-center justify-between gap-3 pb-3 mb-3 border-b border-zinc-800 font-mono text-sm">
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
				<div className="flex flex-wrap items-center justify-between gap-3 pt-3 mt-3 border-t border-zinc-800/80 font-mono text-xs">
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
				<div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-3 mt-3 border-t border-zinc-800 font-mono text-xs">
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
								</span>
							</div>
							<div className="flex justify-between">
								<span className="text-zinc-500">Inference Latency:</span>
								<span className="text-white font-semibold">{metrics.lastLatencyMs.toFixed(1)} ms</span>
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
		</div>
	);
};
