import { DinoObservation, GameState } from './types';
import { drawCactus, drawDino, drawGround } from './sprites';

export class GameEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private animId: number | null = null;
  private lastTime = 0;

  // Coordinate dimensions
  public readonly width = 800;
  public readonly height = 240;
  public readonly groundY = 200;
  public readonly dinoX = 50;

  // Physics constants
  private readonly JUMP_VELOCITY = -12.5;
  private readonly GRAVITY = 0.65;
  private readonly BASE_SPEED = 6.0;

  public state: GameState = {
    dinoY: 0,
    dinoVy: 0,
    isGrounded: true,
    score: 0,
    highScore: 0,
    speed: this.BASE_SPEED,
    gameOver: false,
    obstacles: [],
    ticks: 0,
  };

  private nextSpawnDistance = 300;
  private nextObstacleId = 1;
  public onTick?: (state: GameState) => void;
  public onGameOver?: (score: number) => void;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D canvas context');
    this.ctx = ctx;

    this.setupCanvasDPI();
  }

  private setupCanvasDPI() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.ctx.imageSmoothingEnabled = false; // keep crisp pixel art
  }

  public start() {
    if (this.animId !== null) return;
    this.lastTime = performance.now();
    this.loop(this.lastTime);
  }

  public stop() {
    if (this.animId !== null) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
  }

  public reset() {
    this.state.dinoY = 0;
    this.state.dinoVy = 0;
    this.state.isGrounded = true;
    this.state.score = 0;
    this.state.speed = this.BASE_SPEED;
    this.state.gameOver = false;
    this.state.obstacles = [];
    this.state.ticks = 0;
    this.nextSpawnDistance = 350;
    this.nextObstacleId = 1;
  }

  public jump(): boolean {
    if (this.state.gameOver) {
      this.reset();
      return true;
    }
    if (this.state.isGrounded) {
      this.state.dinoVy = this.JUMP_VELOCITY;
      this.state.isGrounded = false;
      return true;
    }
    return false;
  }

  public getObservation(latencyBudgetMs = 0): DinoObservation {
    const dinoFrontX = this.dinoX + 36; // Dino front edge
    const closest = this.state.obstacles
      .filter((obs) => obs.x + obs.width > this.dinoX)
      .sort((a, b) => a.x - b.x)[0];

    let dist = closest ? Math.max(0, closest.x - dinoFrontX) : 999;
    const width = closest ? closest.width : 0;
    const height = closest ? closest.height : 0;

    if (latencyBudgetMs > 0 && dist < 900) {
      // Speed in px per frame (at 60 FPS -> speed * 60 px/sec = speed * 0.06 px/ms)
      const pxPerMs = this.state.speed * 0.06;
      const projectedAdvance = pxPerMs * latencyBudgetMs;
      dist = Math.max(0, dist - projectedAdvance);
    }

    // Convert pixel distance to normalized cells (1 cell ~ 16px)
    const normalizedDist = Math.round((dist / 16.0) * 10) / 10;

    return {
      obstacle_id: closest ? closest.id : 0,
      distance_to_cactus: normalizedDist,
      cactus_width: Math.round(width / 16.0),
      cactus_height: Math.round(height / 16.0),
      dino_is_grounded: this.state.isGrounded,
      speed: Math.round((this.state.speed / this.BASE_SPEED) * 100) / 100,
      latency_budget_ms: latencyBudgetMs,
    };
  }

  private loop = (currentTime: number) => {
    const dt = Math.min((currentTime - this.lastTime) / 16.667, 2.0); // clamped normalized frame delta
    this.lastTime = currentTime;

    if (!this.state.gameOver) {
      this.updatePhysics(dt);
    }

    this.render();

    if (this.onTick) {
      this.onTick(this.state);
    }

    this.animId = requestAnimationFrame(this.loop);
  };

  private updatePhysics(dt: number) {
    this.state.ticks += 1;
    this.state.score += 1;
    if (this.state.score > this.state.highScore) {
      this.state.highScore = this.state.score;
    }

    // Gentle speed scaling
    this.state.speed = this.BASE_SPEED + Math.min(this.state.score / 1500, 4.0);

    // Jump dynamics
    if (!this.state.isGrounded) {
      this.state.dinoY += this.state.dinoVy * dt;
      this.state.dinoVy += this.GRAVITY * dt;

      if (this.state.dinoY >= 0) {
        this.state.dinoY = 0;
        this.state.dinoVy = 0;
        this.state.isGrounded = true;
      }
    }

    // Move obstacles
    const moveDist = this.state.speed * dt;
    for (const obs of this.state.obstacles) {
      obs.x -= moveDist;
    }

    // Remove off-screen obstacles
    this.state.obstacles = this.state.obstacles.filter((obs) => obs.x + obs.width > -10);

    // Spawning logic with generous spacing
    this.nextSpawnDistance -= moveDist;
    if (this.nextSpawnDistance <= 0) {
      const types: ('small' | 'large' | 'cluster')[] = ['small', 'large', 'cluster'];
      const chosen = types[Math.floor(Math.random() * types.length)];
      const width = chosen === 'small' ? 22 : chosen === 'large' ? 28 : 50;
      const height = chosen === 'large' ? 48 : 36;

      this.state.obstacles.push({
        id: this.nextObstacleId++,
        x: this.width + 20,
        width,
        height,
        type: chosen,
      });

      // Generous spacing: 450px to 700px gap
      this.nextSpawnDistance = 450 + Math.random() * 250 + this.state.speed * 20;
    }

    // Collision check
    if (this.checkCollision()) {
      this.state.gameOver = true;
      if (this.onGameOver) {
        this.onGameOver(this.state.score);
      }
    }
  }

  private checkCollision(): boolean {
    const dinoHitbox = {
      left: this.dinoX + 8,
      right: this.dinoX + 34,
      bottom: this.groundY + this.state.dinoY,
      top: this.groundY + this.state.dinoY - 42,
    };

    for (const obs of this.state.obstacles) {
      const obsHitbox = {
        left: obs.x + 4,
        right: obs.x + obs.width - 4,
        top: this.groundY - obs.height + 4, // 4px forgiving top clearance
        bottom: this.groundY,
      };

      const horizontalOverlap =
        dinoHitbox.right > obsHitbox.left && dinoHitbox.left < obsHitbox.right;
      const verticalOverlap = dinoHitbox.bottom > obsHitbox.top;

      if (horizontalOverlap && verticalOverlap) {
        return true;
      }
    }

    return false;
  }

  private render() {
    this.ctx.clearRect(0, 0, this.width, this.height);

    // 1. Scrolling Ground
    drawGround(this.ctx, this.width, this.groundY, this.state.ticks);

    // 2. Obstacles
    for (const obs of this.state.obstacles) {
      drawCactus(this.ctx, obs.x, this.groundY, obs.type);
    }

    // 3. Dino
    const dinoScreenY = this.groundY - 48 + this.state.dinoY;
    drawDino(
      this.ctx,
      this.dinoX,
      dinoScreenY,
      this.state.isGrounded,
      this.state.gameOver,
      this.state.ticks
    );

    // 4. Game Over Overlay
    if (this.state.gameOver) {
      this.ctx.save();
      this.ctx.fillStyle = '#ef4444';
      this.ctx.font = '800 22px "JetBrains Mono", monospace';
      this.ctx.textAlign = 'center';
      this.ctx.fillText('G A M E   O V E R', this.width / 2, this.height / 2 - 10);

      this.ctx.fillStyle = '#94a3b8';
      this.ctx.font = '600 13px "JetBrains Mono", monospace';
      this.ctx.fillText('PRESS [SPACE] OR TAP TO RESTART', this.width / 2, this.height / 2 + 18);
      this.ctx.restore();
    }
  }
}
