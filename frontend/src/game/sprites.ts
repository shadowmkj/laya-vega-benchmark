/**
 * Retro Pixel Art Rendering Routines for Chrome Dino on HTML5 Canvas.
 * Drawn with crisp vector paths/rectangles for high-DPI scaling.
 */

export function drawDino(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  isGrounded: boolean,
  isGameOver: boolean,
  tick: number,
  color = '#4ade80' // retro phosphor green
) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.translate(x, y);

  const scale = 2.0;
  ctx.scale(scale, scale);

  // Pixel Grid Dimensions for Dino: ~22px wide x 24px high
  // 1. Head & Snout
  ctx.fillRect(11, 0, 10, 8); // Head main
  ctx.fillRect(13, 8, 8, 4);  // Snout lower

  // Eye
  if (isGameOver) {
    ctx.fillStyle = '#000000';
    // X eye
    ctx.fillRect(13, 2, 2, 2);
    ctx.fillRect(15, 2, 2, 2);
    ctx.fillStyle = color;
  } else {
    ctx.fillStyle = '#000000';
    ctx.fillRect(13, 2, 2, 2); // Pupil
    ctx.fillStyle = color;
  }

  // 2. Neck & Body
  ctx.fillRect(9, 6, 4, 8);   // Neck
  ctx.fillRect(5, 10, 10, 8); // Torso

  // 3. Tiny arms
  ctx.fillRect(16, 12, 3, 2);

  // 4. Tail
  ctx.fillRect(3, 11, 2, 4);
  ctx.fillRect(1, 12, 2, 2);

  // 5. Legs
  if (!isGrounded) {
    // Airborne jump tuck
    ctx.fillRect(7, 18, 3, 3);
    ctx.fillRect(11, 18, 3, 3);
  } else {
    // 2-frame running legs
    const runFrame = Math.floor(tick / 6) % 2;
    if (runFrame === 0) {
      // Left leg down, right leg back
      ctx.fillRect(7, 18, 2, 5);
      ctx.fillRect(7, 22, 3, 1); // foot
      ctx.fillRect(11, 18, 3, 3);
    } else {
      // Right leg down, left leg back
      ctx.fillRect(11, 18, 2, 5);
      ctx.fillRect(11, 22, 3, 1); // foot
      ctx.fillRect(7, 18, 3, 3);
    }
  }

  ctx.restore();
}

export function drawCactus(
  ctx: CanvasRenderingContext2D,
  x: number,
  groundY: number,
  type: 'small' | 'large' | 'cluster',
  color = '#ef4444' // Retro alert red/amber
) {
  ctx.save();
  ctx.fillStyle = color;

  if (type === 'small') {
    const h = 36;
    const y = groundY - h;

    // Main stem
    ctx.fillRect(x + 7, y, 6, h);
    // Left arm
    ctx.fillRect(x + 1, y + 10, 6, 4);
    ctx.fillRect(x + 1, y + 6, 4, 8);
    // Right arm
    ctx.fillRect(x + 13, y + 14, 6, 4);
    ctx.fillRect(x + 15, y + 10, 4, 8);
  } else if (type === 'large') {
    const h = 48;
    const y = groundY - h;

    // Main tall stem
    ctx.fillRect(x + 9, y, 8, h);
    // Left arm
    ctx.fillRect(x + 1, y + 14, 8, 5);
    ctx.fillRect(x + 1, y + 8, 5, 12);
    // Right arm
    ctx.fillRect(x + 17, y + 18, 8, 5);
    ctx.fillRect(x + 20, y + 12, 5, 12);
  } else {
    // Cluster: Two cacti side-by-side
    drawCactus(ctx, x, groundY, 'small', color);
    drawCactus(ctx, x + 24, groundY, 'small', color);
  }

  ctx.restore();
}

export function drawGround(
  ctx: CanvasRenderingContext2D,
  width: number,
  groundY: number,
  tick: number,
  color = '#334155'
) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.fillRect(0, groundY, width, 2);

  // Scrolling terrain debris
  const debrisOffset = (tick * 4) % 24;
  for (let x = -debrisOffset; x < width; x += 24) {
    if (x > 0) {
      ctx.fillRect(x, groundY + 5, 3, 1);
      ctx.fillRect(x + 10, groundY + 8, 5, 1);
      ctx.fillRect(x + 18, groundY + 4, 2, 1);
    }
  }
  ctx.restore();
}
