/* pet_face.h - a face that is the whole screen, and the touch that talks to it.
 *
 * Included from esp32_scd40.ino inside its HAS_OLED block, so `u8g2` is already
 * declared when this header is read. It is a header rather than a .cpp because the
 * Arduino build is a single translation unit and splitting it would buy nothing but
 * a linker error about the display object.
 *
 * ---------------------------------------------------------------------------
 * Why the face is parameters rather than pictures
 *
 * The obvious way to draw six expressions is six drawing routines and a switch.
 * That cannot animate: there is no halfway between the "happy" routine and the
 * "concerned" one, so every change of mood is a jump cut.
 *
 * So an expression here is not a picture, it is a POINT IN A SMALL CONTINUOUS
 * SPACE - how open the eyes are, how the brows slant, how the mouth curves, how far
 * it hangs open. Drawing is one routine that reads those numbers. A change of mood
 * sets a new target and the current point eases towards it over a few hundred
 * milliseconds, so every intermediate frame is a real face nobody had to draw, and
 * any new expression is six numbers rather than another routine.
 *
 * It also makes the involuntary movement free. A blink is the eye-open parameter
 * driven briefly to zero and released; it composes with whatever expression is
 * showing, instead of needing a blinking variant of each one.
 *
 * ---------------------------------------------------------------------------
 * Why it is allowed to look alive
 *
 * A face that holds perfectly still reads as a diagram. Two things fix that, both
 * cheap: the eyes blink on a slightly irregular schedule, and they drift to a new
 * point every few seconds the way a real gaze saccades rather than sitting dead
 * centre. Neither carries information - they exist so that the moments the face IS
 * carrying information are legible as a change.
 */

#ifndef PET_FACE_H
#define PET_FACE_H

// ---------------------------------------------------------------- geometry
// Laid out for a 128x64 panel. The eyes sit high and wide and the mouth low, which
// is what makes it read as a face at arm's length rather than as three shapes.
static const int FACE_W      = 128;
static const int EYE_CX_L    = 37;
static const int EYE_CX_R    = 91;
static const int EYE_CY      = 25;
static const int EYE_HALF_W  = 17;
static const int EYE_HALF_H  = 15;
static const int MOUTH_CY    = 52;
static const int MOUTH_HALF  = 26;

// ---------------------------------------------------------------- the space
struct FaceParams {
  float eyeOpen;      // 0 shut .. 1 wide. Blinking drives this and nothing else.
  float eyeSquint;    // 0 round .. 1 lower lid pushed up, which is what a real smile does
  float browAngle;    // -1 outer ends down (sad) .. +1 inner ends down (cross)
  float browLift;     // 0 resting .. 1 raised, for surprise
  float mouthCurve;   // -1 frown .. 0 flat .. +1 smile
  float mouthOpen;    // 0 a line .. 1 wide open
  float gazeBiasY;    // -1 looking up .. +1 looking down, as part of the mood
  float pupil;        // pupil radius in pixels; 0 draws a solid eye
};

//                        open squint brow  lift  curve  mOpen gazeY pupil
static const FaceParams F_NEUTRAL   = {1.00f, 0.00f,  0.0f, 0.0f,  0.05f, 0.00f,  0.0f, 5.0f};
static const FaceParams F_HAPPY     = {0.85f, 0.65f,  0.0f, 0.1f,  1.00f, 0.15f, -0.1f, 5.0f};
static const FaceParams F_CONCERNED = {1.00f, 0.00f, -0.7f, 0.3f, -0.45f, 0.00f,  0.2f, 4.5f};
static const FaceParams F_SAD       = {0.80f, 0.00f, -1.0f, 0.1f, -1.00f, 0.00f,  0.5f, 4.0f};
static const FaceParams F_ALERT     = {1.00f, 0.00f,  0.2f, 1.0f, -0.10f, 0.70f, -0.3f, 3.5f};
static const FaceParams F_SLEEPY    = {0.18f, 0.30f, -0.3f, 0.0f, -0.10f, 0.00f,  0.4f, 4.0f};
// Petting. Eyes nearly shut the way a dog's go when it is enjoying something, the
// grin as wide as the mouth allows, and the gaze tipped up at whoever is doing it.
static const FaceParams F_EXCITED   = {0.45f, 0.95f,  0.3f, 0.6f,  1.00f, 0.55f, -1.0f, 5.5f};

static FaceParams faceNow   = F_NEUTRAL;    // where the face is this frame
static FaceParams faceGoal  = F_NEUTRAL;    // where it is heading

// ---------------------------------------------------------------- involuntary
static float    gazeX = 0, gazeY = 0, gazeTX = 0, gazeTY = 0;
static uint32_t nextSaccadeMs = 0, nextBlinkMs = 0, blinkStartMs = 0;
static const uint32_t BLINK_MS = 150;       // a human blink, near enough

static float ease(float from, float to, float k) { return from + (to - from) * k; }

static void faceEase(FaceParams &a, const FaceParams &b, float k) {
  a.eyeOpen    = ease(a.eyeOpen,    b.eyeOpen,    k);
  a.eyeSquint  = ease(a.eyeSquint,  b.eyeSquint,  k);
  a.browAngle  = ease(a.browAngle,  b.browAngle,  k);
  a.browLift   = ease(a.browLift,   b.browLift,   k);
  a.mouthCurve = ease(a.mouthCurve, b.mouthCurve, k);
  a.mouthOpen  = ease(a.mouthOpen,  b.mouthOpen,  k);
  a.gazeBiasY  = ease(a.gazeBiasY,  b.gazeBiasY,  k);
  a.pupil      = ease(a.pupil,      b.pupil,      k);
}

static const FaceParams &faceFor(const char *expr) {
  if (!expr || !expr[0])                return F_NEUTRAL;
  if (!strcmp(expr, "happy"))           return F_HAPPY;
  if (!strcmp(expr, "concerned"))       return F_CONCERNED;
  if (!strcmp(expr, "sad"))             return F_SAD;
  if (!strcmp(expr, "alert"))           return F_ALERT;
  if (!strcmp(expr, "sleepy"))          return F_SLEEPY;
  if (!strcmp(expr, "excited"))         return F_EXCITED;
  return F_NEUTRAL;
}

/* Advance blink, gaze and the ease towards the target. Call once per frame.
 *
 * `dt` is passed in rather than measured here because the loop stalls for seconds
 * at a time during an HTTP post, and an animation that assumes a fixed frame time
 * jumps when it resumes. Easing by elapsed time instead of by frame makes the face
 * arrive at its target at the same wall-clock moment whatever the loop was doing.
 */
static void faceTick(uint32_t now, uint32_t dt) {
  // Expression: a 300 ms time constant, converted to a per-frame factor so that a
  // long frame catches up in one step instead of lagging.
  float k = 1.0f - expf(-(float)dt / 300.0f);
  faceEase(faceNow, faceGoal, k);

  // Gaze: a new resting point every 1.5-4 s, reached smoothly. Irregular on purpose
  // - a perfectly periodic glance looks mechanical, which is the thing being avoided.
  if (now >= nextSaccadeMs) {
    nextSaccadeMs = now + 1500 + (uint32_t)random(2500);
    gazeTX = (random(201) - 100) / 100.0f * 0.75f;
    gazeTY = (random(201) - 100) / 100.0f * 0.45f;
  }
  float gk = 1.0f - expf(-(float)dt / 120.0f);      // the eye itself moves fast
  gazeX = ease(gazeX, gazeTX, gk);
  gazeY = ease(gazeY, gazeTY, gk);

  if (now >= nextBlinkMs) {
    blinkStartMs = now;
    nextBlinkMs  = now + 2500 + (uint32_t)random(4000);
  }
}

/* How open the eyes are this instant: the expression, interrupted by any blink.
 * Down and up over BLINK_MS, which composes with every expression for free. */
static float blinkFactor(uint32_t now) {
  uint32_t age = now - blinkStartMs;
  if (age >= BLINK_MS) return 1.0f;
  float half = BLINK_MS / 2.0f;
  float f = (age < half) ? 1.0f - age / half : (age - half) / half;
  return f < 0.05f ? 0.05f : f;
}

// ---------------------------------------------------------------- drawing
static void drawEye(int cx, float openF) {
  int h = (int)(EYE_HALF_H * faceNow.eyeOpen * openF);
  int w = EYE_HALF_W;
  int cy = EYE_CY + (int)(faceNow.gazeBiasY * 4.0f);

  if (h < 2) {                                   // shut: one bar, which is a blink
    u8g2.drawBox(cx - w, cy - 1, 2 * w, 2);
    return;
  }
  u8g2.drawRBox(cx - w, cy - h, 2 * w, 2 * h, h < 6 ? h / 2 : 6);

  // The lower lid of a real smile pushes up into the eye, which is the difference
  // between a face that is smiling and a face with a curve under it. Drawn by
  // cutting the bottom of the eye away in the background colour.
  int squint = (int)(faceNow.eyeSquint * h * 0.8f);
  if (squint > 0) {
    u8g2.setDrawColor(0);
    u8g2.drawBox(cx - w - 1, cy + h - squint, 2 * w + 2, squint + 2);
    u8g2.setDrawColor(1);
  }

  // Pupil, as a hole in the eye, following the gaze. Clamped inside the lid so it
  // never escapes the eye when the two animations disagree.
  int pr = (int)faceNow.pupil;
  int inner = h - squint;
  if (pr > 0 && inner > pr + 1) {
    int px = cx + (int)(gazeX * (w - pr - 3));
    int py = cy + (int)(gazeY * (inner - pr - 2));
    if (py > cy + inner - pr - 1) py = cy + inner - pr - 1;
    if (py < cy - h + pr + 1)     py = cy - h + pr + 1;
    u8g2.setDrawColor(0);
    u8g2.drawDisc(px, py, pr);
    u8g2.setDrawColor(1);
  }
}

static void drawBrow(int cx, int dir) {
  if (fabsf(faceNow.browAngle) < 0.08f && faceNow.browLift < 0.08f) return;
  int cy = EYE_CY - (int)(EYE_HALF_H * faceNow.eyeOpen) - 4 - (int)(faceNow.browLift * 5.0f);
  int w  = EYE_HALF_W - 2;
  int tilt = (int)(faceNow.browAngle * 7.0f) * dir;    // dir flips it for the right brow
  u8g2.drawLine(cx - w, cy - tilt, cx + w, cy + tilt);
  u8g2.drawLine(cx - w, cy - tilt + 1, cx + w, cy + tilt + 1);
}

/* The mouth as a parabola, which is what makes a frown become a smile smoothly.
 *
 * y = curve * (1 - (x/half)^2) is one expression covering the whole range: positive
 * curve bows it down at the centre into a smile, negative bows it up into a frown,
 * and zero is a straight line passed through on the way between them. Drawing it
 * per-column rather than with an arc primitive is what allows the intermediate
 * shapes to exist at all - u8g2 can draw a circular arc, but not 40% of one.
 */
static void drawMouth() {
  const int half = MOUTH_HALF;
  float curve = faceNow.mouthCurve * 11.0f;
  int   gap   = (int)(faceNow.mouthOpen * 11.0f);

  for (int x = -half; x <= half; x++) {
    float n = (float)x / half;
    int y = MOUTH_CY + (int)(curve * (1.0f - n * n));
    if (gap <= 1) {
      u8g2.drawPixel(FACE_W / 2 + x, y);
      u8g2.drawPixel(FACE_W / 2 + x, y + 1);       // 2 px so it reads at a glance
    } else {
      // Open: the same curve is the upper lip and the lower one bulges below it,
      // narrowing to meet at the corners so the mouth stays closed at its ends.
      int bulge = (int)(gap * (1.0f - n * n));
      u8g2.drawVLine(FACE_W / 2 + x, y, bulge + 1);
    }
  }
}

static void drawPetFace(uint32_t now) {
  float bf = blinkFactor(now);
  drawEye(EYE_CX_L, bf);
  drawEye(EYE_CX_R, bf);
  drawBrow(EYE_CX_L, +1);
  drawBrow(EYE_CX_R, -1);
  drawMouth();
}

#endif  // PET_FACE_H
