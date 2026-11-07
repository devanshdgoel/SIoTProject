/* pet_touch.h - self-capacitance touch on one ESP32 pin, and the gestures on it.
 *
 * No touch IC. The classic ESP32 has a capacitive sensing peripheral on ten of its
 * pins: it charges the pin, counts how long it takes to discharge through a known
 * path, and returns that count. A finger near the electrode adds perhaps a picofarad
 * to ground, the count falls, and that fall is the whole measurement. One wire from
 * GPIO27 to the conductive panel is the entire sensor.
 *
 * ---------------------------------------------------------------------------
 * Why a fixed threshold cannot work here
 *
 * The untouched count is not a constant. It drifts with supply voltage, with the
 * board warming up, with how much water is in the air, and with whatever the pet is
 * standing on. Across a day that drift is comfortably larger than the step a finger
 * makes, so a threshold chosen on the bench is wrong by the evening - it either
 * stops responding or it reports being stroked all night.
 *
 * This node happens to be measuring temperature and relative humidity, which are two
 * of the causes, so the irony is unusually direct.
 *
 * The fix is the standard one for any slowly-drifting baseline: track the untouched
 * level with a long exponential average and threshold the DEVIATION from it rather
 * than the raw value. The baseline is frozen while a touch is in progress, or it
 * would learn the finger and the touch would fade out under a steady hand - a long
 * press being exactly the gesture that must not do that. It is a high-pass filter on
 * the sensor, and the time constants below set where its corner sits: slow enough to
 * ignore a hand resting for a minute, fast enough to follow a warming room.
 *
 * ---------------------------------------------------------------------------
 * Gestures
 *
 *   TAP        a short contact, reported only once it is clear no second one follows
 *   DOUBLE_TAP two taps inside DOUBLE_GAP_MS
 *   PET        contact held past PET_MS, repeating while it is held
 *
 * A tap is deliberately delayed by DOUBLE_GAP_MS before it is reported. The first
 * half of a double tap is indistinguishable from a single one at the moment it
 * happens, so reporting immediately means every double tap also fires a single -
 * and here that would flip to the info screen and straight back out of it.
 */

#ifndef PET_TOUCH_H
#define PET_TOUCH_H

#ifndef TOUCH_PIN
#define TOUCH_PIN T7            // GPIO27: a touch channel, not a strapping pin, and
#endif                          // clear of I2C on 21/22 and the PMS5003 on 25/26

// A finger has to take this fraction off the baseline to count. 0.20 is loose enough
// for a hand on a printed panel through a layer or two of plastic and tight enough
// that the drift the baseline has not yet absorbed does not trip it.
static const float    TOUCH_DROP      = 0.20f;
static const float    RELEASE_DROP    = 0.12f;   // hysteresis: let go lower than grab
static const uint8_t  TOUCH_CONFIRM   = 3;       // consecutive samples, ~45 ms
static const uint32_t TOUCH_SAMPLE_MS = 15;
static const uint32_t TAP_MAX_MS      = 400;     // longer than this is not a tap
static const uint32_t DOUBLE_GAP_MS   = 450;
static const uint32_t PET_MS          = 700;     // held this long is a stroke
static const uint32_t PET_REPEAT_MS   = 900;     // keep reporting while it continues
static const uint32_t BASELINE_TC_MS  = 20000;   // ~20 s to follow the room

enum TouchEvent { TOUCH_NONE, TOUCH_TAP, TOUCH_DOUBLE, TOUCH_PET };

static float    touchBase     = 0;
static bool     touchBaseInit = false;
static bool     touchHeld     = false;
static uint8_t  touchRun      = 0;
static uint32_t touchDownMs   = 0, touchUpMs = 0, lastPetMs = 0, lastSampleMs = 0;
static uint8_t  pendingTaps   = 0;
static uint16_t touchLastRaw  = 0;

static void petTouchBegin() {
  // Seed the baseline from the pin as it is now. Whatever is touching it at boot
  // becomes "untouched", which is the only sane reference available - and the long
  // average corrects it within a minute if that assumption was wrong.
  uint32_t sum = 0;
  for (int i = 0; i < 16; i++) { sum += touchRead(TOUCH_PIN); delay(2); }
  touchBase = sum / 16.0f;
  touchBaseInit = touchBase > 0;
  Serial.printf("[touch] self-capacitance on T%d, baseline %.0f counts. Touch the "
                "pad now - you should see a line each time.\n",
                (int)TOUCH_PIN, touchBase);
  if (!touchBaseInit)
    Serial.println("[touch] baseline read as zero - the pin is probably shorted to ground");
}

/* Sample, track the baseline, and return at most one gesture per call. */
static TouchEvent petTouchPoll(uint32_t now) {
  if (!touchBaseInit) return TOUCH_NONE;
  if (now - lastSampleMs < TOUCH_SAMPLE_MS) {
    // Even with nothing to sample, a tap that has outlived the double-tap window is
    // now definitely single and can be released.
    if (pendingTaps == 1 && !touchHeld && now - touchUpMs > DOUBLE_GAP_MS) {
      pendingTaps = 0;
      return TOUCH_TAP;
    }
    return TOUCH_NONE;
  }
  lastSampleMs = now;

  uint16_t raw = touchRead(TOUCH_PIN);
  touchLastRaw = raw;
  if (raw == 0) return TOUCH_NONE;                 // a failed conversion, not a touch

  float drop = (touchBase - raw) / touchBase;
  bool  near = drop > (touchHeld ? RELEASE_DROP : TOUCH_DROP);

  /* Say something whenever the pad notices ANYTHING.
   *
   * Without this the only two observable states are "a gesture fired" and
   * silence, and silence has far too many causes: a wire that fell off, an
   * electrode too small, a threshold too high for this panel, or - as actually
   * happened - an initialiser that was written and never called. Printing the raw
   * count beside the baseline separates them at a glance. No line at all while
   * you touch it means the electrode is not connected; a line whose drop never
   * reaches the threshold means TOUCH_DROP is too high. Rate-limited so a resting
   * hand cannot flood the log. */
  static uint32_t lastSay = 0;
  if (drop > TOUCH_DROP / 4.0f && now - lastSay > 1000) {
    lastSay = now;
    Serial.printf("[touch] raw %u  baseline %.0f  drop %.0f%%  (need %.0f%%)%s\n",
                  raw, touchBase, drop * 100.0f, TOUCH_DROP * 100.0f,
                  near ? "  <- registering" : "");
  }

  // The baseline only learns while nothing is on the pad, so a hand held still for
  // ten seconds stays a hand rather than becoming the new normal.
  if (!near) {
    float k = 1.0f - expf(-(float)TOUCH_SAMPLE_MS / BASELINE_TC_MS);
    touchBase += (raw - touchBase) * k;
  }

  if (near) {
    if (touchRun < TOUCH_CONFIRM) touchRun++;
    if (touchRun == TOUCH_CONFIRM && !touchHeld) {
      touchHeld = true;
      touchDownMs = now;
      lastPetMs = 0;
    }
  } else {
    if (touchRun) touchRun--;
    if (touchRun == 0 && touchHeld) {
      touchHeld = false;
      uint32_t held = now - touchDownMs;
      touchUpMs = now;
      if (held <= TAP_MAX_MS) {
        if (pendingTaps == 1 && now - touchDownMs < DOUBLE_GAP_MS + TAP_MAX_MS) {
          pendingTaps = 0;
          return TOUCH_DOUBLE;
        }
        pendingTaps = 1;                           // held back until the window closes
      }
    }
  }

  if (touchHeld && now - touchDownMs >= PET_MS && now - lastPetMs >= PET_REPEAT_MS) {
    lastPetMs = now;
    pendingTaps = 0;                               // a stroke is not two taps
    return TOUCH_PET;
  }
  if (pendingTaps == 1 && !touchHeld && now - touchUpMs > DOUBLE_GAP_MS) {
    pendingTaps = 0;
    return TOUCH_TAP;
  }
  return TOUCH_NONE;
}

#endif  // PET_TOUCH_H
