/*
 * esp32_scd40.ino  -  indoor sensor node (+ optional OLED pet display)
 *
 * Reads a Sensirion SCD40 (CO2, temperature, RH) every 5 s, averages each
 * minute, timestamps it with NTP time, buffers it in RAM and POSTs batches to
 * the host's /ingest endpoint (collectors/scd40.py, backend "http").
 * If WiFi or the host is down the node keeps sampling (up to ~48 h buffered) and
 * uploads the backlog when the link returns; the host de-duplicates by timestamp.
 *
 * The /ingest reply carries the host's decisions back, so the node needs no second
 * connection: ambient pressure for sensor compensation, and the pet's expression
 * and message for the OLED. Until the host says otherwise the face is derived from
 * local CO2 only (LOCAL_FACE_FALLBACK), so the display is never blank.
 *
 * Wiring (3.3 V only), one shared I2C bus:
 *   SCD40  VDD->3V3, GND->GND, SDA->GPIO21, SCL->GPIO22   (address 0x62)
 *   OLED   VCC->3V3, GND->GND, SDA->GPIO21, SCL->GPIO22   (address 0x3C)
 * Mount the sensor on a short lead away from the ESP32 module: WiFi transmit
 * heat raises the reading; tune TEMP_OFFSET_C against a reference thermometer.
 *
 * Libraries: none for the sensor (the SCD4x I2C protocol is implemented directly,
 * so the sketch does not depend on a library version). The display needs "U8g2"
 * by oliver, installed from Arduino IDE > Tools > Manage Libraries.
 * Board: "ESP32 Dev Module".
 *
 * Serial monitor (115200 baud) narrates every stage with a [tag]:
 *   [boot] [i2c] [scd40] [oled] [wifi] [time] [minute] [upload]
 */
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <time.h>
#include "secrets.h"

// Per-node identity, set by the NODE switch in secrets.h: one sketch serves every
// board. ROOM_NAME becomes the data location on the host, so each sensor keeps its
// own series (lower-case letters, digits, - and _ only). Defaults below apply only
// if secrets.h does not define them.
#ifndef DEVICE_NAME
#define DEVICE_NAME "esp32-1"
#endif
#ifndef ROOM_NAME
#define ROOM_NAME "desk"
#endif
#ifndef HAS_PMS
#define HAS_PMS 0         // 1 on the node with a PMS5003 particle counter wired
#endif
#ifndef HAS_OLED
#define HAS_OLED 0        // 1 only on the node that has a display wired
#endif
// Capacitive touch on one pin, for the conductive panel on the pet's back. Needs no
// sensor of its own: see pet_touch.h for how the ESP32 does this with a bare wire.
#ifndef HAS_TOUCH
#define HAS_TOUCH 0
#endif

// Display panel. Defaults suit the common 0.96" 128x64 SSD1306 module; override in
// secrets.h if the controller is an SH1106 (0.96"/1.3" clones) or the panel is 32 px tall.
#ifndef OLED_SH1106
#define OLED_SH1106 0
#endif
#ifndef OLED_H
#define OLED_H 64
#endif
#ifndef LOCAL_FACE_FALLBACK
#define LOCAL_FACE_FALLBACK 1   // pick a face from local CO2 until the host sends one
#endif

static const uint8_t  SCD_ADDR       = 0x62;
static const float    TEMP_OFFSET_C  = 4.0;      // self-heating compensation
#if HAS_PMS
// Each stored minute carries three mass figures and six count bins on top of the
// CO2 fields, so it is roughly twice the size. Halving the depth keeps the static
// allocation where it was: a day of backlog still outlasts any outage seen so far,
// and running the board out of DRAM would cost every reading, not just the old ones.
static const int      BUF_SIZE       = 1440;     // minutes of backlog (~24 h)
#else
static const int      BUF_SIZE       = 720;     // minutes of backlog (~12 h)
#endif
static const int      BATCH_MAX      = 30;       // readings per POST
static const int      MAX_BATCHES_PER_CALL = 8;  // keep sampling while a backlog drains
static const uint32_t UPLOAD_EVERY_MS = 60000;
static const uint32_t CATCHUP_MS      = 5000;    // retry sooner while still behind

// Particle data rides along on the minutes when the counter actually ran. Masses
// are the "atmospheric" figures in ug/m3; counts are particles per 0.1 L above each
// size, which is the part of this sensor most people throw away and the part that
// says WHAT the particles are rather than just how many.
struct Particles {
  uint16_t pm1, pm25, pm10;                       // ug/m3
  uint16_t n03, n05, n10, n25, n50, n100;         // per 0.1 L, cumulative
};

struct Reading {
  uint32_t ts; float co2, co2_max, t, rh; uint8_t n;
  uint8_t  hasPm;                                 // 0 on the minutes the fan was off
  Particles pm;
};
static Reading buf[BUF_SIZE];
static int bufHead = 0, bufCount = 0;            // ring buffer: oldest at bufHead

// Shared node state, also what the display shows.
static float lastCo2 = 0, lastT = 0, lastRh = 0;
static bool  haveReading = false, timeOk = false, uploadOk = false, uploadTried = false;
static uint32_t lastUpload = 0;                  // used by upload() to pace catching up
static uint8_t rejectStreak = 0;                 // consecutive batches the host refused
static char  dispExpr[16] = "";                  // from the host; empty until it replies
static char  dispMsg[64]  = "";
// Lines for the info screen, filled by the host. The node renders whatever it is
// given and knows nothing about air quality: deciding what is worth saying needs the
// forecast, the weather and the room model, none of which live on a microcontroller.
#define INFO_LINES 4
#define INFO_CHARS 24
static char  dispInfo[INFO_LINES][INFO_CHARS] = {{0}};
static uint8_t dispInfoN = 0;

// ---------------------------------------------------------------- PMS5003
#if HAS_PMS
/* Plantower PMS5003 on UART2.
 *
 * The fan is the part that wears out - about 8000 hours, which is under a year if
 * it never stops - and it is also the only moving part in the project. So it is not
 * left running. The sensor sleeps between samples and wakes for a warm-up before
 * each one, which at one sample every five minutes puts the duty cycle near a tenth
 * and the fan's life near a decade. Plantower specify 30 s to a stable reading from
 * cold, and that number is why the warm-up is as long as it is.
 *
 * Frames are 32 bytes: 0x42 0x4D, a length, thirteen big-endian 16-bit fields and a
 * checksum over everything before it. A burst is read rather than a single frame and
 * the median taken, because the occasional frame arrives mangled and a median does
 * not care, where a mean would carry the damage into the record.
 */
static const int      PMS_RX_PIN     = 25;   // ESP32 receives here <- sensor TXD
static const int      PMS_TX_PIN     = 26;   // ESP32 sends here   -> sensor RXD
static const uint32_t PM_EVERY_MS    = 300000;  // one sample every 5 minutes
static const uint32_t PM_WARMUP_MS   = 30000;   // Plantower's figure for a stable fan
static const uint32_t PM_BURST_MS    = 8000;    // listen this long, then sleep
static const int      PM_MAX_FRAMES  = 16;

static Particles pmLatest;
static bool      pmHave = false, pmEverRead = false;
static uint32_t  pmLastSample = 0, pmPhaseAt = 0;
static uint32_t  pmFramesOk = 0, pmFramesBad = 0;
enum PmPhase { PM_ASLEEP, PM_WARMING, PM_LISTENING };
static PmPhase   pmPhase = PM_ASLEEP;
static uint16_t  pmBurst[PM_MAX_FRAMES][9];
static int       pmFrames = 0;

static void pmsCommand(uint8_t cmd, uint8_t datah, uint8_t datal) {
  uint8_t f[7] = {0x42, 0x4D, cmd, datah, datal, 0, 0};
  uint16_t sum = 0;
  for (int i = 0; i < 5; i++) sum += f[i];
  f[5] = sum >> 8; f[6] = sum & 0xFF;
  Serial2.write(f, 7);
}
static void pmsSleep() { pmsCommand(0xE4, 0x00, 0x00); }
static void pmsWake()  { pmsCommand(0xE4, 0x00, 0x01); }

/* Read one complete frame if the bytes are there. Returns false when they are not,
   so the caller can keep looping without blocking. */
static bool pmsFrame(uint16_t out[9]) {
  while (Serial2.available() >= 32) {
    if (Serial2.peek() != 0x42) { Serial2.read(); continue; }   // resynchronise
    uint8_t b[32];
    if (Serial2.readBytes(b, 32) != 32) return false;
    if (b[1] != 0x4D) { pmFramesBad++; continue; }
    uint16_t sum = 0;
    for (int i = 0; i < 30; i++) sum += b[i];
    uint16_t given = (uint16_t)b[30] << 8 | b[31];
    if (sum != given) { pmFramesBad++; continue; }
    // Fields 4..6 are the atmospheric masses, 7..12 the cumulative counts. The
    // first three are the CF=1 "standard" masses, which agree with the atmospheric
    // ones below 100 ug/m3 and are left on the floor.
    for (int i = 0; i < 3; i++) out[i] = (uint16_t)b[10 + 2 * i] << 8 | b[11 + 2 * i];
    for (int i = 0; i < 6; i++) out[3 + i] = (uint16_t)b[16 + 2 * i] << 8 | b[17 + 2 * i];
    pmFramesOk++;
    return true;
  }
  return false;
}

static uint16_t medianOf(int field) {
  uint16_t v[PM_MAX_FRAMES];
  for (int i = 0; i < pmFrames; i++) v[i] = pmBurst[i][field];
  for (int i = 1; i < pmFrames; i++) {            // insertion sort, at most 16 items
    uint16_t key = v[i]; int j = i - 1;
    while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; j--; }
    v[j + 1] = key;
  }
  return pmFrames ? v[pmFrames / 2] : 0;
}

static void pmsBegin() {
  Serial2.begin(9600, SERIAL_8N1, PMS_RX_PIN, PMS_TX_PIN);
  delay(100);
  pmsWake();                                      // in case it was left asleep
  delay(50);
  pmsSleep();
  pmPhase = PM_ASLEEP;
  pmPhaseAt = pmLastSample = millis();
  Serial.printf("[pms] PMS5003 on UART2, RX=GPIO%d TX=GPIO%d, 9600 baud\n",
                PMS_RX_PIN, PMS_TX_PIN);
  Serial.printf("[pms] sampling every %lu s with a %lu s warm-up (fan duty ~%d%%)\n",
                (unsigned long)(PM_EVERY_MS / 1000), (unsigned long)(PM_WARMUP_MS / 1000),
                (int)(100 * (PM_WARMUP_MS + PM_BURST_MS) / PM_EVERY_MS));
  Serial.println("[pms] no reading yet - the first one lands after the first warm-up");
}

/* Non-blocking: called every loop, moves the sensor through sleep -> warm -> listen. */
static void pmsPoll() {
  uint32_t now = millis();
  switch (pmPhase) {
    case PM_ASLEEP:
      if (now - pmLastSample >= PM_EVERY_MS) {
        pmsWake();
        while (Serial2.available()) Serial2.read();     // drop anything stale
        pmFrames = 0;
        pmPhase = PM_WARMING; pmPhaseAt = now;
      }
      break;
    case PM_WARMING:
      while (Serial2.available()) Serial2.read();       // fan still spinning up
      if (now - pmPhaseAt >= PM_WARMUP_MS) { pmPhase = PM_LISTENING; pmPhaseAt = now; }
      break;
    case PM_LISTENING: {
      uint16_t f[9];
      while (pmFrames < PM_MAX_FRAMES && pmsFrame(f)) {
        memcpy(pmBurst[pmFrames], f, sizeof f);
        pmFrames++;
      }
      bool done = pmFrames >= PM_MAX_FRAMES || now - pmPhaseAt >= PM_BURST_MS;
      if (!done) break;
      if (pmFrames >= 3) {
        pmLatest.pm1  = medianOf(0); pmLatest.pm25 = medianOf(1); pmLatest.pm10 = medianOf(2);
        pmLatest.n03  = medianOf(3); pmLatest.n05  = medianOf(4); pmLatest.n10  = medianOf(5);
        pmLatest.n25  = medianOf(6); pmLatest.n50  = medianOf(7); pmLatest.n100 = medianOf(8);
        pmHave = true; pmEverRead = true;
        Serial.printf("[pms] PM1 %u  PM2.5 %u  PM10 %u ug/m3   (median of %d frames)\n",
                      pmLatest.pm1, pmLatest.pm25, pmLatest.pm10, pmFrames);
        Serial.printf("[pms]   per 0.1 L: >0.3um %u  >0.5 %u  >1.0 %u  >2.5 %u  >5 %u  >10 %u\n",
                      pmLatest.n03, pmLatest.n05, pmLatest.n10,
                      pmLatest.n25, pmLatest.n50, pmLatest.n100);
      } else {
        Serial.printf("[pms] only %d good frame(s) in %lu s - check TXD is on GPIO%d "
                      "and that VCC is 5V, not 3V3 (the fan needs it)\n",
                      pmFrames, (unsigned long)(PM_BURST_MS / 1000), PMS_RX_PIN);
      }
      pmsSleep();
      pmPhase = PM_ASLEEP; pmLastSample = now;
      break;
    }
  }
}
#endif  // HAS_PMS

// ---------------------------------------------------------------- SCD4x I2C
static uint8_t crc8(const uint8_t *d, int n) {
  uint8_t c = 0xFF;
  for (int i = 0; i < n; i++) { c ^= d[i]; for (int b = 0; b < 8; b++) c = (c & 0x80) ? (c << 1) ^ 0x31 : (c << 1); }
  return c;
}
static bool scdCmd(uint16_t cmd) {
  Wire.beginTransmission(SCD_ADDR); Wire.write(cmd >> 8); Wire.write(cmd & 0xFF);
  return Wire.endTransmission() == 0;
}
static bool scdCmdArg(uint16_t cmd, uint16_t arg) {
  uint8_t a[2] = { (uint8_t)(arg >> 8), (uint8_t)(arg & 0xFF) };
  Wire.beginTransmission(SCD_ADDR); Wire.write(cmd >> 8); Wire.write(cmd & 0xFF);
  Wire.write(a[0]); Wire.write(a[1]); Wire.write(crc8(a, 2));
  return Wire.endTransmission() == 0;
}
static bool scdRead(uint16_t cmd, uint16_t *out, int words, int waitMs) {
  if (!scdCmd(cmd)) return false;
  delay(waitMs);
  if (Wire.requestFrom((int)SCD_ADDR, words * 3) != words * 3) return false;
  for (int i = 0; i < words; i++) {
    uint8_t b[3] = { (uint8_t)Wire.read(), (uint8_t)Wire.read(), (uint8_t)Wire.read() };
    if (crc8(b, 2) != b[2]) return false;
    out[i] = (b[0] << 8) | b[1];
  }
  return true;
}
static bool scdInit() {
  scdCmd(0x3F86); delay(500);                                   // stop periodic measurement
  scdCmdArg(0x2416, 0); delay(1);                               // automatic self-calibration OFF (occupied room)
  scdCmdArg(0x241D, (uint16_t)(TEMP_OFFSET_C * 65535.0 / 175.0)); delay(1);
  return scdCmd(0x21B1);                                        // start periodic measurement (5 s)
}
static bool scdDataReady() { uint16_t w; return scdRead(0xE4B8, &w, 1, 1) && (w & 0x07FF) != 0; }
static bool scdMeasure(float &co2, float &t, float &rh) {
  uint16_t w[3];
  if (!scdRead(0xEC05, w, 3, 1)) return false;
  co2 = w[0]; t = -45.0 + 175.0 * w[1] / 65535.0; rh = 100.0 * w[2] / 65535.0;
  return w[0] > 0;
}

// ---------------------------------------------------------------- OLED
#if HAS_OLED
#include <U8g2lib.h>

#if OLED_SH1106
  U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
#elif OLED_H == 32
  U8G2_SSD1306_128X32_UNIVISION_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
#else
  U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
#endif

// pet_face.h draws on the u8g2 object above and must come after it. Including it
// before the declaration is a compile error with a confusing message ("'u8g2' was
// not declared in this scope; did you mean 'U8G2'?"), so the header checks for
// this marker and says what is actually wrong.
#define PET_FACE_U8G2_READY 1
#include "pet_face.h"


static bool oledOk = false;

static void oledBegin() {
  oledOk = u8g2.begin();
  // u8g2 raises the bus to 400 kHz; the SCD4x is specified for 100 kHz, so put it back.
  Wire.setClock(100000);
  Serial.println(oledOk ? "[oled] display initialised" : "[oled] display did NOT initialise - check wiring and the OLED_SH1106 / OLED_H settings");
}

/* One 1-bit face, drawn with primitives so no bitmaps are needed. */
static void drawFace(int cx, int cy, int r, const char *expr) {
  #define IS(x) (strcmp(expr, x) == 0)
  u8g2.drawCircle(cx, cy, r);
  const int ex = r / 2, ey = r / 3, my = cy + r / 3;

  if (IS("sleepy")) {                                  // closed eyes: short lines
    u8g2.drawHLine(cx - ex - 2, cy - ey, 5);
    u8g2.drawHLine(cx + ex - 2, cy - ey, 5);
  } else if (IS("alert")) {                            // wide open eyes
    u8g2.drawCircle(cx - ex, cy - ey, 3);
    u8g2.drawCircle(cx + ex, cy - ey, 3);
  } else {
    u8g2.drawDisc(cx - ex, cy - ey, 2);
    u8g2.drawDisc(cx + ex, cy - ey, 2);
  }

  if (IS("happy")) {                                   // smile: lower half of a circle
    u8g2.drawCircle(cx, my - r / 4, r / 2, U8G2_DRAW_LOWER_LEFT | U8G2_DRAW_LOWER_RIGHT);
  } else if (IS("sad")) {                              // frown: upper half
    u8g2.drawCircle(cx, my + r / 4, r / 2, U8G2_DRAW_UPPER_LEFT | U8G2_DRAW_UPPER_RIGHT);
  } else if (IS("alert")) {                            // open mouth
    u8g2.drawDisc(cx, my, 3);
  } else {                                             // neutral / concerned / sleepy
    u8g2.drawHLine(cx - r / 2, my, r);
  }

  if (IS("concerned") || IS("sad")) {                  // brows slanting in towards the nose
    u8g2.drawLine(cx - ex - 4, cy - ey - 7, cx - ex + 3, cy - ey - 4);
    u8g2.drawLine(cx + ex + 4, cy - ey - 7, cx + ex - 3, cy - ey - 4);
  }
  #undef IS
}

/* Expression to show: the host's choice wins; otherwise local CO2 (clearly a fallback). */
static const char *currentExpr() {
  if (dispExpr[0]) return dispExpr;
#if LOCAL_FACE_FALLBACK
  if (!haveReading) return "sleepy";
  if (lastCo2 < 800)  return "happy";
  if (lastCo2 < 1000) return "neutral";
  if (lastCo2 < 1500) return "concerned";
  return "alert";
#else
  return "neutral";
#endif
}

/* Message line: the host's message, else a short local status. */
static const char *currentMsg() {
  if (dispMsg[0]) return dispMsg;
  if (!timeOk) return "waiting for clock";
  if (WiFi.status() != WL_CONNECTED) return "offline - buffering";
  if (!uploadTried) return "first upload due";
  return uploadOk ? "sending to host" : "host unreachable";
}

static void oledBoot(const char *line) {
  if (!oledOk) return;
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tf);
  u8g2.drawStr(0, 10, "starting up");
  u8g2.drawStr(0, 22, ROOM_NAME);
  u8g2.drawStr(0, OLED_H == 32 ? 31 : 40, line);
  u8g2.sendBuffer();
}

static void oledRender() {
  if (!oledOk) return;
  static int msgOffset = 0;
  char l1[24], l2[24], l3[24], st[32];
  const char *expr = currentExpr(), *msg = currentMsg();
  u8g2.clearBuffer();

#if OLED_H == 32
  drawFace(15, 16, 13, expr);
  u8g2.setFont(u8g2_font_6x10_tf);
  if (haveReading) {
    snprintf(l1, sizeof l1, "%.0f ppm", lastCo2);
    snprintf(l2, sizeof l2, "%.1fC %.0f%%", lastT, lastRh);
  } else { snprintf(l1, sizeof l1, "no reading"); l2[0] = 0; }
  u8g2.drawStr(34, 13, l1);
  u8g2.drawStr(34, 25, l2);
  u8g2.setFont(u8g2_font_4x6_tf);
  snprintf(st, sizeof st, "%s Q%d", WiFi.status() == WL_CONNECTED ? "wifi" : "----", bufCount);
  u8g2.drawStr(34, 32, st);
#else
  drawFace(21, 21, 19, expr);

  u8g2.setFont(u8g2_font_6x10_tf);
  if (haveReading) {
    snprintf(l1, sizeof l1, "CO2 %.0f", lastCo2);
    snprintf(l2, sizeof l2, "%.1f C", lastT);
    snprintf(l3, sizeof l3, "%.0f %%RH", lastRh);
  } else {
    snprintf(l1, sizeof l1, "CO2 --");
    snprintf(l2, sizeof l2, "warming");
    snprintf(l3, sizeof l3, "up...");
  }
  u8g2.drawStr(46, 12, l1);
  u8g2.drawStr(46, 24, l2);
  u8g2.drawStr(46, 36, l3);

  // message line, scrolled one character per second when it is too long to fit
  const int maxc = 21;
  int len = strlen(msg);
  if (len <= maxc) {
    msgOffset = 0;
    u8g2.drawStr(0, 52, msg);
  } else {
    char win[maxc + 1];
    strncpy(win, msg + msgOffset, maxc); win[maxc] = 0;
    u8g2.drawStr(0, 52, win);
    if (++msgOffset > len - maxc) msgOffset = 0;
  }

  u8g2.setFont(u8g2_font_4x6_tf);
  snprintf(st, sizeof st, "%s %s Q%-3d %s", WiFi.status() == WL_CONNECTED ? "wifi" : "----",
           timeOk ? "clk" : "---", bufCount,
           !uploadTried ? "" : uploadOk ? "sent" : "FAIL");
  u8g2.drawStr(0, 63, st);
  u8g2.drawStr(128 - 4 * (int)strlen(ROOM_NAME), 63, ROOM_NAME);   // room label, right-aligned
#endif
  u8g2.sendBuffer();
}
#else
static void oledBegin() {}
static void oledBoot(const char *) {}
static void oledRender() {}
#endif

// ---------------------------------------------------------------- the pet
/* Everything below is the face as a face: full screen, animated, and answering to
 * being touched. The small-panel layout above is left exactly as it was, because a
 * 128x32 strip has no room for a face and numbers and the numbers are the point
 * there.
 */
#if HAS_OLED && OLED_H != 32
#if HAS_TOUCH
#include "pet_touch.h"
#endif

/* Bring up anything the pet needs. Called from setup().
 *
 * petTouchBegin() existed and was never called, so touchBaseInit stayed false and
 * petTouchPoll() returned on its first line for ever: the pad read as permanently
 * untouched, every gesture was dead, and nothing anywhere said so. A sensor that
 * is never initialised and never complains is the worst shape of bug, so the
 * no-touch build now prints that it has none rather than staying silent - the
 * serial log should always say what the board thinks it has. */
static void petBegin() {
#if HAS_TOUCH
  petTouchBegin();
#else
  Serial.println("[touch] not built in on this node (HAS_TOUCH is 0)");
#endif
}

enum PetMode { PET_FACE, PET_INFO };
static PetMode  petMode      = PET_FACE;
static uint32_t petModeUntil = 0;       // when the info screen gives up and goes back
static uint32_t petReactUntil = 0;      // while set, the pet's own mood beats the host's
static char     petReact[16] = "";
static uint32_t lastFrameMs = 0, lastTickMs = 0;
static const uint32_t FRAME_MS    = 33;     // ~30 fps, which is enough for a blink
static const uint32_t INFO_HOLD_MS = 12000;

/* The face the pet should be wearing, in priority order.
 *
 * Being stroked outranks the host, and briefly. The host's expression is a judgement
 * about the air that is still true a second later, so if petting merely nudged a
 * queue it would be overwritten by the next reply and the pet would ignore you. A
 * short override that expires is what makes the reaction feel like a reaction. */
static const char *petExpression(uint32_t now) {
  if (petReact[0] && now < petReactUntil) return petReact;
  return currentExpr();
}

static void petReactNow(const char *expr, uint32_t now, uint32_t ms) {
  strncpy(petReact, expr, sizeof petReact - 1);
  petReact[sizeof petReact - 1] = 0;
  petReactUntil = now + ms;
}

static void drawInfoScreen() {
  u8g2.setFont(u8g2_font_6x10_tf);
  if (!dispInfoN) {
    u8g2.drawStr(0, 12, "no advice yet");
    u8g2.setFont(u8g2_font_4x6_tf);
    u8g2.drawStr(0, 26, "the host has not answered");
    u8g2.drawStr(0, 34, "with anything to show.");
  } else {
    for (uint8_t i = 0; i < dispInfoN && i < INFO_LINES; i++)
      u8g2.drawStr(0, 11 + 13 * i, dispInfo[i]);
  }
  u8g2.setFont(u8g2_font_4x6_tf);
  char st[32];
  snprintf(st, sizeof st, "%s  %s  Q%d", ROOM_NAME,
           WiFi.status() == WL_CONNECTED ? "wifi" : "----", bufCount);
  u8g2.drawStr(0, 63, st);
}

/* One frame. The I2C clock is raised for the transfer and put back afterwards.
 *
 * A full 128x64 frame is 1024 bytes, and at the 100 kHz the SCD4x is specified for
 * that is about 90 ms on the wire - under 11 fps, with the processor blocked for all
 * of it. The display is good for 400 kHz even though the sensor is not, and nothing
 * else is on the bus mid-frame, so the clock goes up for the transfer and comes back
 * down before anything speaks to the sensor again. The same frame then costs about
 * 23 ms and the blink looks like a blink. */
static void petRender(uint32_t now) {
  if (!oledOk) return;
  u8g2.clearBuffer();
  if (petMode == PET_INFO) drawInfoScreen();
  else                     drawPetFace(now);
  Wire.setClock(400000);
  u8g2.sendBuffer();
  Wire.setClock(100000);
}

/* Run the face and the touch pad for about `ms`, instead of sleeping through it.
 *
 * The main loop used to end in delay(1000), which is fine for a display that shows
 * numbers and fatal for one that animates: nothing can blink if the processor is
 * asleep 97% of the time. The per-second work above is untouched - this only fills
 * the gap that was already idle. */
static void petIdle(uint32_t ms) {
  uint32_t until = millis() + ms;
  for (;;) {
    uint32_t now = millis();
    if ((int32_t)(now - until) >= 0) break;

#if HAS_TOUCH
    switch (petTouchPoll(now)) {
      case TOUCH_PET:
        // Being stroked. Up on the hind legs, which is the one expression the air
        // quality can never produce, so it always reads as "that was you".
        petReactNow("excited", now, 2500);
        petMode = PET_FACE;
        break;
      case TOUCH_DOUBLE:
        petMode = PET_INFO;
        petModeUntil = now + INFO_HOLD_MS;
        break;
      case TOUCH_TAP:
        // A single tap acknowledges rather than navigates: it looks at you. Leaving
        // it inert would make the pad feel broken on the way to a double tap.
        if (petMode == PET_INFO) { petMode = PET_FACE; }
        else                     { petReactNow("alert", now, 1200); }
        break;
      default: break;
    }
#endif
    if (petMode == PET_INFO && (int32_t)(now - petModeUntil) >= 0) petMode = PET_FACE;

    faceGoal = faceFor(petExpression(now));
    uint32_t dt = lastTickMs ? now - lastTickMs : FRAME_MS;
    if (dt > 500) dt = 500;        // came back from a long HTTP post; do not lurch
    lastTickMs = now;
    faceTick(now, dt);

    if (now - lastFrameMs >= FRAME_MS) { lastFrameMs = now; petRender(now); }
    delay(4);
  }
}
#else
static void petIdle(uint32_t ms) { delay(ms); }
static void petBegin() {}
#endif

// ---------------------------------------------------------------- diagnostics
static void i2cScan() {
  Serial.println("[i2c] scanning bus (SDA=GPIO21, SCL=GPIO22)...");
  int found = 0;
  bool oled = false;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() != 0) continue;
    found++;
    if (a == 0x3C || a == 0x3D) oled = true;
    const char *what = a == SCD_ADDR ? "SCD40 CO2 sensor"
                     : (a == 0x3C || a == 0x3D) ? "OLED display"
                     : "unknown device";
    Serial.printf("[i2c]   found 0x%02X  %s\n", a, what);
  }
  if (!found) Serial.println("[i2c] NOTHING found -> check 3V3 + GND, and that SDA/SCL are not swapped or loose");
  else Serial.printf("[i2c] %d device(s) on the bus\n", found);
  if (HAS_OLED && !oled) Serial.println("[i2c] this node expects a display but none answered at 0x3C/0x3D -> check its wiring");
  if (!HAS_OLED && oled) Serial.println("[i2c] a display is wired but NODE says this board has none (set HAS_OLED in secrets.h)");
}

static const char *wifiStatusText(wl_status_t s) {
  switch (s) {
    case WL_CONNECTED:       return "connected";
    case WL_NO_SSID_AVAIL:   return "network name not found (check WIFI_SSID spelling; ESP32 needs 2.4 GHz, not 5 GHz)";
    case WL_CONNECT_FAILED:  return "connection failed (wrong WIFI_PASS?)";
    case WL_CONNECTION_LOST: return "connection lost";
    case WL_DISCONNECTED:    return "not connected yet";
    case WL_IDLE_STATUS:     return "idle / connecting";
    default:                 return "unknown state";
  }
}

static wl_status_t lastWifiState = (wl_status_t)255;
static uint32_t lastWifiMsg = 0;
static void reportWifi() {
  wl_status_t s = WiFi.status();
  if (s != lastWifiState) {
    lastWifiState = s; lastWifiMsg = millis();
    if (s == WL_CONNECTED)
      Serial.printf("[wifi] CONNECTED to \"%s\"  ip=%s  signal=%d dBm\n", WIFI_SSID, WiFi.localIP().toString().c_str(), WiFi.RSSI());
    else
      Serial.printf("[wifi] %s\n", wifiStatusText(s));
  } else if (s != WL_CONNECTED && millis() - lastWifiMsg > 10000) {
    lastWifiMsg = millis();
    Serial.printf("[wifi] still trying to join \"%s\": %s\n", WIFI_SSID, wifiStatusText(s));
  }
}

// ---------------------------------------------------------------- buffering / upload
static void pushReading(const Reading &r) {
  int idx = (bufHead + bufCount) % BUF_SIZE;
  buf[idx] = r;
  if (bufCount < BUF_SIZE) bufCount++; else bufHead = (bufHead + 1) % BUF_SIZE;   // drop oldest
}

/* Pull one "key":"value" string out of the reply without a JSON library. */
static bool jsonStr(const String &s, const char *key, char *out, size_t n) {
  String pat = String("\"") + key + "\":\"";
  int i = s.indexOf(pat);
  if (i < 0) return false;
  i += pat.length();
  int j = s.indexOf('"', i);
  if (j < 0) return false;
  s.substring(i, j).toCharArray(out, n);
  return true;
}

/* Pull "info":["a","b"] out of the reply into dispInfo. Returns how many it found.
 *
 * Hand-written for the same reason jsonStr is: pulling in a JSON library to read two
 * keys would cost more flash than the whole display does. It scans the array between
 * its brackets and takes quoted strings, which is enough for a field this node only
 * ever renders and never interprets. */
static uint8_t jsonStrArray(const String &s, const char *key,
                            char out[][INFO_CHARS], uint8_t maxN) {
  String pat = String("\"") + key + "\":[";
  int i = s.indexOf(pat);
  if (i < 0) return 0;
  i += pat.length();
  int end = s.indexOf(']', i);
  if (end < 0) return 0;
  uint8_t n = 0;
  while (n < maxN) {
    int a = s.indexOf('"', i);
    if (a < 0 || a > end) break;
    int b = s.indexOf('"', a + 1);
    if (b < 0 || b > end) break;
    s.substring(a + 1, b).toCharArray(out[n], INFO_CHARS);
    n++;
    i = b + 1;
  }
  return n;
}

/* Send one batch. Returns 1 sent, 0 nothing to send, -1 the host would not take it. */
static int uploadBatch() {
  if (bufCount == 0) return 0;
  if (WiFi.status() != WL_CONNECTED) { Serial.printf("[upload] skipped: WiFi is down, %d reading(s) waiting in memory\n", bufCount); return -1; }
  int n = bufCount < BATCH_MAX ? bufCount : BATCH_MAX;
  String body = "{\"device\":\"" DEVICE_NAME "\",\"room\":\"" ROOM_NAME "\",\"readings\":[";
  for (int i = 0; i < n; i++) {
    const Reading &r = buf[(bufHead + i) % BUF_SIZE];
    char item[320];
    int k = snprintf(item, sizeof item,
                     "%s{\"ts\":%lu,\"co2\":%.1f,\"co2_max\":%.0f,\"t\":%.2f,\"rh\":%.1f,\"n\":%u",
                     i ? "," : "", (unsigned long)r.ts, r.co2, r.co2_max, r.t, r.rh, r.n);
    // Particle fields appear only on the minutes that carry them. An absent field
    // and a zero are different claims - zero is clean air, absent is no measurement
    // - and a sensor that samples every five minutes must not assert the first.
    if (r.hasPm && k > 0 && k < (int)sizeof item)
      snprintf(item + k, sizeof item - k,
               ",\"pm1\":%u,\"pm25\":%u,\"pm10\":%u,"
               "\"n03\":%u,\"n05\":%u,\"n10\":%u,\"n25\":%u,\"n50\":%u,\"n100\":%u}",
               r.pm.pm1, r.pm.pm25, r.pm.pm10,
               r.pm.n03, r.pm.n05, r.pm.n10, r.pm.n25, r.pm.n50, r.pm.n100);
    else
      snprintf(item + k, sizeof item - k, "}");
    body += item;
  }
  body += "]}";
  Serial.printf("[upload] sending %d reading(s) to %s ...\n", n, INGEST_URL);
  HTTPClient http;
  http.setTimeout(8000);
  http.begin(INGEST_URL);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Auth-Token", INGEST_TOKEN);
  int code = http.POST(body);
  uploadTried = true;
  if (code == 200) {
    uploadOk = true;
    String resp = http.getString();
    /* 200 means the host answered, not that it kept anything.
     *
     * The reply carries "accepted": how many of the n readings passed validation
     * and were queued. This used to be ignored and the buffer was dropped on the
     * status code alone, so a batch the host threw away - every reading failing
     * its range check, or an NTP slip putting every timestamp outside the accepted
     * window - was deleted from the only other copy in existence. It is not
     * hypothetical: this host has recorded 59 silently rejected readings from the
     * bedroom node, which is 59 minutes of air nobody can get back.
     *
     * So the batch is only released once the host says it took all of it.
     * Readings the host will never accept would otherwise block the queue for
     * ever, so a batch rejected twice is dropped and said so loudly - the point is
     * that it is reported, not that it is kept at any cost.
     */
    int acc = n;
    int ai = resp.indexOf("\"accepted\":");
    if (ai >= 0) acc = resp.substring(ai + 11).toInt();
    if (acc >= n) {
      bufHead = (bufHead + n) % BUF_SIZE; bufCount -= n;        // all of it stored
      rejectStreak = 0;
    } else {
      rejectStreak++;
      Serial.printf("[upload] host accepted only %d of %d - KEEPING the batch "
                    "(attempt %d). Check the clock and the CO2 range.",
                    acc, n, rejectStreak);
      if (rejectStreak >= 2) {
        bufHead = (bufHead + n) % BUF_SIZE; bufCount -= n;
        rejectStreak = 0;
        Serial.println("[upload] dropping it: the host will not take these readings. "
                       "THIS IS DATA LOSS and it is deliberate - the queue cannot "
                       "stall for ever on a batch nobody will store.");
      }
    }
    int k = resp.indexOf("\"pressure_hpa\":");
    if (k >= 0) {
      float p = resp.substring(k + 15).toFloat();               // 'null' parses to 0 -> ignored
      if (p > 800 && p < 1100) {
        scdCmdArg(0xE000, (uint16_t)p);                         // ambient pressure compensation
        Serial.printf("[upload] host supplied pressure %.0f hPa -> applied to sensor\n", p);
      }
    }
    char e[sizeof dispExpr], m[sizeof dispMsg];                 // host's pet state, if it sent any
    if (jsonStr(resp, "expression", e, sizeof e) && e[0]) {
      bool changed = strcmp(e, dispExpr) != 0;
      strncpy(dispExpr, e, sizeof dispExpr - 1); dispExpr[sizeof dispExpr - 1] = 0;
      if (jsonStr(resp, "message", m, sizeof m)) {
        changed |= strcmp(m, dispMsg) != 0;
        strncpy(dispMsg, m, sizeof dispMsg - 1); dispMsg[sizeof dispMsg - 1] = 0;
      }
      if (changed) Serial.printf("[oled] host says: %s - \"%s\"\n", dispExpr, dispMsg);
    }
    // Lines for the info screen. Absent is not empty: if the host sent no "info" at
    // all the previous lines stay up, because a reply that simply did not mention it
    // is not the host saying there is nothing to report.
    uint8_t ni = jsonStrArray(resp, "info", dispInfo, INFO_LINES);
    if (ni) dispInfoN = ni;
    Serial.printf("[upload] OK - host accepted %d reading(s). Backlog now %d. THE WHOLE CHAIN WORKS.\n", n, bufCount);
    http.end();
    return 1;
  }
  uploadOk = false;
  if (code < 0) {
    Serial.printf("[upload] FAILED: %s\n", HTTPClient::errorToString(code).c_str());
    Serial.println("[upload]   -> host unreachable. Check: INGEST_URL IP is the laptop's current IP,");
    Serial.println("[upload]      'python main.py collect' is running, and Windows Firewall allows TCP 8765.");
  } else if (code == 401 || code == 403) {
    Serial.printf("[upload] FAILED: HTTP %d - the host reached but REJECTED the token. INGEST_TOKEN in secrets.h must equal INGEST_TOKEN in .env\n", code);
  } else {
    Serial.printf("[upload] FAILED: HTTP %d from host: %s\n", code, http.getString().c_str());
  }
  Serial.printf("[upload]   %d reading(s) kept in memory, will retry in %lu s\n", bufCount, (unsigned long)(UPLOAD_EVERY_MS / 1000));
  http.end();
  return -1;
}

/* Drain the backlog in a loop, never by recursion.
 *
 * A full 48 h buffer is 96 batches. Ninety-six nested calls, each holding an
 * HTTPClient and a 3 kB request body that is not released until the frame returns,
 * would exhaust the task stack and the heap long before the backlog cleared - and a
 * crash there loses the very data the buffer existed to protect.
 *
 * Only a few batches go out per call so the sensor keeps being read while catching
 * up; if a backlog remains the next attempt is seconds away instead of a minute, so
 * a full buffer still clears in a couple of minutes. */
static void upload() {
  if (bufCount == 0) {
    Serial.println("[upload] nothing to send yet (the first reading is stored after one full minute)");
    return;
  }
  for (int sent = 0; sent < MAX_BATCHES_PER_CALL; sent++) {
    if (uploadBatch() <= 0) return;                 // done, or the host is not taking it
    if (bufCount == 0) return;
  }
  Serial.printf("[upload] %d still queued, carrying on in %lu s\n",
                bufCount, (unsigned long)(CATCHUP_MS / 1000));
  lastUpload = millis() - UPLOAD_EVERY_MS + CATCHUP_MS;
}

// ---------------------------------------------------------------- main
static double sumCo2 = 0, sumT = 0, sumRh = 0, maxCo2 = 0;
static int nSamples = 0;
static uint32_t curMinute = 0, lastWifi = 0;
static bool timeReported = false, sensorOk = false;
static uint32_t lastTimeMsg = 0, lastGoodSample = 0, lastSensorWarn = 0;

void setup() {
  Serial.begin(115200);
  delay(1500);                                                   // give the serial monitor time to attach
  Serial.println();
  Serial.println("==================================================");
  Serial.println("[boot] ESP32 SCD40 node starting (serial 115200 baud)");
  /* Say which build this is, in one line, before anything can go wrong.
   *
   * "The bedroom node is sending but I see nothing about the PMS" is not a wiring
   * question, it is a which-build-am-I-running question - and the board knew the
   * answer all along and was not saying. A peripheral compiled out is
   * indistinguishable from one that is broken unless the banner names it. */
  Serial.printf("[boot] device \"%s\" in room \"%s\" (NODE %d)\n",
                DEVICE_NAME, ROOM_NAME, (int)NODE);
  Serial.printf("[boot] built with: display %s | particle counter %s | touch pad %s\n",
                HAS_OLED ? "YES" : "no", HAS_PMS ? "YES" : "no",
                HAS_TOUCH ? "YES" : "no");
  if (!HAS_PMS)
    Serial.println("[boot] no PMS in this build - if you expected one, NODE is "
                   "wrong in secrets.h (bedroom is NODE 2) and this board needs "
                   "reflashing");
  Serial.printf("[boot] will join WiFi \"%s\" and upload to %s\n", WIFI_SSID, INGEST_URL);
  Serial.printf("[boot] token is %s (%d chars)\n", strlen(INGEST_TOKEN) ? "set" : "EMPTY!", (int)strlen(INGEST_TOKEN));

  Wire.begin(21, 22); Wire.setClock(100000);
#if HAS_PMS
  pmsBegin();
#else
  Serial.println("[pms] no particle counter on this node (set HAS_PMS in secrets.h)");
#endif
  i2cScan();
  oledBegin();
  petBegin();
  oledBoot("sensor...");

  sensorOk = scdInit();
  if (sensorOk) Serial.println("[scd40] started periodic measurement; first reading in ~5 s");
  else Serial.println("[scd40] NOT RESPONDING at 0x62 -> check wiring (SDA=GPIO21, SCL=GPIO22, VDD=3V3, GND)");
  lastGoodSample = millis();

  oledBoot(sensorOk ? "wifi..." : "SENSOR FAIL");
  Serial.println("[wifi] connecting...");
  WiFi.mode(WIFI_STA); WiFi.begin(WIFI_SSID, WIFI_PASS);
  configTime(0, 0, "pool.ntp.org", "time.google.com");           // UTC; host stores unix seconds
}

void loop() {
#if HAS_PMS
  pmsPoll();          // non-blocking: sleeps the fan between samples
#endif
  reportWifi();
  if (WiFi.status() != WL_CONNECTED && millis() - lastWifi > 30000) { lastWifi = millis(); WiFi.reconnect(); }

  time_t now = time(nullptr);
  timeOk = now >= 1700000000;                                    // before NTP the clock reads 1970
  if (timeOk && !timeReported) {
    timeReported = true;
    struct tm tmv; gmtime_r(&now, &tmv);
    Serial.printf("[time] clock synced from NTP: %04d-%02d-%02d %02d:%02d:%02d UTC\n",
                  tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
  } else if (!timeOk && millis() - lastTimeMsg > 5000) {
    lastTimeMsg = millis();
    Serial.println(WiFi.status() == WL_CONNECTED ? "[time] waiting for NTP clock..." : "[time] no clock yet - needs WiFi first");
  }

  // sensor is read (and shown) even before the clock is set; readings are only stored once time is valid
  if (scdDataReady()) {
    float co2, t, rh;
    if (scdMeasure(co2, t, rh)) {
      lastGoodSample = millis(); sensorOk = true;
      lastCo2 = co2; lastT = t; lastRh = rh; haveReading = true;
      Serial.printf("[scd40] CO2 %4.0f ppm | %.1f C | %.0f %% RH%s\n", co2, t, rh, timeOk ? "" : "   (not stored yet: no clock)");
      if (timeOk) { sumCo2 += co2; sumT += t; sumRh += rh; nSamples++; if (co2 > maxCo2) maxCo2 = co2; }
    } else {
      Serial.println("[scd40] read failed (bad CRC or no data) - will retry");
    }
  } else if (millis() - lastGoodSample > 20000 && millis() - lastSensorWarn > 20000) {
    lastSensorWarn = millis();
    Serial.println("[scd40] WARNING: no new reading for 20 s -> sensor not responding, check wiring/power");
  }

  if (timeOk) {
    uint32_t minute = now / 60;
    if (curMinute == 0) curMinute = minute;
    if (minute != curMinute) {                                   // minute rolled over: finalise window
      if (nSamples > 0) {
        Reading r = { (uint32_t)(curMinute * 60 + 60), (float)(sumCo2 / nSamples), (float)maxCo2,
                      (float)(sumT / nSamples), (float)(sumRh / nSamples), (uint8_t)nSamples,
                      0, {0, 0, 0, 0, 0, 0, 0, 0, 0} };
#if HAS_PMS
        // A particle sample belongs to the minute it was taken in, and to no other.
        // Consuming it here rather than repeating it means a minute either carries a
        // real measurement or carries none, and the host is never handed the same
        // reading five times over as though the air had been sampled five times.
        if (pmHave) { r.hasPm = 1; r.pm = pmLatest; pmHave = false; }
#endif
        pushReading(r);                                          // ts = window END, same as the host convention
        Serial.printf("[minute] stored 1-min average: CO2 %.0f ppm, %.1f C, %.0f %% RH (%d samples). Waiting to upload: %d\n",
                      r.co2, r.t, r.rh, nSamples, bufCount);
        if (r.hasPm)
          Serial.printf("[minute]   with particles: PM2.5 %u ug/m3, %u per 0.1 L above 0.3 um\n",
                        r.pm.pm25, r.pm.n03);
      } else {
        Serial.println("[minute] window had no valid samples - nothing stored");
      }
      sumCo2 = sumT = sumRh = maxCo2 = 0; nSamples = 0; curMinute = minute;
    }
  }

  if (millis() - lastUpload > UPLOAD_EVERY_MS) { lastUpload = millis(); upload(); }
#if HAS_OLED && OLED_H != 32
  petIdle(1000);          // the second that used to be delay(1000), spent animating
#else
  oledRender();
  delay(1000);
#endif
}
