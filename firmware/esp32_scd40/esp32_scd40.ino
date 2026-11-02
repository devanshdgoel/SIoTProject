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
#ifndef HAS_OLED
#define HAS_OLED 0        // 1 only on the node that has a display wired
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
static const int      BUF_SIZE       = 2880;     // minutes of backlog (~48 h)
static const int      BATCH_MAX      = 30;       // readings per POST
static const int      MAX_BATCHES_PER_CALL = 8;  // keep sampling while a backlog drains
static const uint32_t UPLOAD_EVERY_MS = 60000;
static const uint32_t CATCHUP_MS      = 5000;    // retry sooner while still behind

struct Reading { uint32_t ts; float co2, co2_max, t, rh; uint8_t n; };
static Reading buf[BUF_SIZE];
static int bufHead = 0, bufCount = 0;            // ring buffer: oldest at bufHead

// Shared node state, also what the display shows.
static float lastCo2 = 0, lastT = 0, lastRh = 0;
static bool  haveReading = false, timeOk = false, uploadOk = false, uploadTried = false;
static uint32_t lastUpload = 0;                  // used by upload() to pace catching up
static char  dispExpr[16] = "";                  // from the host; empty until it replies
static char  dispMsg[64]  = "";

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

/* Send one batch. Returns 1 sent, 0 nothing to send, -1 the host would not take it. */
static int uploadBatch() {
  if (bufCount == 0) return 0;
  if (WiFi.status() != WL_CONNECTED) { Serial.printf("[upload] skipped: WiFi is down, %d reading(s) waiting in memory\n", bufCount); return -1; }
  int n = bufCount < BATCH_MAX ? bufCount : BATCH_MAX;
  String body = "{\"device\":\"" DEVICE_NAME "\",\"room\":\"" ROOM_NAME "\",\"readings\":[";
  for (int i = 0; i < n; i++) {
    const Reading &r = buf[(bufHead + i) % BUF_SIZE];
    char item[128];
    snprintf(item, sizeof item, "%s{\"ts\":%lu,\"co2\":%.1f,\"co2_max\":%.0f,\"t\":%.2f,\"rh\":%.1f,\"n\":%u}",
             i ? "," : "", (unsigned long)r.ts, r.co2, r.co2_max, r.t, r.rh, r.n);
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
    bufHead = (bufHead + n) % BUF_SIZE; bufCount -= n;          // acknowledged: drop from buffer
    String resp = http.getString();
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
  Serial.printf("[boot] device \"%s\" in room \"%s\" (display %s)\n",
                DEVICE_NAME, ROOM_NAME, HAS_OLED ? "expected" : "none");
  Serial.printf("[boot] will join WiFi \"%s\" and upload to %s\n", WIFI_SSID, INGEST_URL);
  Serial.printf("[boot] token is %s (%d chars)\n", strlen(INGEST_TOKEN) ? "set" : "EMPTY!", (int)strlen(INGEST_TOKEN));

  Wire.begin(21, 22); Wire.setClock(100000);
  i2cScan();
  oledBegin();
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
                      (float)(sumT / nSamples), (float)(sumRh / nSamples), (uint8_t)nSamples };
        pushReading(r);                                          // ts = window END, same as the host convention
        Serial.printf("[minute] stored 1-min average: CO2 %.0f ppm, %.1f C, %.0f %% RH (%d samples). Waiting to upload: %d\n",
                      r.co2, r.t, r.rh, nSamples, bufCount);
      } else {
        Serial.println("[minute] window had no valid samples - nothing stored");
      }
      sumCo2 = sumT = sumRh = maxCo2 = 0; nSamples = 0; curMinute = minute;
    }
  }

  if (millis() - lastUpload > UPLOAD_EVERY_MS) { lastUpload = millis(); upload(); }
  oledRender();
  delay(1000);
}
