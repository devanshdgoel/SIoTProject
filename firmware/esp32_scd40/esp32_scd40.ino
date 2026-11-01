/*
 * esp32_scd40.ino  -  indoor sensor node
 *
 * Reads a Sensirion SCD40 (CO2, temperature, RH) every 5 s, averages each
 * minute, timestamps it with NTP time, buffers it in RAM and POSTs batches to
 * the host's /ingest endpoint (collectors/scd40.py, backend "http").
 * If WiFi or the host is down the node keeps sampling (up to ~48 h buffered) and
 * uploads the backlog when the link returns; the host de-duplicates by timestamp.
 *
 * Wiring (3.3 V only):  SCD40 VDD->3V3, GND->GND, SDA->GPIO21, SCL->GPIO22.
 * Mount the sensor on a short lead away from the ESP32 module: WiFi transmit
 * heat raises the reading; tune TEMP_OFFSET_C against a reference thermometer.
 *
 * The SCD4x I2C protocol is implemented directly (no library) so the sketch does
 * not depend on a particular library version. Board: "ESP32 Dev Module".
 */
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <time.h>
#include "secrets.h"

static const uint8_t  SCD_ADDR       = 0x62;
static const float    TEMP_OFFSET_C  = 4.0;      // self-heating compensation
static const int      BUF_SIZE       = 2880;     // minutes of backlog (~48 h)
static const int      BATCH_MAX      = 30;       // readings per POST
static const uint32_t UPLOAD_EVERY_MS = 60000;

struct Reading { uint32_t ts; float co2, co2_max, t, rh; uint8_t n; };
static Reading buf[BUF_SIZE];
static int bufHead = 0, bufCount = 0;            // ring buffer: oldest at bufHead

// ---------------------------------------------------------------- SCD4x I2C
static uint8_t crc8(const uint8_t *d, int n) {
  uint8_t c = 0xFF;
  for (int i = 0; i < n; i++) { c ^= d[i]; for (int b = 0; b < 8; b++) c = (c & 0x80) ? (c << 1) ^ 0x31 : (c << 1); }
  return c;
}
static void scdCmd(uint16_t cmd) {
  Wire.beginTransmission(SCD_ADDR); Wire.write(cmd >> 8); Wire.write(cmd & 0xFF); Wire.endTransmission();
}
static void scdCmdArg(uint16_t cmd, uint16_t arg) {
  uint8_t a[2] = { (uint8_t)(arg >> 8), (uint8_t)(arg & 0xFF) };
  Wire.beginTransmission(SCD_ADDR); Wire.write(cmd >> 8); Wire.write(cmd & 0xFF);
  Wire.write(a[0]); Wire.write(a[1]); Wire.write(crc8(a, 2)); Wire.endTransmission();
}
static bool scdRead(uint16_t cmd, uint16_t *out, int words, int waitMs) {
  scdCmd(cmd); delay(waitMs);
  if (Wire.requestFrom((int)SCD_ADDR, words * 3) != words * 3) return false;
  for (int i = 0; i < words; i++) {
    uint8_t b[3] = { (uint8_t)Wire.read(), (uint8_t)Wire.read(), (uint8_t)Wire.read() };
    if (crc8(b, 2) != b[2]) return false;
    out[i] = (b[0] << 8) | b[1];
  }
  return true;
}
static void scdInit() {
  scdCmd(0x3F86); delay(500);                                   // stop periodic measurement
  scdCmdArg(0x2416, 0); delay(1);                               // automatic self-calibration OFF (occupied room)
  scdCmdArg(0x241D, (uint16_t)(TEMP_OFFSET_C * 65535.0 / 175.0)); delay(1);
  scdCmd(0x21B1);                                               // start periodic measurement (5 s)
}
static bool scdDataReady() { uint16_t w; return scdRead(0xE4B8, &w, 1, 1) && (w & 0x07FF) != 0; }
static bool scdMeasure(float &co2, float &t, float &rh) {
  uint16_t w[3];
  if (!scdRead(0xEC05, w, 3, 1)) return false;
  co2 = w[0]; t = -45.0 + 175.0 * w[1] / 65535.0; rh = 100.0 * w[2] / 65535.0;
  return w[0] > 0;
}

// ---------------------------------------------------------------- buffering / upload
static void pushReading(const Reading &r) {
  int idx = (bufHead + bufCount) % BUF_SIZE;
  buf[idx] = r;
  if (bufCount < BUF_SIZE) bufCount++; else bufHead = (bufHead + 1) % BUF_SIZE;   // drop oldest
}

static void upload() {
  if (bufCount == 0 || WiFi.status() != WL_CONNECTED) return;
  int n = bufCount < BATCH_MAX ? bufCount : BATCH_MAX;
  String body = "{\"device\":\"esp32-1\",\"readings\":[";
  for (int i = 0; i < n; i++) {
    const Reading &r = buf[(bufHead + i) % BUF_SIZE];
    char item[128];
    snprintf(item, sizeof item, "%s{\"ts\":%lu,\"co2\":%.1f,\"co2_max\":%.0f,\"t\":%.2f,\"rh\":%.1f,\"n\":%u}",
             i ? "," : "", (unsigned long)r.ts, r.co2, r.co2_max, r.t, r.rh, r.n);
    body += item;
  }
  body += "]}";
  HTTPClient http;
  http.setTimeout(8000);
  http.begin(INGEST_URL);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Auth-Token", INGEST_TOKEN);
  int code = http.POST(body);
  if (code == 200) {
    bufHead = (bufHead + n) % BUF_SIZE; bufCount -= n;          // acknowledged: drop from buffer
    String resp = http.getString();
    int k = resp.indexOf("\"pressure_hpa\":");
    if (k >= 0) {
      float p = resp.substring(k + 15).toFloat();               // 'null' parses to 0 -> ignored
      if (p > 800 && p < 1100) scdCmdArg(0xE000, (uint16_t)p);  // ambient pressure compensation
    }
    Serial.printf("uploaded %d, backlog %d\n", n, bufCount);
    if (bufCount > 0) upload();                                 // drain backlog promptly
  } else {
    Serial.printf("upload failed (%d), backlog %d\n", code, bufCount);
  }
  http.end();
}

// ---------------------------------------------------------------- main
static double sumCo2 = 0, sumT = 0, sumRh = 0, maxCo2 = 0;
static int nSamples = 0;
static uint32_t curMinute = 0, lastUpload = 0, lastWifi = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); Wire.setClock(100000);
  scdInit();
  WiFi.mode(WIFI_STA); WiFi.begin(WIFI_SSID, WIFI_PASS);
  configTime(0, 0, "pool.ntp.org", "time.google.com");           // UTC; host stores unix seconds
}

void loop() {
  time_t now = time(nullptr);
  if (now < 1700000000) { delay(500); return; }                  // wait for NTP before sampling

  if (WiFi.status() != WL_CONNECTED && millis() - lastWifi > 30000) { lastWifi = millis(); WiFi.reconnect(); }

  uint32_t minute = now / 60;
  if (curMinute == 0) curMinute = minute;
  if (minute != curMinute) {                                     // minute rolled over: finalise window
    if (nSamples > 0) {
      Reading r = { (uint32_t)(curMinute * 60 + 60), (float)(sumCo2 / nSamples), (float)maxCo2,
                    (float)(sumT / nSamples), (float)(sumRh / nSamples), (uint8_t)nSamples };
      pushReading(r);                                            // ts = window END, same as the host convention
      Serial.printf("%lu co2=%.0f t=%.1f rh=%.0f n=%d\n", (unsigned long)r.ts, r.co2, r.t, r.rh, nSamples);
    }
    sumCo2 = sumT = sumRh = maxCo2 = 0; nSamples = 0; curMinute = minute;
  }

  if (scdDataReady()) {
    float co2, t, rh;
    if (scdMeasure(co2, t, rh)) { sumCo2 += co2; sumT += t; sumRh += rh; nSamples++; if (co2 > maxCo2) maxCo2 = co2; }
  }

  if (millis() - lastUpload > UPLOAD_EVERY_MS) { lastUpload = millis(); upload(); }
  delay(1000);
}
