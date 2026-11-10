/* pet_spill.h - when the host is gone for a long time, keep the readings in flash.
 *
 * The RAM ring was the whole backlog, and RAM is the wrong place for a backlog that
 * has to survive anything. Two failures made that concrete. A 35-hour outage cost
 * 23 hours of data because the ring held only the most recent 12. And an hour-long
 * window test was lost outright because the board was power-cycled, which empties
 * RAM entirely however much of it there is.
 *
 * Making the ring bigger fixes neither properly: it costs DRAM the WiFi stack needs
 * - 1440 readings is about 56 KB, which is why it was halved - and it still dies
 * with the power. So the ring gets SMALLER and flash takes the weight.
 *
 *   RAM   SPILL_RAM readings, the hot buffer, written every minute
 *   flash spilled in blocks only when the host has been unreachable for a while
 *
 * The split matters for flash life. A healthy node never writes to flash at all:
 * readings arrive, upload a minute later, and the ring never fills. Flash is
 * touched only when uploads have been failing, which is rare and is exactly when
 * the data is worth keeping. At one block per SPILL_EVERY minutes of outage, a
 * decade of outages is a few thousand writes against a rated 100,000 per sector,
 * with LittleFS wear-levelling on top.
 *
 * The file is append-only with a separate read offset, so a crash mid-write loses
 * at most the last block rather than the whole backlog, and the offset advances
 * only after the host has confirmed it stored the batch.
 */

#ifndef PET_SPILL_H
#define PET_SPILL_H

#include <LittleFS.h>

static const char *SPILL_PATH = "/backlog.bin";
static const char *SPILL_OFF  = "/backlog.off";

// Stop growing the file somewhere short of the partition. A week of minutes at
// ~48 bytes is about 480 KB; this leaves room for that several times over while
// never filling the filesystem, because a full filesystem fails in ways that are
// much harder to reason about than a dropped oldest reading.
static const uint32_t SPILL_MAX_BYTES = 900000UL;

static bool spillReady = false;

static void spillBegin() {
  spillReady = LittleFS.begin(true);              // format on first run
  if (!spillReady) {
    Serial.println("[spill] LittleFS would not mount - backlog stays in RAM only");
    return;
  }
  File f = LittleFS.open(SPILL_PATH, "r");
  uint32_t bytes = f ? f.size() : 0;
  if (f) f.close();
  Serial.printf("[spill] flash backlog ready, %lu bytes held (%lu readings)\n",
                (unsigned long)bytes, (unsigned long)(bytes / sizeof(Reading)));
}

static uint32_t spillOffset() {
  if (!spillReady) return 0;
  File f = LittleFS.open(SPILL_OFF, "r");
  if (!f) return 0;
  uint32_t off = 0;
  f.read((uint8_t *)&off, sizeof off);
  f.close();
  return off;
}

static void spillSetOffset(uint32_t off) {
  if (!spillReady) return;
  File f = LittleFS.open(SPILL_OFF, "w");
  if (!f) return;
  f.write((const uint8_t *)&off, sizeof off);
  f.close();
}

/* How many readings are waiting in flash. */
static uint32_t spillCount() {
  if (!spillReady) return 0;
  File f = LittleFS.open(SPILL_PATH, "r");
  if (!f) return 0;
  uint32_t size = f.size();
  f.close();
  uint32_t off = spillOffset();
  return off >= size ? 0 : (size - off) / sizeof(Reading);
}

/* Everything sent: throw the file away rather than letting it grow for ever. */
static void spillClear() {
  if (!spillReady) return;
  LittleFS.remove(SPILL_PATH);
  LittleFS.remove(SPILL_OFF);
}

/* Append `n` readings from the ring to flash. Returns how many were written. */
static int spillWrite(const Reading *ring, int ringSize, int head, int n) {
  if (!spillReady || n <= 0) return 0;
  File f = LittleFS.open(SPILL_PATH, "a");
  if (!f) {
    Serial.println("[spill] could not open the backlog for writing");
    return 0;
  }
  if (f.size() > SPILL_MAX_BYTES) {
    f.close();
    Serial.println("[spill] flash backlog is full - dropping the OLDEST readings, "
                   "which are the least useful ones to keep");
    return 0;
  }
  int written = 0;
  for (int i = 0; i < n; i++) {
    const Reading &r = ring[(head + i) % ringSize];
    if (f.write((const uint8_t *)&r, sizeof r) != sizeof r) break;
    written++;
  }
  f.close();
  Serial.printf("[spill] moved %d reading(s) to flash; %lu now waiting there\n",
                written, (unsigned long)spillCount());
  return written;
}

/* Read up to `max` readings from the front of the flash backlog, oldest first.
 * Nothing is consumed here - the offset moves only once the host has them. */
static int spillRead(Reading *out, int max) {
  if (!spillReady || max <= 0) return 0;
  File f = LittleFS.open(SPILL_PATH, "r");
  if (!f) return 0;
  uint32_t off = spillOffset();
  if (off >= f.size()) { f.close(); return 0; }
  f.seek(off);
  int got = 0;
  while (got < max && f.available() >= (int)sizeof(Reading)) {
    if (f.read((uint8_t *)&out[got], sizeof(Reading)) != sizeof(Reading)) break;
    got++;
  }
  f.close();
  return got;
}

/* The host confirmed `n`: move the read offset past them, and tidy up when done. */
static void spillConsume(int n) {
  if (!spillReady || n <= 0) return;
  uint32_t off = spillOffset() + (uint32_t)n * sizeof(Reading);
  File f = LittleFS.open(SPILL_PATH, "r");
  uint32_t size = f ? f.size() : 0;
  if (f) f.close();
  if (off >= size) {
    spillClear();
    Serial.println("[spill] flash backlog fully delivered and cleared");
  } else {
    spillSetOffset(off);
  }
}

#endif  // PET_SPILL_H
