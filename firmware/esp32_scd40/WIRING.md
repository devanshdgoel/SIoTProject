# Wiring

Two nodes run the same sketch. `NODE` in `secrets.h` picks which one you are
flashing; everything below follows from that one digit.

## Node 1 — desk (SCD40 + OLED)

| From | Pin | To (ESP32) |
|---|---|---|
| SCD40 | VCC | 3V3 |
| SCD40 | GND | GND |
| SCD40 | SDA | GPIO21 |
| SCD40 | SCL | GPIO22 |
| OLED | VCC | 3V3 |
| OLED | GND | GND |
| OLED | SDA | GPIO21 |
| OLED | SCL | GPIO22 |

The OLED shares the I2C bus with the SCD40. They have different addresses
(0x3C for the display, 0x62 for the sensor), so one pair of wires carries both.

## Node 2 — bedroom (SCD40 + PMS5003)

SCD40 exactly as above: 3V3, GND, SDA to GPIO21, SCL to GPIO22.

The PMS5003 is four more jumper wires, on pins nothing else uses:

| PMS5003 | Pin | To (ESP32) | Why this one |
|---|---|---|---|
| VCC | 1 | **VIN / 5V** | the fan needs 5 V — see the warning below |
| GND | 2 | GND | |
| RXD | 4 | GPIO26 | the ESP32 sends sleep and wake commands here |
| TXD | 5 | GPIO25 | the sensor sends data frames; the ESP32 listens |

Pins 3 (SET), 6 (RESET), 7 and 8 are left unconnected. SET and RESET are pulled
up inside the sensor, so leaving them floating keeps it awake and unreset; sleep
is commanded over the serial line instead, which is two fewer wires to push onto a
breadboard and one less thing to get backwards.

### Three things that will bite

**VCC must be 5 V, not 3V3.** The PMS5003's logic is 3.3 V but its fan is not. On
3V3 the sensor often powers up, answers, and reports plausible-looking numbers from
an airflow that is too weak to be a sample. That is the worst kind of failure
because it does not look like one. Use the VIN / 5V pin while the board is on USB.

**TXD goes to GPIO25, not GPIO26.** Serial crosses over: the sensor's transmit is
the ESP32's receive. Swapped, you get a sensor that never answers and a serial log
full of `only 0 good frame(s)`.

**Why GPIO25 and 26 and not the usual 16 and 17.** GPIO16 and 17 are the default
UART2 pins and would work on a WROOM module, but on a WROVER they are wired to the
PSRAM and taking them breaks the board in ways that look like random crashes. 25
and 26 are free on both, are not strapping pins, and stay clear of the I2C pair.

### What it does once it is wired

The fan is the only moving part in the project and is rated for roughly 8000 hours,
which is under a year if it never stops. So it does not run continuously: the
sensor sleeps, wakes for a 30-second warm-up (Plantower's own figure for a stable
reading), listens for 8 seconds, and sleeps again — one sample every five minutes,
a duty cycle near 12%, and a fan that should outlast the degree.

Each sample is the median of up to 16 frames rather than a single frame, because
the occasional frame arrives mangled and a median does not care.

Every sample records nine numbers, not three:

- **PM1.0, PM2.5, PM10** in µg/m³ — the atmospheric figures.
- **Six cumulative particle counts** per 0.1 L, above 0.3, 0.5, 1.0, 2.5, 5.0 and
  10 µm. These are the part of this sensor most projects throw away, and they are
  the part that says *what* the particles are rather than only how many: cooking,
  traffic coming in through an open window, and dust kicked up off the floor have
  different size distributions even when they give the same PM2.5.

Particle fields are sent only on the minutes the fan actually ran. The other four
minutes in five carry no particle fields at all, because absent and zero are
different claims and zero would mean clean air.

## Checking it

Open the serial monitor at 115200. Within a minute of boot you should see:

```
[pms] PMS5003 on UART2, RX=GPIO25 TX=GPIO26, 9600 baud
[pms] sampling every 300 s with a 30 s warm-up (fan duty ~12%)
[pms] no reading yet - the first one lands after the first warm-up
```

and then, after the first warm-up completes:

```
[pms] PM1 6  PM2.5 9  PM10 11 ug/m3   (median of 16 frames)
[pms]   per 0.1 L: >0.3um 1480  >0.5 430  >1.0 96  >2.5 8  >5 2  >10 1
```

If you get `only 0 good frame(s)`, check TXD is on GPIO25 and VCC is on 5 V.
