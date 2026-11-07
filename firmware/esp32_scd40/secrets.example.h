// Copy to secrets.h (git-ignored) and fill in. 2.4 GHz WiFi only.
#pragma once

// ---- WHICH BOARD AM I FLASHING? ------------------------------------------
// One sketch serves every node: change this ONE digit, upload, change it back.
//   1 = desk node    (SCD40 + OLED)
//   2 = bedroom node (SCD40 + PMS5003 particle counter)
#define NODE 1
// --------------------------------------------------------------------------

#define WIFI_SSID    "your-wifi-name"
#define WIFI_PASS    "your-wifi-password"
#define INGEST_URL   "http://192.168.1.50:8765/ingest"   // LAN IP of the machine running `python main.py collect`
#define INGEST_TOKEN "same-value-as-INGEST_TOKEN-in-.env"

// Per-node identity. ROOM_NAME becomes the data location on the host, so the two
// sensors never mix into one series. Everything above is shared by both boards.
#if NODE == 1
  #define DEVICE_NAME "esp32-1"
  #define ROOM_NAME   "desk"
  #define HAS_OLED    1
  #define HAS_PMS     0
  #define HAS_TOUCH 1        // conductive panel on GPIO27, same node as the display
  // Uncomment to match the panel if the display stays blank or looks shifted:
  // #define OLED_SH1106 1    // controller is SH1106, not SSD1306
  // #define OLED_H      32   // panel is 128x32, not 128x64
#elif NODE == 2
  #define DEVICE_NAME "esp32-2"
  #define ROOM_NAME   "bedroom"
  #define HAS_OLED    0
  #define HAS_PMS     1    // PMS5003 particle counter on UART2 (GPIO25 RX, GPIO26 TX)
  #define HAS_TOUCH 0        // conductive panel on GPIO27, same node as the display
#else
  #error "Set NODE to 1 (desk) or 2 (bedroom)"
#endif
