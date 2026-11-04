# Getting the library onto the deck: three options

Design note, 2026-11-04. Nothing here is built. Facts about the deck come from the
Pajoniiir docs (`docs/SHARED_P4_BOARDS.md`, `docs/HARDWARE_WIRING.md`,
`p4-core/components/wifi_link`); speeds marked *estimate* must be measured on a deck.

## What the hardware does today

- **The library medium is a USB drive**, not an SD card, on every board: a P4 USB **host**
  port (JC4880: USB0 high-speed; M3: USB3 high-speed; JC1060: USB1). The other P4 USB
  port is also a **host**, for the DJ controller (FLX4/Hercules). No P4 USB controller is
  free to act as a device towards a PC.
- **Wi-Fi** is an ESP32-C6 on SDIO (esp_hosted). JC4880: available, default off, the deck's
  own access point (AP) is the normal mode and joining another network (STA) is "a visit,
  not a state to live in". M3: on by default, WPA2, AP+STA at the same time. JC1060: none.
  On M3 the C6 uses the P4's SDMMC controller, so an internal SD card would share it.
- **The deck's HTTP API has no authentication.** `api_request_allowed()` in web_server.c
  checks the Host header (against DNS rebinding) and, for changes, an `X-DDJ-Control: 1`
  header (so web pages can't trigger actions). Both stop browsers, not programs: any app
  on the network can send them. Firmware updates are safe only because bundles are signed
  and the deck verifies them. There is no file upload route.
- **Storage access is already arbitrated** (`media_io_gate`, `sd_io_gate`, the library
  load gate), because playback and library reads share the drive.

What one sync moves (format-3 stems): a track is 8-15 MB (MP3) or 40-60 MB (WAV/FLAC);
its stems are about 20 MB per minute (a 5-minute track: ~100 MB); the analysis files and
`export.pdb` are a few hundred KB.

## (a) The deck as a USB drive over USB-C while idle

The PC sees the deck's library drive as a normal disk; Pocket Prep's existing export
writes to it. The player unmounts the drive first and remounts and reloads after.

- **Speed:** USB 2.0 high-speed mass storage, 15-30 MB/s *estimate* (the drive's write
  speed decides). A track with stems in 3-8 s. Full libraries in minutes. Fastest option.
- **Firmware work, large, plus hardware:**
  - One P4 USB controller must switch from host to **device** role: the TinyUSB MSC device
    stack, role switching, and an "Idle / connected to computer" state that stops
    playback, unmounts FatFs, exposes the drive, then on disconnect remounts and reloads
    the library.
  - The library drive sits on a host port. The P4 can't be a USB device on that port and
    the drive's host at the same time, so the PC would need **its own connector** and
    something that relays sectors (or a USB mux). The board also drives VBUS to the
    drive, while a PC drives VBUS too: the wiring doc forbids that kind of backfeed. This
    means a hardware revision with a dedicated device-role USB-C port and VBUS isolation.
  - With an internal SD card instead of a USB drive (a future board), the device side
    would expose SDMMC sectors directly; simpler, but SDMMC is shared with Wi-Fi on M3.
- **Risks:** filesystem corruption if the cable is pulled while the PC still caches
  writes (needs "eject first" and a FAT dirty-bit check on remount); the controller port
  and the drive must never be powered from two sides; can only happen when the deck isn't
  playing.
- **Authentication:** physical access to the cable is the trust boundary, as with any USB
  stick. No network exposure.
- **Note:** today the same result is had by moving the USB drive from the deck to the PC,
  which Pocket Prep already supports. (a) only pays off once the medium is internal.

## (b) Sync over the home Wi-Fi

The deck joins the home network as a client (STA); Pocket Prep finds it and syncs in the
background whenever a playlist changes. The only option that can be truly automatic.

- **Speed:** limited by the C6 over SDIO: 10-30 Mbit/s TCP *estimate* (1-3.5 MB/s), less
  through a busy router. A track without stems in 3-15 s; with stems 30-90 s. Fine for
  "add a few tracks", slow for a first full library (hours).
- **Firmware work, medium:**
  - Keep STA up as a normal state (M3 does AP+STA already; JC4880 treats STA as temporary).
  - Discovery: mDNS (`_pocketdeck._tcp`), so the app finds the deck without an IP address.
  - An **upload API**: `PUT /api/library/files/<path>` in chunks with resume, written to a
    temp file and renamed only when complete and checksummed; paths limited to
    `Contents/`, `PIONEER/`, `POCKETDECK/` (no `..`), size and free-space checks.
  - An **atomic commit**: upload everything, then `export.pdb` and `stems.json` last as
    temp files, then one "commit" call that renames them and triggers the library reload
    gate. A sync cut off half way leaves the old library intact.
  - **Throttling while playing:** uploads go through the existing IO gates at low
    priority and pause during playback (or below a buffer threshold), so a sync never
    causes a dropout.
- **Risks:** dropouts if write arbitration is wrong (must be tested under playback);
  partial sync after power loss (solved by the commit step); home-network credentials
  stored on the deck (NVS encryption); the deck becomes reachable by every device on the
  home network.
- **Authentication (required):** pairing. The deck shows a 6-digit code; the app sends
  it once and gets a per-app random token (stored in the app, revocable on the deck).
  Every upload carries the token; sessions use HTTPS with the deck's self-signed
  certificate pinned at pairing (mbedTLS on the P4), or at least an HMAC of each request
  with the token so it can't be replayed. The existing `X-DDJ-Control` header is not
  enough.

## (c) Sync over the deck's own access point

The computer joins the deck's Wi-Fi network (what the OTA update does today) and Pocket
Prep pushes the changes.

- **Speed:** like (b) or slightly better (no router in between): 10-30 Mbit/s *estimate*.
- **Firmware work, small-medium:** the AP exists already; needs the same upload API,
  commit step, throttling and authentication as (b), but no STA mode or discovery (the
  deck is always at its AP address).
- **Risks:** the computer loses its internet connection while joined; operating systems
  tend to hop back to known networks mid-sync; so it can't run unattended in the
  background. It's a "Sync now" button near the deck, not background sync. The AP
  password is often shared (e.g. at gigs), so it can't count as authentication.
- **Authentication (required):** the same pairing token as (b). WPA2 on the AP protects
  the radio link only; it says nothing about who may write files.

## Comparison

| | (a) USB drive | (b) Home Wi-Fi | (c) Deck AP |
|---|---|---|---|
| Speed (track + stems) | 3-8 s | 30-90 s | 30-90 s |
| Background, automatic | no (cable, idle only) | **yes** | no (network switch) |
| Firmware work | large + hardware revision | medium | small-medium |
| Boards | new board needed | M3 now, JC4880 after STA rework | JC4880, M3 |
| Main risk | corruption on unplug, VBUS | dropouts while playing, LAN exposure | network hopping |
| Upload auth needed | no (physical) | yes: pairing token + TLS/HMAC | yes: pairing token |

## Suggested order

1. Keep the USB-drive move for big loads (works today, fastest).
2. Build the shared pieces once: upload API with temp-file + commit, IO throttling,
   pairing/token auth. These serve both (b) and (c).
3. Ship (c) first as "Sync now" (no STA work), then (b) for background sync on M3.
4. Revisit (a) only if a board revision moves the library to internal storage.
5. Before committing to numbers, measure the C6 link on a deck (iperf, and a 100 MB
   upload while a track plays).
