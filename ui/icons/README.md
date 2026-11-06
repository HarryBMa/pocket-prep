# Icons

The `<symbol>` set at the top of `ui/index.html` is Lucide geometry (ISC, `LICENSE-Lucide-ISC.txt`;
parts derived from Feather, MIT, `LICENSE-Feather-MIT.txt`). Where the DECK001 has the same control,
the path data is copied from the firmware's `assets/icons/`, so a control looks the same on the deck
and in DKPRP. The rest come from `lucide-static` 0.460.0. All are 24×24, stroke 2, drawn in the text
colour.

| symbol | source | deck meaning | DKPRP use |
|---|---|---|---|
| `i-library` | deck `list-music` | BROWSE | Library tab, artwork placeholder |
| `i-export` | deck `usb` | USB library | Export tab, dock |
| `i-deck` | deck `audio-lines` | LIVE | Deck tab, dock device panel |
| `i-settings` | deck `settings` | SETTINGS | Settings tab |
| `i-play`, `i-pause` | deck `play`, `pause` | PLAY / STOP | transport |
| `i-lock` | deck `key` | KEYLOCK | preview key lock |
| `i-prev`, `i-next` | Lucide `skip-back`, `skip-forward` | – | previous / next track |
| `i-volume` | Lucide `volume-2` | – | preview volume |
| `i-jobs` | Lucide `list-checks` | – | jobs |
| `i-list`, `i-folder` | Lucide `list`, `folder` | – | playlist, folder |
| `i-search`, `i-x`, `i-check` | Lucide | – | search, close, checklist |
| `i-sun`, `i-moon` | Lucide | – | theme switch (the deck is dark only) |

Deck icons not used in DKPRP yet: `repeat-2` (LOOP), `arrow-left-right` (BEAT JUMP), `magnet`
(QUANTIZE), `grid-3x3` (grid editor), `plane` (AUTO PILOT), `siren` (live warning), `square`,
`plus`, `minus`, `repeat`, `folder-up`, `chevron-*`.
