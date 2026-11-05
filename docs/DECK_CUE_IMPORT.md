# Plan: importing cues edited on the deck back into Pocket Prep

Status: plan only, nothing built. Facts are from Pajoniiir `p4-core` (components
`hot_cue_store`, `media_identity`, `library`, `web_server`) and `docs/M3_CUE_MIGRATION.md`.

## What the deck stores, and where

**Only hot cues can be edited on the deck.** Memory cues are read-only there: the deck
shows them from the ANLZ file (`anlz_metadata_t.memory_cues`) and has no local store for
them. So "importing deck cues" means hot cue pads A-H (singles and loops).

**Edits are not written to the USB drive.** They live in the deck's internal flash
(ESP-IDF NVS), as *overrides* on top of the drive's rekordbox cues ("source"):

| | |
|---|---|
| NVS namespace | `hotcue_v3` (older `hotcue_v2` is still read as a fallback) |
| Key | `h` + the first 7 bytes of the media identity in hex (15 characters) |
| Record (v3, 148 bytes, little-endian) | magic `0x33435648` ("HVC3") u32, version 3 u16, payload length u16, **full 32-byte media identity**, `valid_mask` u32, `override_mask` u32, 8 slots × {`pos_ms` u32, `end_ms` u32, `type` u8 (1 single, 2 loop), 3 bytes pad}, CRC-32 (ISO) of everything before it |
| Meaning | Bit *n* of `override_mask` set: slot *n* replaces the source cue, **including an empty slot** (the user deleted it on the deck). Bit clear: the source (rekordbox) cue shows. `valid_mask` says which override slots hold a cue. |

`hot_cue_store_merge(source, local)` gives what the deck shows; Pocket Prep has to do the
same merge.

**The media identity** (`media_persistent_id_derive`) is
`SHA-256("pajoniiir.hotcue.v2" ‖ export_digest[32] ‖ path_len u16 LE ‖ path ‖ file_size u64 LE ‖ fat_mtime i64 LE)`
where:

- `export_digest` = SHA-256 of `PIONEER/rekordbox/export.pdb` as the deck read it
  (`library.c: hash_export_file`);
- `path` = the track's path on the drive as in the PDB, `file_size` = audio file size;
- `fat_mtime` = the audio file's modification time **as the P4's FAT driver reports it**.
  The PC can't reproduce it reliably (FAT local time, 2 s steps, exFAT time zones), which
  is why the migration tool takes identities from the deck rather than computing them.

**The deck already exposes identities:** `GET /api/cue-migration/catalog?offset=N&generation=G`
returns 8 tracks per page: `track_id`, `path`, `title`, `export_digest`, `persistent_id`,
`file_size`, `mtime`, `status`, plus `generation`, `count` and `next_offset`. It answers
409 while a deck is playing or if the library changed mid-way. It does **not** return the
cue overrides themselves.

## Consequence for exports: re-exporting orphans deck edits

The identity includes the hash of `export.pdb`. Pocket Prep writes a new `export.pdb` on
every export, so after a re-export every override on the deck points at an identity that
no longer exists. The deck then shows the plain rekordbox cues again. (It doesn't delete the
old records, so they also use up NVS space: the partition is 24 KiB, room for roughly
100 tracks' overrides.)

So cues must come **back from the deck before Pocket Prep re-exports**, and the export
writes them into the new ANLZ files as ordinary rekordbox cues.

## Firmware work (small)

1. **Read-only endpoint**, e.g. `GET /api/cue-overrides?offset=N&generation=G`: for each
   catalog track that has a `hotcue_v3`/`v2` record, return `persistent_id`,
   `override_mask`, `valid_mask` and the 8 slots (decoded and CRC-checked by
   `hot_cue_store_load`). Same paging, generation fence and "decks stopped" rule as the
   catalog. No writes.
2. **Optional clean-up after a successful import:** `POST /api/cue-overrides/clear` with a
   list of identities (uses `hot_cue_store_clear`), behind the same authentication as any
   other write (see SYNC_OPTIONS.md: pairing token), so NVS doesn't fill with orphans.
3. Alternative without Wi-Fi (e.g. JC1060): on "eject", the deck writes the same data to
   `POCKETDECK/deck-cues.json` on the drive. More firmware work (writes to the medium),
   but it reaches Pocket Prep with the drive itself.

## Pocket Prep side: reading them back on connect

When the deck is reachable (deck page, or before an export to that drive):

1. **Fetch** `/api/cue-migration/catalog` and the new `/api/cue-overrides`, all pages,
   with one `generation`; on 409 ask the user to stop both decks, and on a generation
   change start again.
2. **Check it's our library:** `export_digest` must equal the SHA-256 of the `export.pdb`
   we wrote (add `pdb_sha256` next to `pdb_blake3` in `POCKETDECK/export.json`, since the
   deck uses SHA-256). A different digest means the deck holds cues for another export:
   offer them only for tracks whose `path` and `file_size` still match, clearly marked.
3. **Match tracks:** catalog `path` equals the PDB file path we wrote for that track, which
   is stored per export (keep a table `exported_tracks(export_id, track_id, card_path,
   file_size)` in the library). Optionally check `persistent_id` by recomputing it from
   the deck's `mtime`.
4. **Merge per slot** (as `hot_cue_store_merge`): override bit clear: keep ours; bit set and
   valid: the deck's cue (`pos_ms`, loop `end_ms` when type 2); bit set and not valid: the
   user deleted that pad on the deck, so delete ours.
5. **Conflicts:** if a slot changed both on the deck and in Pocket Prep since that export
   (we know the exported cues from a snapshot taken at export time), don't guess: show
   both side by side with "Use deck" / "Keep Pocket Prep". Otherwise apply automatically.
6. **Apply** as one undoable change per track (`library::set_cues`, logged with
   `cues_restore`), mark the cues as user cues (`auto=0`) so re-analysis keeps them, then
   toast "Imported N cue changes from the deck. Undo".
7. On the next export they're written to ANLZ (PCOB) as normal hot cues, and optionally
   the deck's overrides are cleared (firmware step 2).

## Risks

- Identity depends on the FAT mtime: copying the audio again (e.g. a fresh export to a new
  drive) changes it, so deck edits must be imported before that, not after.
- Reading while the decks play is refused by design (the catalog needs a stable library).
- Overrides for tracks Pocket Prep no longer has are reported, not silently dropped.
- Authentication: reading cues is low risk, but it should use the same pairing as uploads,
  and clearing overrides must require it.
