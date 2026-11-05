# Vendored ClubCraft (MIT OR Apache-2.0)

From https://github.com/beneaze/ClubCraft at commit d654ccd8710bf132d4587479338290d51632bcfc,
only the crates Pocket Prep uses: core, audio, analysis, stems. Licences: LICENSE-MIT,
LICENSE-APACHE, NOTICE, ATTRIBUTION.md (unchanged). Wired in with `[patch]` in ../../Cargo.toml.

Local changes (keep this list complete; offer them upstream):

1. crates/stems/src/backend/onnx.rs: DirectML session sets `ep.dml.disable_graph_fusion=1`.
   ONNX Runtime's DirectML graph fusion miscomputes htdemucs (-82 dB SDR vs CPU on an
   RTX 3080 Ti, ORT 1.20 and 1.24); without fusion it matches the CPU (72/91 dB) at the
   same speed (~0.087 s per 7.8 s segment vs 0.77 s on the CPU).
2. Removed crates/stems/{tests,examples,tools} (need reference files not in git).
