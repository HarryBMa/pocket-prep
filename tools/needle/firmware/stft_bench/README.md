# STFT/iSTFT benchmark for the RA8P1 (two decks)

Measures what Needle's front and back end cost the Cortex-M85: per deck and per 512-sample hop,
2 windowed 2048-point real FFTs, mask dequantisation and normalisation (4 sources sum to 1),
mask x spectrum for 8 stems/channels, 8 inverse FFTs and overlap-add. Two decks = 20 transforms
per hop, 86.13 hops per second. The NPU part is measured separately by Vela (README one level up).

It prints (and stores in `needle_bench_result`, readable in a debugger) the average and worst
cycles per hop and the share of a 1 GHz M85, and checks the arithmetic: with equal masks the four
stems must add up to the input (error printed; should be ~1e-6).

Decision rule (agreed): spectral Needle stays if the M85 share is about 5 % or less with the
deck's other work running.

## Toolchain: Renesas FSP, command line

1. Install the Arm GNU Toolchain (arm-none-eabi, 13.x or newer) or the Arm Toolchain for Embedded
   (LLVM), CMake and Ninja.
2. Renesas FSP 6.x with the RA Smart Configurator (RASC). New project for R7KA8P1 (the Titan
   Mini's part), bare metal, no RTOS; clocks: CPU0 (M85) at 1 GHz; enable I/D caches and the
   DWT. Add the **Arm CMSIS-DSP** software component (Helium-optimised build).
3. Copy `stft_bench.c` into `src/`, define `NEEDLE_TARGET_M85` and call it from `hal_entry()`:

       void needle_stft_bench(uint32_t hops);
       void hal_entry(void) { needle_stft_bench(1000); while (1) {} }

4. Compiler flags: `-O3 -mcpu=cortex-m85 -mfloat-abi=hard -mfpu=auto -DARM_MATH_MVEF -DARM_MATH_HELIUM`.
   Put the large static buffers in SRAM (default), not TCM, unless the deck firmware will.
5. Output: `printf` via SEGGER RTT or a UART (RASC), or read `needle_bench_result` in the debugger.

Run it twice: alone (best case), and with a load running on the M85 that resembles the deck
(audio decoding, mixing), to see cache and memory contention.

## Host build (correctness only, timings meaningless)

    git clone --depth 1 https://github.com/ARM-software/CMSIS-DSP.git
    D=CMSIS-DSP
    gcc -O2 -std=c11 -Ihost_shim -I$D/Include -I$D/PrivateInclude stft_bench.c \
        $D/Source/TransformFunctions/TransformFunctions.c $D/Source/CommonTables/CommonTables.c \
        $D/Source/BasicMathFunctions/BasicMathFunctions.c $D/Source/ComplexMathFunctions/ComplexMathFunctions.c \
        -lm -o stft_bench && ./stft_bench

`host_shim/cmsis_compiler.h` replaces the Arm-only core header on a PC. Not yet run: building the
downloaded library in Claude's workspace needs your permission, or run it on any PC with gcc.
