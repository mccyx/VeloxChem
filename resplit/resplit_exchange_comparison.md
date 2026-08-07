# Supervisor Resplit Exchange Comparison

## Inputs

- Supervisor source: `re-split-EriExchange.cu`
- Current source: `src/gpu/EriExchange.cu`
- Current header: `src/gpu/EriExchange.hpp`
- Current launch/validation code: `src/gpu/FockDriverGPU.cu`

The supervisor file is a generated CUDA source body. It starts directly with a
kernel definition and does not contain the license header, includes, or
`namespace gpu` wrapper used by the tracked `EriExchange.cu`. It should therefore
be treated as generator output to integrate, not copied over the tracked source
without adding the surrounding file structure.

## Kernel inventory

| Inventory | Current | Supervisor resplit |
|---|---:|---:|
| Original FP64 kernel definitions | 91 | 96 |
| MP FP64 kernel definitions | 91 | 0 |
| MP FP32 kernel definitions | 91 | 0 |
| Total source lines | 80,563 | 27,622 |

The supervisor version has five more original FP64 kernels because it uses a
different split distribution.

## Changed split families

| Logical combination | Current kernels | Supervisor kernels | Change |
|---|---:|---:|---:|
| `DDDD` | 19 (`0..18`) | 26 (`0..25`) | +7 |
| `DDDP` | 7 (`0..6`) | 6 (`0..5`) | -1 |
| `DDDS` | 1 unsplit | 2 (`0..1`) | +1 |
| `DPDD` | 7 (`0..6`) | 5 (`0..4`) | -2 |
| `DSDD` | 1 unsplit | 2 (`0..1`) | +1 |
| `PDDD` | 8 (`0..7`) | 5 (`0..4`) | -3 |
| `PPDD` | 1 unsplit | 2 (`0..1`) | +1 |
| `SDDD` | 1 unsplit | 2 (`0..1`) | +1 |

All other logical combinations retain the same kernel names and split counts.

## Symbols added by the supervisor version

```text
computeExchangeFockDDDD19
computeExchangeFockDDDD20
computeExchangeFockDDDD21
computeExchangeFockDDDD22
computeExchangeFockDDDD23
computeExchangeFockDDDD24
computeExchangeFockDDDD25
computeExchangeFockDDDS0
computeExchangeFockDDDS1
computeExchangeFockDSDD0
computeExchangeFockDSDD1
computeExchangeFockPPDD0
computeExchangeFockPPDD1
computeExchangeFockSDDD0
computeExchangeFockSDDD1
```

## Symbols removed by the supervisor version

```text
computeExchangeFockDDDP6
computeExchangeFockDDDS
computeExchangeFockDPDD5
computeExchangeFockDPDD6
computeExchangeFockDSDD
computeExchangeFockPDDD5
computeExchangeFockPDDD6
computeExchangeFockPDDD7
computeExchangeFockPPDD
computeExchangeFockSDDD
```

## Signature compatibility

There are 81 kernel names common to both sources. All 81 have identical
normalized argument lists. The resplit therefore does not introduce a new
argument convention for kernels whose names are retained.

The added split kernels use the same logical-combination argument convention as
their corresponding current kernels. Header declarations and driver launches
must nevertheless be regenerated because the symbol names and counts differ.

## Requirements for MP generation

The generated MP version should contain three variants for every one of the 96
supervisor kernels:

```text
computeExchangeFock<name>       original FP64 reference
computeExchangeFock<name>_FP64  precision-cut FP64 portion
computeExchangeFock<name>_FP32  precision-cut FP32 portion
```

This gives an expected total of 288 exchange kernel definitions and matching
header declarations.

The MP generator must preserve the supervisor split boundaries. It must not
derive the new file by merely renaming the current MP kernels, because the work
assigned to split indices has changed.

## Integration changes required on Dardel

1. Replace the original, MP FP64, and MP FP32 kernel bodies in
   `src/gpu/EriExchange.cu` while retaining the tracked file header, includes,
   and namespace structure.
2. Regenerate `src/gpu/EriExchange.hpp` for all 96 names and all three variants.
3. Update normal FP64 launches in `src/gpu/FockDriverGPU.cu` for the eight
   changed split families.
4. Update validation MP FP64, MP FP32, and reference launches for the same
   families.
5. Keep the existing cut arguments on every `_FP64` and `_FP32` signature.
6. Rebuild and validate all 54 logical combinations before timing.

The current uncommitted timing-boundary correction in `FockDriverGPU.cu` should
be retained when the launch lists are regenerated.
