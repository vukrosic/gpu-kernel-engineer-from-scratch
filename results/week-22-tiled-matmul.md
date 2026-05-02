# Week 22

Status: writing template

## What To Capture

- one tile diagram
- one note about reuse
- one comparison between tiled and naive matmul
- one tradeoff you noticed

## Tile Sketch

```text
A tile:
[   ][   ]
[   ][   ]

B tile:
[   ][   ]
[   ][   ]

Output tile:
[   ][   ]
[   ][   ]
```

## What Was Built

Describe the tiled block or kernel you wrote or studied. Name the tile sizes
and the area of the output they cover.

## Correctness Check

Record why the tile accumulation should still match the naive result. If you
compared against the reference, say what matched.

## Benchmark Or Observation

If you measured anything, note whether the tile changed reuse or data movement.
If you did not measure, write the comparison you would make next.

## Lesson Learned

Summarize why tiling changes the shape of the work.

## Limitation Or Next Step

Write one sentence about what tile size tradeoff you want to explore next.

## Write-Back Prompts

1. What is a tile?
2. What is reused inside a tile?
3. What stays the same as the naive version?
