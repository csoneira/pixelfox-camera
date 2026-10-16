# PixelFox camera capture

A small Linux acquisition application for the Vision SX80 PixelFox microscope
camera, which contains an IDS Imaging UI-3580LE-C (`1409:3580`). The camera uses
the IDS uEye userspace driver rather than V4L2/`uvcvideo`.

## Start the camera

Connect the PixelFox camera, then run:

```bash
cd ~/pixelfox_test
./pixelfox
```

The terminal remains open for status and prints the path of every saved photo.
A separate window provides the live image and controls.

## Workflow

1. Place a neutral white reference in focus so it fills most of the image.
2. Click **AUTO TUNE ON WHITE** or press **F5**. Keep it still for eight seconds.
3. Replace the reference with the specimen.
4. Enter its identifier in **Piece label**.
5. Press **Enter** or click **TAKE PHOTO**. Enter works while the label field has
   focus, and the cursor remains there so the next piece label can be entered
   immediately. Spaces remain available inside labels.

Photos are lossless, native-resolution PNG files in `photos/`, named like:

```text
PIECE-17_20261016_143522_381.png
```

The label is filename-sanitized. Empty labels become `unlabelled`. Exposure and
gain may also be adjusted manually with the sliders. Escape closes the program
and releases the camera.

Auto tune freezes and saves exposure, gain, and RGB balance into the local
`config/photo_profile.json`; the next launch reuses those settings. This file and
all photographs are intentionally excluded from Git.

## Layout

```text
pixelfox_test/
├── pixelfox                 # terminal launcher
├── scripts/                 # GUI and diagnostic/capture programs
├── config/                  # local calibration and public example
├── photos/                  # photographs (not committed)
└── diagnostics/             # machine reports (not committed)
```

## Requirements

- Ubuntu/Linux with IDS uEye 4.96 installed
- `ueyeusbdrc.service` active
- Python 3 with `pyueye`, OpenCV, NumPy, Pillow, and Tk

On the configured acquisition machine these requirements are already installed.
The supplied `take_pixelfox_photo.py` remains available for non-interactive
capture.

## Optional paths

```bash
./pixelfox --photos /path/to/output --profile /path/to/profile.json
```
