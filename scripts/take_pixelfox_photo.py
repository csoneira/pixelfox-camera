#!/usr/bin/env python3
"""Take a full-resolution, calibrated, low-noise PixelFox photograph."""

import argparse
import ctypes
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pyueye import ueye


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROFILE_PATH = ROOT / "config" / "photo_profile.json"


def check(name, status):
    if status != ueye.IS_SUCCESS:
        raise RuntimeError(f"{name} failed with IDS status {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Capture a calibrated 2560x1920 PNG from the PixelFox/IDS camera."
    )
    parser.add_argument("--output", type=Path, help="output PNG path")
    parser.add_argument(
        "--average",
        type=int,
        default=3,
        metavar="N",
        help="average N frames for lower noise (default: 3; use 1 for moving subjects)",
    )
    parser.add_argument(
        "--manual-exposure",
        type=float,
        metavar="MS",
        help="skip highlight metering and use this exposure in milliseconds",
    )
    parser.add_argument(
        "--max-exposure",
        type=float,
        default=300.0,
        metavar="MS",
        help="maximum automatic exposure (default: 300 ms for low-light subjects)",
    )
    args = parser.parse_args()
    if args.average < 1 or args.average > 16:
        parser.error("--average must be between 1 and 16")
    if args.max_exposure < 20.0 or args.max_exposure > 300.0:
        parser.error("--max-exposure must be between 20 and 300 ms")

    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    output = args.output or ROOT / "photos" / f"pixelfox_photo_{datetime.now():%Y%m%d_%H%M%S}.png"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    camera = ueye.HIDS(0)
    memory = ueye.c_mem_p()
    memory_id = ueye.int()
    initialized = False
    allocated = False

    try:
        check("is_InitCamera", ueye.is_InitCamera(camera, None))
        initialized = True

        camera_info = ueye.CAMINFO()
        check("is_GetCameraInfo", ueye.is_GetCameraInfo(camera, camera_info))
        serial = bytes(camera_info.SerNo).split(b"\0", 1)[0].decode("ascii")
        if serial != profile["camera_serial"]:
            raise RuntimeError(
                f"profile is for camera {profile['camera_serial']}, but found {serial}"
            )

        check("is_ResetToDefault", ueye.is_ResetToDefault(camera))
        check("is_SetDisplayMode", ueye.is_SetDisplayMode(camera, ueye.IS_SET_DM_DIB))
        check("is_SetColorMode", ueye.is_SetColorMode(camera, ueye.IS_CM_BGR8_PACKED))

        # A fixed photographic profile must not be silently changed by one of
        # the camera's automatic loops after the calibrated values are applied.
        disabled = ueye.double(0.0)
        dummy = ueye.double(0.0)
        for label, command in (
            ("gain", ueye.IS_SET_ENABLE_AUTO_GAIN),
            ("shutter", ueye.IS_SET_ENABLE_AUTO_SHUTTER),
            ("white balance", ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE),
        ):
            check(
                f"disable auto {label}",
                ueye.is_SetAutoParameter(camera, command, disabled, dummy),
            )

        correction = ueye.double(profile["color_correction_factor"])
        check(
            "is_SetColorCorrection",
            ueye.is_SetColorCorrection(
                camera, ueye.IS_CCOR_ENABLE_HQ_ENHANCED, correction
            ),
        )

        clock = ueye.UINT(profile["pixel_clock_mhz"])
        check(
            "is_PixelClock(SET)",
            ueye.is_PixelClock(
                camera, ueye.IS_PIXELCLOCK_CMD_SET, clock, ctypes.sizeof(clock)
            ),
        )
        requested_fps = ueye.double(profile["frame_rate_fps"])
        actual_fps = ueye.double()
        check("is_SetFrameRate", ueye.is_SetFrameRate(camera, requested_fps, actual_fps))

        exposure = ueye.double(
            args.manual_exposure
            if args.manual_exposure is not None
            else min(profile["exposure_ms"], args.max_exposure)
        )
        check(
            "is_Exposure(SET)",
            ueye.is_Exposure(
                camera,
                ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
                exposure,
                ctypes.sizeof(exposure),
            ),
        )
        current_gain = profile["master_gain_percent"]
        check(
            "is_SetHardwareGain",
            ueye.is_SetHardwareGain(
                camera,
                current_gain,
                ueye.IS_IGNORE_PARAMETER,
                ueye.IS_IGNORE_PARAMETER,
                ueye.IS_IGNORE_PARAMETER,
            ),
        )
        for label, command in (
            ("red", ueye.IS_SET_RED_GAIN_FACTOR),
            ("green", ueye.IS_SET_GREEN_GAIN_FACTOR),
            ("blue", ueye.IS_SET_BLUE_GAIN_FACTOR),
        ):
            applied = ueye.is_SetHWGainFactor(
                camera, command, profile[f"{label}_gain_factor"]
            )
            if applied < 0:
                raise RuntimeError(
                    f"is_SetHWGainFactor({label}) failed with IDS status {applied}"
                )

        width = profile["width"]
        height = profile["height"]
        bits_per_pixel = 24
        check(
            "is_AllocImageMem",
            ueye.is_AllocImageMem(
                camera, width, height, bits_per_pixel, memory, memory_id
            ),
        )
        allocated = True
        check("is_SetImageMem", ueye.is_SetImageMem(camera, memory, memory_id))
        pitch = ueye.int()
        check("is_GetImageMemPitch", ueye.is_GetImageMemPitch(camera, pitch))
        check("is_CaptureVideo", ueye.is_CaptureVideo(camera, ueye.IS_DONT_WAIT))

        def read_frame():
            raw = ueye.get_data(
                memory, width, height, bits_per_pixel, int(pitch.value), copy=True
            )
            return np.reshape(raw, (height, width, 3))

        # Discard startup frames, then meter from the brightest 0.5% of the
        # image. This protects white microscope subjects against a black field,
        # where ordinary whole-frame auto exposure tends to destroy detail.
        interval = 1.0 / float(actual_fps.value)
        time.sleep(max(1.0, 3.0 * interval))
        highlight = None
        if args.manual_exposure is None:
            target_highlight = 220.0
            # First use exposure alone, which has essentially no noise penalty.
            for _ in range(5):
                sample = read_frame()
                gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                highlight = float(np.percentile(gray, 99.5))
                if highlight <= 1.0:
                    break
                ratio = float(np.clip(target_highlight / highlight, 0.5, 2.0))
                proposed = float(
                    np.clip(exposure.value * ratio, 20.0, args.max_exposure)
                )
                # Integrate over complete 50 Hz lighting cycles to reject flicker.
                proposed = round(proposed / 20.0) * 20.0
                if abs(proposed - exposure.value) < 10.0:
                    break
                exposure = ueye.double(proposed)
                check(
                    "is_Exposure(SET metered)",
                    ueye.is_Exposure(
                        camera,
                        ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
                        exposure,
                        ctypes.sizeof(exposure),
                    ),
                )
                time.sleep(2.0 * interval)

            # If maximum clean exposure is insufficient, binary-search only the
            # bounded 0-60% gain range and select the closest highlight result.
            if (
                highlight is not None
                and highlight < target_highlight * 0.95
                and exposure.value > args.max_exposure - 10.0
            ):
                low_gain, high_gain = 0, 60
                best_gain, best_error = 0, abs(highlight - target_highlight)
                for _ in range(6):
                    trial_gain = (low_gain + high_gain) // 2
                    check(
                        "is_SetHardwareGain(search)",
                        ueye.is_SetHardwareGain(
                            camera,
                            trial_gain,
                            ueye.IS_IGNORE_PARAMETER,
                            ueye.IS_IGNORE_PARAMETER,
                            ueye.IS_IGNORE_PARAMETER,
                        ),
                    )
                    time.sleep(2.0 * interval)
                    sample = read_frame()
                    trial_highlight = float(
                        np.percentile(cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY), 99.5)
                    )
                    trial_error = abs(trial_highlight - target_highlight)
                    if trial_error < best_error:
                        best_gain, best_error = trial_gain, trial_error
                    if trial_highlight < target_highlight:
                        low_gain = trial_gain + 1
                    else:
                        high_gain = trial_gain - 1
                current_gain = best_gain
                check(
                    "is_SetHardwareGain(best)",
                    ueye.is_SetHardwareGain(
                        camera,
                        current_gain,
                        ueye.IS_IGNORE_PARAMETER,
                        ueye.IS_IGNORE_PARAMETER,
                        ueye.IS_IGNORE_PARAMETER,
                    ),
                )
                time.sleep(2.0 * interval)
                sample = read_frame()
                highlight = float(
                    np.percentile(cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY), 99.5)
                )

        accumulator = np.zeros((height, width, 3), dtype=np.float64)
        for _ in range(args.average):
            time.sleep(interval)
            accumulator += read_frame()
        image = np.clip(np.rint(accumulator / args.average), 0, 255).astype(np.uint8)

        if not cv2.imwrite(str(output), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"failed to save {output}")

        means = image.mean(axis=(0, 1))
        print(f"Saved: {output}")
        print(f"Resolution: {width}x{height}")
        print(
            f"Exposure: {float(exposure.value):.3f} ms; gain: {current_gain}%; "
            f"averaged frames: {args.average}"
        )
        if highlight is not None:
            print(f"Highlight-meter reading: {highlight:.1f}/255 (target 220)")
            if highlight < 154.0:
                print(
                    "WARNING: insufficient illumination for the selected exposure limit; "
                    "increase the microscope light or use --max-exposure with a larger value."
                )
        print(f"Mean B/G/R: {means[0]:.1f} / {means[1]:.1f} / {means[2]:.1f}")
        print(
            "Clipped white: "
            f"{np.mean(np.min(image, axis=2) >= 253) * 100.0:.3f}%"
        )

    finally:
        if initialized:
            ueye.is_StopLiveVideo(camera, ueye.IS_FORCE_VIDEO_STOP)
        if allocated:
            ueye.is_FreeImageMem(camera, memory, memory_id)
        if initialized:
            ueye.is_ExitCamera(camera)


if __name__ == "__main__":
    main()
