#!/usr/bin/env python3
"""Inspect and capture from the IDS uEye camera used on the PixelFox microscope."""

import ctypes
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from pyueye import ueye


OUT = Path(__file__).resolve().parent.parent / "photos"


def text(value):
    raw = bytes(value)
    return raw.split(b"\0", 1)[0].decode("ascii", "replace")


def number(value):
    raw = getattr(value, "value", value)
    if isinstance(raw, bytes):
        return int.from_bytes(raw, "little")
    return int(raw)


def check(name, result):
    if result != ueye.IS_SUCCESS:
        raise RuntimeError(f"{name} failed with IDS status {result}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--still-only", action="store_true", help="capture the still without recording video"
    )
    args = parser.parse_args()
    camera = ueye.HIDS(0)
    memory = ueye.c_mem_p()
    memory_id = ueye.int()
    initialized = False
    allocated = False

    try:
        check("is_InitCamera", ueye.is_InitCamera(camera, None))
        initialized = True

        camera_info = ueye.CAMINFO()
        sensor_info = ueye.SENSORINFO()
        check("is_GetCameraInfo", ueye.is_GetCameraInfo(camera, camera_info))
        check("is_GetSensorInfo", ueye.is_GetSensorInfo(camera, sensor_info))
        check("is_ResetToDefault", ueye.is_ResetToDefault(camera))
        check("is_SetDisplayMode", ueye.is_SetDisplayMode(camera, ueye.IS_SET_DM_DIB))

        aoi = ueye.IS_RECT()
        check(
            "is_AOI(GET)",
            ueye.is_AOI(camera, ueye.IS_AOI_IMAGE_GET_AOI, aoi, ctypes.sizeof(aoi)),
        )
        width = int(aoi.s32Width.value)
        height = int(aoi.s32Height.value)

        # BGR8 is directly usable by OpenCV and avoids ambiguity about Bayer order.
        color_mode = ueye.IS_CM_BGR8_PACKED
        bits_per_pixel = 24
        check("is_SetColorMode", ueye.is_SetColorMode(camera, color_mode))

        color_correction_factor = ueye.double(1.0)
        check(
            "is_SetColorCorrection",
            ueye.is_SetColorCorrection(
                camera, ueye.IS_CCOR_ENABLE_HQ_ENHANCED, color_correction_factor
            ),
        )

        pixel_clock_range = (ueye.UINT * 3)()
        check(
            "is_PixelClock(GET_RANGE)",
            ueye.is_PixelClock(
                camera,
                ueye.IS_PIXELCLOCK_CMD_GET_RANGE,
                pixel_clock_range,
                ctypes.sizeof(pixel_clock_range),
            ),
        )
        pixel_clock = ueye.UINT()
        selected_pixel_clock = ueye.UINT(int(pixel_clock_range[1]))
        check(
            "is_PixelClock(SET)",
            ueye.is_PixelClock(
                camera,
                ueye.IS_PIXELCLOCK_CMD_SET,
                selected_pixel_clock,
                ctypes.sizeof(selected_pixel_clock),
            ),
        )
        check(
            "is_PixelClock(GET)",
            ueye.is_PixelClock(
                camera,
                ueye.IS_PIXELCLOCK_CMD_GET,
                pixel_clock,
                ctypes.sizeof(pixel_clock),
            ),
        )

        min_frame_time = ueye.double()
        max_frame_time = ueye.double()
        frame_time_step = ueye.double()
        check(
            "is_GetFrameTimeRange",
            ueye.is_GetFrameTimeRange(
                camera, min_frame_time, max_frame_time, frame_time_step
            ),
        )
        requested_fps = ueye.double(
            1.0 / float(max_frame_time.value) if args.still_only else 15.0
        )
        actual_fps = ueye.double()
        check(
            "is_SetFrameRate",
            ueye.is_SetFrameRate(camera, requested_fps, actual_fps),
        )

        exposure_range = (ueye.double * 3)()
        check(
            "is_Exposure(GET_RANGE)",
            ueye.is_Exposure(
                camera,
                ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE,
                exposure_range,
                ctypes.sizeof(exposure_range),
            ),
        )

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

        # Photo-oriented automatic control: prefer exposure over noisy gain, run
        # high-speed white balance once, and allow enough frames to converge.
        enabled = ueye.double(1.0)
        dummy = ueye.double(0.0)
        # Put a diffuse white/neutral reference near 75% of full scale.  This
        # leaves highlight headroom while making good use of the sensor range.
        auto_reference = ueye.double(190.0)
        auto_gain_max = ueye.double(15.0)
        auto_wb_speed = ueye.double(100.0)
        auto_wb_skipframes = ueye.double(0.0)
        check(
            "is_SetAutoParameter(reference)",
            ueye.is_SetAutoParameter(
                camera, ueye.IS_SET_AUTO_REFERENCE, auto_reference, dummy
            ),
        )
        check(
            "is_SetAutoParameter(gain max)",
            ueye.is_SetAutoParameter(
                camera, ueye.IS_SET_AUTO_GAIN_MAX, auto_gain_max, dummy
            ),
        )
        check(
            "is_SetAutoParameter(WB speed)",
            ueye.is_SetAutoParameter(
                camera, ueye.IS_SET_AUTO_WB_SPEED, auto_wb_speed, dummy
            ),
        )
        check(
            "is_SetAutoParameter(WB skipframes)",
            ueye.is_SetAutoParameter(
                camera, ueye.IS_SET_AUTO_WB_SKIPFRAMES, auto_wb_skipframes, dummy
            ),
        )
        check(
            "is_SetAutoParameter(WB once)",
            ueye.is_SetAutoParameter(
                camera, ueye.IS_SET_AUTO_WB_ONCE, enabled, dummy
            ),
        )
        auto_gain_status = int(
            ueye.is_SetAutoParameter(camera, ueye.IS_SET_ENABLE_AUTO_GAIN, enabled, dummy)
        )
        auto_shutter_status = int(
            ueye.is_SetAutoParameter(
                camera, ueye.IS_SET_ENABLE_AUTO_SHUTTER, enabled, dummy
            )
        )
        auto_white_balance_status = int(
            ueye.is_SetAutoParameter(
                camera, ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE, enabled, dummy
            )
        )
        time.sleep(max(8.0, 12.0 / float(actual_fps.value)))

        def frame():
            raw = ueye.get_data(
                memory,
                width,
                height,
                bits_per_pixel,
                int(pitch.value),
                copy=True,
            )
            return np.reshape(raw, (height, width, 3))

        still = frame()
        still_path = OUT / "pixelfox_ids_still.png"
        if not cv2.imwrite(str(still_path), still):
            raise RuntimeError("OpenCV failed to write the still image")

        video_path = OUT / "pixelfox_ids_video.avi"
        frame_count = 0
        if not args.still_only:
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                float(actual_fps.value),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("OpenCV failed to open the MJPEG video writer")

            capture_seconds = 5.0
            frame_count = max(1, round(float(actual_fps.value) * capture_seconds))
            deadline = time.monotonic()
            for index in range(frame_count):
                deadline += 1.0 / float(actual_fps.value)
                writer.write(frame())
                time.sleep(max(0.0, deadline - time.monotonic()))
            writer.release()

        current_exposure = ueye.double()
        check(
            "is_Exposure(GET)",
            ueye.is_Exposure(
                camera,
                ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
                current_exposure,
                ctypes.sizeof(current_exposure),
            ),
        )

        report = {
            "model": text(sensor_info.strSensorName),
            "serial": text(camera_info.SerNo),
            "camera_id": text(camera_info.ID),
            "sensor_id": number(sensor_info.SensorID),
            "sensor_color_mode": number(sensor_info.nColorMode),
            "sensor_max_width": number(sensor_info.nMaxWidth),
            "sensor_max_height": number(sensor_info.nMaxHeight),
            "capture_width": width,
            "capture_height": height,
            "capture_pixel_format": "BGR8 packed (24-bit)",
            "pixel_clock_mhz": int(pixel_clock.value),
            "pixel_clock_range_mhz": [int(v) for v in pixel_clock_range],
            "fps_range": [
                1.0 / float(max_frame_time.value),
                1.0 / float(min_frame_time.value),
            ],
            "requested_fps": float(requested_fps.value),
            "actual_fps": float(actual_fps.value),
            "exposure_range_ms": [float(v) for v in exposure_range],
            "current_exposure_ms": float(current_exposure.value),
            "auto_gain_status": auto_gain_status,
            "auto_shutter_status": auto_shutter_status,
            "auto_white_balance_status": auto_white_balance_status,
            "master_gain_percent": int(
                ueye.is_SetHardwareGain(
                    camera,
                    ueye.IS_GET_MASTER_GAIN,
                    ueye.IS_IGNORE_PARAMETER,
                    ueye.IS_IGNORE_PARAMETER,
                    ueye.IS_IGNORE_PARAMETER,
                )
            ),
            "red_gain_factor": int(
                ueye.is_SetHWGainFactor(camera, ueye.IS_GET_RED_GAIN_FACTOR, 0)
            ),
            "green_gain_factor": int(
                ueye.is_SetHWGainFactor(camera, ueye.IS_GET_GREEN_GAIN_FACTOR, 0)
            ),
            "blue_gain_factor": int(
                ueye.is_SetHWGainFactor(camera, ueye.IS_GET_BLUE_GAIN_FACTOR, 0)
            ),
            "mean_bgr": [float(v) for v in still.mean(axis=(0, 1))],
            "clipped_black_percent": float(np.mean(np.max(still, axis=2) <= 2) * 100.0),
            "clipped_white_percent": float(np.mean(np.min(still, axis=2) >= 253) * 100.0),
            "focus_score_laplacian_variance": float(
                cv2.Laplacian(cv2.cvtColor(still, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            ),
            "still": str(still_path),
            "video": None if args.still_only else str(video_path),
            "video_frames": frame_count,
        }
        (OUT / "pixelfox_ids_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))

    finally:
        if initialized:
            ueye.is_StopLiveVideo(camera, ueye.IS_FORCE_VIDEO_STOP)
        if allocated:
            ueye.is_FreeImageMem(camera, memory, memory_id)
        if initialized:
            ueye.is_ExitCamera(camera)


if __name__ == "__main__":
    main()
