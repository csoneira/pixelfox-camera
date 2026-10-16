#!/usr/bin/env python3
"""Live preview and labelled still capture for the PixelFox IDS uEye camera."""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from pyueye import ueye


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = ROOT / "config" / "photo_profile.json"
EXAMPLE_PROFILE = ROOT / "config" / "photo_profile.example.json"
DEFAULT_PHOTOS = ROOT / "photos"


def check(name: str, status: int) -> None:
    if status != ueye.IS_SUCCESS:
        raise RuntimeError(f"{name} failed with IDS status {status}")


def c_text(value) -> str:
    return bytes(value).split(b"\0", 1)[0].decode("ascii", "replace")


def c_int(value) -> int:
    return int(getattr(value, "value", value))


def safe_label(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._-") or "unlabelled"


class PixelFoxCamera:
    def __init__(self, profile_path: Path):
        self.profile_path = profile_path
        if not profile_path.exists():
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(
                EXAMPLE_PROFILE.read_text(encoding="utf-8"), encoding="utf-8"
            )
        self.profile = json.loads(profile_path.read_text(encoding="utf-8"))
        self.camera = ueye.HIDS(0)
        self.memory = ueye.c_mem_p()
        self.memory_id = ueye.int()
        self.initialized = False
        self.allocated = False
        self.width = 0
        self.height = 0
        self.pitch = ueye.int()
        self.fps = 0.0
        self.exposure_ms = 0.0
        self.gain_percent = 0
        self.tuning = False

        try:
            check("is_InitCamera", ueye.is_InitCamera(self.camera, None))
            self.initialized = True

            camera_info = ueye.CAMINFO()
            sensor_info = ueye.SENSORINFO()
            check("is_GetCameraInfo", ueye.is_GetCameraInfo(self.camera, camera_info))
            check("is_GetSensorInfo", ueye.is_GetSensorInfo(self.camera, sensor_info))
            self.serial = c_text(camera_info.SerNo)
            self.model = c_text(sensor_info.strSensorName)
            self.width = c_int(sensor_info.nMaxWidth)
            self.height = c_int(sensor_info.nMaxHeight)

            check("is_ResetToDefault", ueye.is_ResetToDefault(self.camera))
            check("is_SetDisplayMode", ueye.is_SetDisplayMode(self.camera, ueye.IS_SET_DM_DIB))
            check("is_SetColorMode", ueye.is_SetColorMode(self.camera, ueye.IS_CM_BGR8_PACKED))
            correction = ueye.double(float(self.profile.get("color_correction_factor", 1.0)))
            check(
                "is_SetColorCorrection",
                ueye.is_SetColorCorrection(
                    self.camera, ueye.IS_CCOR_ENABLE_HQ_ENHANCED, correction
                ),
            )

            clock_range = (ueye.UINT * 3)()
            check(
                "is_PixelClock(GET_RANGE)",
                ueye.is_PixelClock(
                    self.camera,
                    ueye.IS_PIXELCLOCK_CMD_GET_RANGE,
                    clock_range,
                    ctypes.sizeof(clock_range),
                ),
            )
            requested_clock = min(
                int(self.profile.get("pixel_clock_mhz", clock_range[1])),
                int(clock_range[1]),
            )
            clock = ueye.UINT(requested_clock)
            check(
                "is_PixelClock(SET)",
                ueye.is_PixelClock(
                    self.camera,
                    ueye.IS_PIXELCLOCK_CMD_SET,
                    clock,
                    ctypes.sizeof(clock),
                ),
            )

            min_frame_time = ueye.double()
            max_frame_time = ueye.double()
            frame_step = ueye.double()
            check(
                "is_GetFrameTimeRange",
                ueye.is_GetFrameTimeRange(
                    self.camera, min_frame_time, max_frame_time, frame_step
                ),
            )
            # The slow native rate permits exposures near 300 ms on the current
            # USB 2 connection and keeps preview/photo settings identical.
            requested_fps = ueye.double(1.0 / float(max_frame_time.value))
            actual_fps = ueye.double()
            check(
                "is_SetFrameRate",
                ueye.is_SetFrameRate(self.camera, requested_fps, actual_fps),
            )
            self.fps = float(actual_fps.value)

            self.disable_auto()
            self.set_exposure(float(self.profile.get("exposure_ms", 120.0)))
            self.set_gain(int(self.profile.get("master_gain_percent", 0)))
            for label, command in (
                ("red", ueye.IS_SET_RED_GAIN_FACTOR),
                ("green", ueye.IS_SET_GREEN_GAIN_FACTOR),
                ("blue", ueye.IS_SET_BLUE_GAIN_FACTOR),
            ):
                applied = ueye.is_SetHWGainFactor(
                    self.camera,
                    command,
                    int(self.profile.get(f"{label}_gain_factor", 100)),
                )
                if applied < 0:
                    raise RuntimeError(f"setting {label} gain failed: IDS status {applied}")

            check(
                "is_AllocImageMem",
                ueye.is_AllocImageMem(
                    self.camera,
                    self.width,
                    self.height,
                    24,
                    self.memory,
                    self.memory_id,
                ),
            )
            self.allocated = True
            check("is_SetImageMem", ueye.is_SetImageMem(self.camera, self.memory, self.memory_id))
            check("is_GetImageMemPitch", ueye.is_GetImageMemPitch(self.camera, self.pitch))
            check("is_CaptureVideo", ueye.is_CaptureVideo(self.camera, ueye.IS_DONT_WAIT))
        except Exception:
            self.close()
            raise

    def disable_auto(self) -> None:
        off = ueye.double(0.0)
        dummy = ueye.double(0.0)
        for command in (
            ueye.IS_SET_ENABLE_AUTO_GAIN,
            ueye.IS_SET_ENABLE_AUTO_SHUTTER,
            ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE,
        ):
            check("disable automatic control", ueye.is_SetAutoParameter(self.camera, command, off, dummy))

    def frame(self) -> np.ndarray:
        raw = ueye.get_data(
            self.memory,
            self.width,
            self.height,
            24,
            int(self.pitch.value),
            copy=True,
        )
        return np.reshape(raw, (self.height, self.width, 3))

    def set_exposure(self, milliseconds: float) -> float:
        value = ueye.double(float(milliseconds))
        check(
            "is_Exposure(SET)",
            ueye.is_Exposure(
                self.camera,
                ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
                value,
                ctypes.sizeof(value),
            ),
        )
        self.exposure_ms = float(value.value)
        return self.exposure_ms

    def set_gain(self, percent: int) -> int:
        percent = max(0, min(100, int(percent)))
        check(
            "is_SetHardwareGain",
            ueye.is_SetHardwareGain(
                self.camera,
                percent,
                ueye.IS_IGNORE_PARAMETER,
                ueye.IS_IGNORE_PARAMETER,
                ueye.IS_IGNORE_PARAMETER,
            ),
        )
        self.gain_percent = percent
        return percent

    def start_autotune(self) -> None:
        reference = ueye.double(190.0)
        gain_max = ueye.double(60.0)
        wb_speed = ueye.double(100.0)
        skipframes = ueye.double(0.0)
        on = ueye.double(1.0)
        dummy = ueye.double(0.0)
        for name, command, value in (
            ("reference", ueye.IS_SET_AUTO_REFERENCE, reference),
            ("gain maximum", ueye.IS_SET_AUTO_GAIN_MAX, gain_max),
            ("white-balance speed", ueye.IS_SET_AUTO_WB_SPEED, wb_speed),
            ("white-balance skipframes", ueye.IS_SET_AUTO_WB_SKIPFRAMES, skipframes),
            ("one-shot white balance", ueye.IS_SET_AUTO_WB_ONCE, on),
            ("auto gain", ueye.IS_SET_ENABLE_AUTO_GAIN, on),
            ("auto shutter", ueye.IS_SET_ENABLE_AUTO_SHUTTER, on),
            ("auto white balance", ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE, on),
        ):
            check(name, ueye.is_SetAutoParameter(self.camera, command, value, dummy))
        self.tuning = True

    def finish_autotune(self) -> dict:
        self.disable_auto()
        self.tuning = False
        exposure = ueye.double()
        check(
            "is_Exposure(GET)",
            ueye.is_Exposure(
                self.camera,
                ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
                exposure,
                ctypes.sizeof(exposure),
            ),
        )
        self.exposure_ms = float(exposure.value)
        self.gain_percent = int(
            ueye.is_SetHardwareGain(
                self.camera,
                ueye.IS_GET_MASTER_GAIN,
                ueye.IS_IGNORE_PARAMETER,
                ueye.IS_IGNORE_PARAMETER,
                ueye.IS_IGNORE_PARAMETER,
            )
        )
        gains = {
            "red_gain_factor": int(
                ueye.is_SetHWGainFactor(self.camera, ueye.IS_GET_RED_GAIN_FACTOR, 0)
            ),
            "green_gain_factor": int(
                ueye.is_SetHWGainFactor(self.camera, ueye.IS_GET_GREEN_GAIN_FACTOR, 0)
            ),
            "blue_gain_factor": int(
                ueye.is_SetHWGainFactor(self.camera, ueye.IS_GET_BLUE_GAIN_FACTOR, 0)
            ),
        }
        self.profile.update(
            {
                "camera_serial": self.serial,
                "width": self.width,
                "height": self.height,
                "exposure_ms": self.exposure_ms,
                "master_gain_percent": self.gain_percent,
                "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                **gains,
            }
        )
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(
            json.dumps(self.profile, indent=2) + "\n", encoding="utf-8"
        )
        return gains

    def close(self) -> None:
        if self.initialized:
            ueye.is_StopLiveVideo(self.camera, ueye.IS_FORCE_VIDEO_STOP)
        if self.allocated:
            ueye.is_FreeImageMem(self.camera, self.memory, self.memory_id)
            self.allocated = False
        if self.initialized:
            ueye.is_ExitCamera(self.camera)
            self.initialized = False


class PixelFoxApp:
    def __init__(self, root: tk.Tk, camera: PixelFoxCamera, photo_dir: Path):
        self.root = root
        self.camera = camera
        self.photo_dir = photo_dir
        self.photo_dir.mkdir(parents=True, exist_ok=True)
        self.closed = False
        self.latest_frame: np.ndarray | None = None
        self.preview_image = None

        root.title(f"PixelFox Camera — {camera.model} ({camera.serial})")
        root.geometry("1120x900")
        root.minsize(820, 650)
        root.protocol("WM_DELETE_WINDOW", self.close)

        main = ttk.Frame(root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        self.preview = tk.Label(main, background="black", anchor=tk.CENTER)
        self.preview.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main, padding=(0, 10, 0, 0))
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Piece label:").grid(row=0, column=0, sticky="w")
        self.label_var = tk.StringVar()
        self.label_entry = ttk.Entry(controls, textvariable=self.label_var, width=30)
        self.label_entry.grid(row=0, column=1, padx=(8, 16), sticky="ew")

        self.photo_button = ttk.Button(controls, text="TAKE PHOTO  [Space]", command=self.take_photo)
        self.photo_button.grid(row=0, column=2, padx=4)
        self.tune_button = ttk.Button(
            controls, text="AUTO TUNE ON WHITE  [F5]", command=self.autotune
        )
        self.tune_button.grid(row=0, column=3, padx=4)

        ttk.Label(controls, text="Exposure (ms):").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.exposure_var = tk.DoubleVar(value=camera.exposure_ms)
        self.exposure_scale = ttk.Scale(
            controls,
            from_=1.0,
            to=319.0,
            variable=self.exposure_var,
            command=self.change_exposure,
        )
        self.exposure_scale.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 16), pady=(10, 0))
        self.exposure_value = ttk.Label(controls, width=10)
        self.exposure_value.grid(row=1, column=3, sticky="w", pady=(10, 0))

        ttk.Label(controls, text="Gain (%):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.gain_var = tk.DoubleVar(value=camera.gain_percent)
        self.gain_scale = ttk.Scale(
            controls,
            from_=0,
            to=100,
            variable=self.gain_var,
            command=self.change_gain,
        )
        self.gain_scale.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(8, 16), pady=(6, 0))
        self.gain_value = ttk.Label(controls, width=10)
        self.gain_value.grid(row=2, column=3, sticky="w", pady=(6, 0))

        self.status_var = tk.StringVar(
            value="Ready. Enter a piece label, then click TAKE PHOTO."
        )
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(10, 0)
        )
        controls.columnconfigure(1, weight=1)

        root.bind("<F5>", lambda _event: self.autotune())
        root.bind("<Escape>", lambda _event: self.close())
        root.bind("<space>", self.space_photo)
        self.label_entry.focus_set()
        self.update_setting_labels()
        self.update_preview()

    def update_setting_labels(self) -> None:
        self.exposure_value.configure(text=f"{self.camera.exposure_ms:.1f} ms")
        self.gain_value.configure(text=f"{self.camera.gain_percent:d}%")

    def update_preview(self) -> None:
        if self.closed:
            return
        try:
            frame = self.camera.frame()
            self.latest_frame = frame
            available_w = max(640, self.preview.winfo_width())
            available_h = max(480, self.preview.winfo_height())
            scale = min(available_w / self.camera.width, available_h / self.camera.height)
            shown = cv2.resize(
                frame,
                (max(1, int(self.camera.width * scale)), max(1, int(self.camera.height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
            self.preview_image = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview.configure(image=self.preview_image)
        except Exception as exc:
            self.status_var.set(f"Preview error: {exc}")
        self.root.after(max(50, round(500 / max(self.camera.fps, 1.0))), self.update_preview)

    def space_photo(self, _event=None):
        if self.root.focus_get() is not self.label_entry:
            self.take_photo()

    def take_photo(self) -> None:
        if self.camera.tuning:
            self.status_var.set("Wait for auto tune to finish before taking a photo.")
            return
        try:
            frame = self.camera.frame()
            label = safe_label(self.label_var.get())
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            destination = self.photo_dir / f"{label}_{stamp}.png"
            if not cv2.imwrite(
                str(destination), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]
            ):
                raise RuntimeError("OpenCV could not write the PNG")
            self.status_var.set(f"Saved {destination.name}")
            print(destination, flush=True)
        except Exception as exc:
            messagebox.showerror("Photo failed", str(exc), parent=self.root)

    def autotune(self) -> None:
        if self.camera.tuning:
            return
        try:
            self.camera.start_autotune()
            self.tune_button.configure(state=tk.DISABLED)
            self.photo_button.configure(state=tk.DISABLED)
            self.exposure_scale.configure(state=tk.DISABLED)
            self.gain_scale.configure(state=tk.DISABLED)
            self.status_var.set(
                "Auto tuning for 8 seconds — keep the white reference still and filling the view…"
            )
            self.root.after(8000, self.finish_autotune)
        except Exception as exc:
            messagebox.showerror("Auto tune failed", str(exc), parent=self.root)

    def finish_autotune(self) -> None:
        try:
            gains = self.camera.finish_autotune()
            self.exposure_var.set(self.camera.exposure_ms)
            self.gain_var.set(self.camera.gain_percent)
            self.update_setting_labels()
            self.status_var.set(
                "Auto tune saved — "
                f"exposure {self.camera.exposure_ms:.1f} ms, gain {self.camera.gain_percent}%, "
                f"RGB {gains['red_gain_factor']}/{gains['green_gain_factor']}/{gains['blue_gain_factor']}."
            )
        except Exception as exc:
            messagebox.showerror("Auto tune failed", str(exc), parent=self.root)
        finally:
            self.tune_button.configure(state=tk.NORMAL)
            self.photo_button.configure(state=tk.NORMAL)
            self.exposure_scale.configure(state=tk.NORMAL)
            self.gain_scale.configure(state=tk.NORMAL)

    def change_exposure(self, value: str) -> None:
        if self.camera.tuning:
            return
        try:
            self.camera.set_exposure(float(value))
            self.update_setting_labels()
        except Exception as exc:
            self.status_var.set(f"Exposure error: {exc}")

    def change_gain(self, value: str) -> None:
        if self.camera.tuning:
            return
        try:
            self.camera.set_gain(round(float(value)))
            self.update_setting_labels()
        except Exception as exc:
            self.status_var.set(f"Gain error: {exc}")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.camera.close()
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, default=DEFAULT_PHOTOS)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        camera = PixelFoxCamera(args.profile.expanduser().resolve())
    except Exception as exc:
        print(f"Could not open PixelFox camera: {exc}", file=sys.stderr)
        print("Connect the IDS camera and verify ueyeusbdrc.service is active.", file=sys.stderr)
        return 1
    root = tk.Tk()
    PixelFoxApp(root, camera, args.photos.expanduser().resolve())
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
