"""
Media processing functions for video thumbnail generation.

Handles FFmpeg execution, BIF file generation, and all media processing
logic including HDR detection, skip frame heuristics, and GPU acceleration.
"""

import os
import re
import struct
import array
import glob
import time
import subprocess
import shutil
import sys
import http.client
import xml.etree.ElementTree
import tempfile
from typing import Optional, List, Tuple
from loguru import logger

from .utils import sanitize_path

try:
    from pymediainfo import MediaInfo
    # Test that native library is available
    MediaInfo.can_parse()
except ImportError:
    print('ERROR: pymediainfo Python package not found.')
    print('Please install: pip install pymediainfo')
    sys.exit(1)
except OSError as e:
    if 'libmediainfo' in str(e).lower():
        print('ERROR: MediaInfo native library not found.')
        print('Please install MediaInfo:')
        if sys.platform == 'darwin':  # macOS
            print('  macOS: brew install media-info')
        elif sys.platform.startswith('linux'):
            print('  Ubuntu/Debian: sudo apt-get install mediainfo libmediainfo-dev')
            print('  Fedora/RHEL: sudo dnf install mediainfo mediainfo-devel')
        else:
            print('  See: https://mediaarea.net/en/MediaInfo/Download')
        sys.exit(1)
except Exception as e:
    print(f'WARNING: Could not validate MediaInfo library: {e}')
    print('Proceeding anyway, but errors may occur during processing')

from .config import Config
from .plex_client import retry_plex_call


class CodecNotSupportedError(Exception):
    """
    Exception raised when a video codec is not supported by GPU hardware.

    This exception signals that the file should be processed by a CPU worker
    instead of attempting CPU fallback within the GPU worker thread.
    """
    pass


def parse_ffmpeg_progress_line(line: str, total_duration: float, progress_callback=None):
    """
    Parse a single FFmpeg progress line and call progress callback if provided.

    Args:
        line: FFmpeg output line to parse
        total_duration: Total video duration in seconds
        progress_callback: Callback function for progress updates
    """
    # Parse duration
    if 'Duration:' in line:
        duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})', line)
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return total_duration

    # Parse FFmpeg progress line with all data
    elif 'time=' in line:
        # Extract all FFmpeg data fields
        frame_match = re.search(r'frame=\s*(\d+)', line)
        fps_match = re.search(r'fps=\s*([0-9.]+)', line)
        q_match = re.search(r'q=([0-9.]+)', line)
        size_match = re.search(r'size=\s*(\d+)kB', line)
        time_match = re.search(r'time=(\d{2}):(\d{2}):(\d{2}\.\d{2})', line)
        bitrate_match = re.search(r'bitrate=\s*([0-9.]+)kbits/s', line)
        speed_match = re.search(r'speed=\s*([0-9]+\.?[0-9]*|\.[0-9]+)x', line)

        # Extract values
        frame = int(frame_match.group(1)) if frame_match else 0
        fps = float(fps_match.group(1)) if fps_match else 0
        q = float(q_match.group(1)) if q_match else 0
        size = int(size_match.group(1)) if size_match else 0
        bitrate = float(bitrate_match.group(1)) if bitrate_match else 0
        speed = speed_match.group(1) + "x" if speed_match else None

        if time_match:
            hours, minutes, seconds = time_match.groups()
            current_time = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            time_str = f"{hours}:{minutes}:{seconds}"

            # Update progress
            progress_percent = 0
            if total_duration and total_duration > 0:
                progress_percent = min(100, int((current_time / total_duration) * 100))

            # Calculate remaining time from FFmpeg data
            remaining_time = 0
            if total_duration and total_duration > 0 and current_time < total_duration:
                remaining_time = total_duration - current_time

            # Call progress callback with all FFmpeg data
            if progress_callback:
                progress_callback(progress_percent, current_time, total_duration, speed or "0.0x",
                                remaining_time, frame, fps, q, size, time_str, bitrate)

    return total_duration


def _detect_codec_error(returncode: int, stderr_lines: List[str]) -> bool:
    """
    Detect if FFmpeg failure is due to unsupported codec/hardware decoder error.

    Checks exit codes and stderr patterns to identify codec-related errors.
    Based on FFmpeg documentation: exit code -22 (EINVAL) and 69 (max error rate)
    are common for unsupported codecs, but stderr parsing is more reliable.

    Args:
        returncode: FFmpeg exit code
        stderr_lines: List of stderr output lines (case-insensitive matching)

    Returns:
        bool: True if codec/decoder error detected, False otherwise
    """
    # Combine all stderr lines into a single lowercase string for pattern matching
    stderr_text = ' '.join(stderr_lines).lower()

    # Pattern list for codec/decoder errors (based on FFmpeg documentation and common error messages)
    # Focus ONLY on errors that indicate the codec is not supported by the hardware decoder
    # Avoid patterns that could indicate other issues (corruption, memory, permissions, etc.)
    codec_error_patterns = [
        # Specific decoder errors (indicate codec not available for hardware decoder)
        'no decoder for',
        'unknown decoder',
        'decoder not found',
        'could not find codec',
        'unsupported codec id',
        # Hardware decoder specific errors (clearly indicate hardware decoder limitations)
        'hardware decoder not found',
        'hardware decoder unavailable',
        'hwaccel decoder not found',
        'hwaccel decoder unavailable',
        # Generic codec errors (check these carefully - only in GPU context after failure)
        'unsupported codec',
        'codec not supported',
    ]

    # Check for codec error patterns in stderr (primary detection method)
    for pattern in codec_error_patterns:
        if pattern in stderr_text:
            return True

    # Check exit codes that may indicate codec issues
    # -22 (EINVAL) - invalid argument, often codec-related
    # 234 (wrapped -22 on Unix systems)
    # 69 (max error rate) - FFmpeg hits error rate limit, often due to decode failures
    # 1 (generic error) when combined with codec error patterns in stderr
    # Note: We check returncode != 0 to avoid false positives on success
    if returncode != 0:
        # Primary codes for codec errors
        if returncode in [-22, 234, 69]:
            return True
        # For other non-zero codes, rely on stderr patterns (already checked above)

    return False


def heuristic_allows_skip(ffmpeg_path: str, video_file: str) -> bool:
    """
    Using the first 10 frames of file to decide if -skip_frame:v nokey is safe.
    Uses -err_detect explode + -xerror to bail immediately on the first decode error.
    Returns True if the probe succeeds, else False. Logs a short tail if available.
    """
    null_sink = "NUL" if os.name == "nt" else "/dev/null"
    cmd = [
        ffmpeg_path,
        "-hide_banner", "-nostats",
        "-v", "error",            # only errors
        "-xerror",                # make errors set non-zero exit
        "-err_detect", "explode", # fail fast on decode issues
        "-skip_frame:v", "nokey",
        "-threads:v", "1",
        "-i", video_file,
        "-an", "-sn", "-dn",
        "-frames:v", "10",         # stop as soon as one frame decodes
        "-f", "null", null_sink
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    ok = (proc.returncode == 0)
    if not ok:
        last = (proc.stderr or "").strip().splitlines()[-1:]  # tail(1)
        logger.debug(f"skip_frame probe FAILED at 0s: rc={proc.returncode} msg={last}")
    else:
        logger.debug("skip_frame probe OK at 0s")
    return ok


def generate_images(video_file: str, output_folder: str, gpu: Optional[str],
                   gpu_device_path: Optional[str], config: Config, progress_callback=None,
                   media_info: Optional['MediaInfo'] = None) -> tuple:
    """
    Generate thumbnail images from a video using FFmpeg.

    Runs FFmpeg with hardware acceleration when configured. If the skip-frame
    heuristic allowed it, attempts with '-skip_frame:v nokey' first. If that
    yields zero images or returns non-zero, automatically retries without '-skip_frame'.

    If GPU processing fails with a codec error (detected via stderr parsing for
    patterns like "Codec not supported", "Unsupported codec", etc., or exit codes
    -22/EINVAL or 69/max error rate) and CPU threads are available, automatically
    falls back to CPU processing. This ensures files are processed even when the
    GPU doesn't support the codec (e.g., AV1 on RTX 2060 SUPER).

    Args:
        video_file: Path to input video file
        output_folder: Directory where thumbnail images will be written
        gpu: GPU type ('NVIDIA', 'AMD', 'INTEL', 'WINDOWS_GPU', 'APPLE', or None)
        gpu_device_path: GPU device path (e.g., '/dev/dri/renderD128' for VAAPI)
        config: Configuration object
        progress_callback: Optional progress callback for UI updates

    Returns:
        (success, image_count, hw_used, seconds, speed):
            success (bool): True if at least one image was produced
            image_count (int): Number of images written
            hw_used (bool): Whether hardware acceleration was actually used
                           (False if CPU fallback occurred)
            seconds (float): Elapsed processing time (last attempt)
            speed (str): Reported or computed FFmpeg speed string
    """
    if media_info is None:
        media_info = MediaInfo.parse(video_file)
    fps_value = round(1 / config.plex_bif_frame_interval, 6)

    # Base video filter for SDR content
    base_scale = "scale=w=320:h=240:force_original_aspect_ratio=decrease"
    vf_parameters = f"fps=fps={fps_value}:round=up,{base_scale}"

    # Check if we have HDR Format. Note: Sometimes it can be returned as "None" (string) hence the check for None type or "None" (String)
    if media_info.video_tracks:
        if media_info.video_tracks[0].hdr_format != "None" and media_info.video_tracks[0].hdr_format is not None:
            vf_parameters = f"fps=fps={fps_value}:round=up,zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,{base_scale}"

    def _run_ffmpeg(use_skip: bool, gpu_override: Optional[str] = None, gpu_device_path_override: Optional[str] = None) -> tuple:
        """Run FFmpeg once and return (returncode, seconds, speed, stderr_lines)."""
        # Build FFmpeg command with proper argument ordering
        # Hardware acceleration flags must come BEFORE the input file (-i)
        args = [
            config.ffmpeg_path, "-loglevel", "info",
            "-threads:v", "1",
        ]

        # Add hardware acceleration for decoding (before -i flag)
        # Allow overriding GPU settings for CPU fallback
        effective_gpu = gpu_override if gpu_override is not None else gpu
        effective_gpu_device_path = gpu_device_path_override if gpu_device_path_override is not None else gpu_device_path

        use_gpu = effective_gpu is not None
        if use_gpu:
            if effective_gpu == 'NVIDIA':
                args += ["-hwaccel", "cuda"]
            elif effective_gpu == 'WINDOWS_GPU':
                args += ["-hwaccel", "d3d11va"]
            elif effective_gpu == 'APPLE':
                args += ["-hwaccel", "videotoolbox"]
            elif effective_gpu_device_path and effective_gpu_device_path.startswith('/dev/dri/'):
                args += ["-hwaccel", "vaapi", "-vaapi_device", effective_gpu_device_path]

        # Add skip_frame option for faster decoding (if safe)
        if use_skip:
            args += ["-skip_frame:v", "nokey"]

        # Add input file and output options
        args += [
            "-i", video_file, "-an", "-sn", "-dn",
            "-q:v", str(config.thumbnail_quality),
            "-vf", vf_parameters,
            f'{output_folder}/img-%06d.jpg'
        ]

        if config.regenerate_thumbnails:
            args += ["-y"]

        start_local = time.time()
        logger.debug(f'Executing: {" ".join(args)}')

        # Use file polling approach for non-blocking, high-frequency progress monitoring
        import threading
        thread_id = threading.get_ident()
        output_file = os.path.join(tempfile.gettempdir(), f'ffmpeg_output_{os.getpid()}_{thread_id}_{time.time_ns()}.log')
        proc = subprocess.Popen(args, stderr=open(output_file, 'w', encoding='utf-8'), stdout=subprocess.DEVNULL)

        # Signal that FFmpeg process has started
        if progress_callback:
            progress_callback(0, 0, 0, "0.0x", media_file=video_file)

        # Track progress
        total_duration = None
        speed_local = "0.0x"
        ffmpeg_output_lines = []
        line_count = 0

        def speed_capture_callback(progress_percent, current_duration, total_duration_param, speed_value,
                                  remaining_time=None, frame=0, fps=0, q=0, size=0, time_str="00:00:00.00", bitrate=0):
            nonlocal speed_local
            if speed_value and speed_value != "0.0x":
                speed_local = speed_value
            if progress_callback:
                progress_callback(progress_percent, current_duration, total_duration_param, speed_value,
                                remaining_time, frame, fps, q, size, time_str, bitrate, media_file=video_file)

        time.sleep(0.02)
        while proc.poll() is None:
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) > line_count:
                            for i in range(line_count, len(lines)):
                                line = lines[i].strip()
                                if line:
                                    ffmpeg_output_lines.append(line)
                                    total_duration = parse_ffmpeg_progress_line(line, total_duration, speed_capture_callback)
                            line_count = len(lines)
                except (OSError, IOError):
                    pass
            time.sleep(0.005)

        # Process any remaining data
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > line_count:
                        for i in range(line_count, len(lines)):
                            line = lines[i].strip()
                            if line:
                                ffmpeg_output_lines.append(line)
                                total_duration = parse_ffmpeg_progress_line(line, total_duration, speed_capture_callback)
            except (OSError, IOError):
                pass

        try:
            os.remove(output_file)
        except OSError:
            pass

        # Error logging
        if proc.returncode != 0:
            logger.error(f'FFmpeg failed with return code {proc.returncode} for {video_file}')

            # Check for permission-related errors in FFmpeg output
            # FFmpeg outputs "Permission denied" in messages like "av_interleaved_write_frame(): Permission denied"
            # We use lowercase for case-insensitive matching
            permission_keywords = ['permission denied', 'access denied']
            permission_errors = []
            for line in ffmpeg_output_lines:
                line_lower = line.lower()
                for keyword in permission_keywords:
                    if keyword.lower() in line_lower:
                        permission_errors.append(line.strip())
                        break

            # Log permission errors at INFO level so users can see them without DEBUG
            if permission_errors:
                logger.info(f'Permission error detected while processing {video_file}:')
                for error_line in permission_errors[:3]:  # Show up to 3 permission error lines
                    logger.info(f'  {error_line}')
                if len(permission_errors) > 3:
                    logger.info(f'  ... and {len(permission_errors) - 3} more permission-related error(s)')

            # Always log full FFmpeg output at DEBUG level for detailed troubleshooting
            if logger.level("DEBUG").no <= logger._core.min_level:
                logger.debug(f"FFmpeg output ({len(ffmpeg_output_lines)} lines):")
                for i, line in enumerate(ffmpeg_output_lines[-10:]):
                    logger.debug(f"  {i+1:3d}: {line}")

        end_local = time.time()
        seconds_local = round(end_local - start_local, 1)
        # Calculate fallback speed if needed
        if speed_local == "0.0x" and total_duration and total_duration > 0 and seconds_local > 0:
            calculated_speed = total_duration / seconds_local
            speed_local = f"{calculated_speed:.0f}x"

        return proc.returncode, seconds_local, speed_local, ffmpeg_output_lines

    # Decide initial skip usage from heuristic
    use_skip_initial = heuristic_allows_skip(config.ffmpeg_path, video_file)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # First attempt
    rc, seconds, speed, stderr_lines = _run_ffmpeg(use_skip_initial)

    # Retry once without skip_frame only if FFmpeg returned non-zero and we tried with skip
    # (If we didn't use skip initially, retrying without skip would just repeat the same command)
    did_retry = False
    retry_rc = rc
    retry_stderr_lines = stderr_lines

    if rc != 0 and use_skip_initial:
        did_retry = True
        logger.warning(f"No thumbnails generated from {video_file} with -skip_frame; retrying without skip-frame")
        # Clean up any partial files from first attempt (no need to rename if we're retrying)
        for img in glob.glob(os.path.join(output_folder, '*.jpg')):
            try:
                os.remove(img)
            except Exception:
                pass
        retry_rc, seconds, speed, retry_stderr_lines = _run_ffmpeg(use_skip=False)
        # Update rc and stderr_lines to retry results for codec error detection
        rc = retry_rc
        stderr_lines = retry_stderr_lines

    # Count images first to see if we have any (even if rc != 0, we might have partial success)
    image_count = len(glob.glob(os.path.join(output_folder, 'img*.jpg')))

    # Check for codec errors after both attempts (with and without skip_frame)
    # If in GPU context and codec error detected, raise exception for worker pool to handle
    if rc != 0 and image_count == 0 and gpu is not None:
        if _detect_codec_error(rc, stderr_lines):
            # Log relevant stderr excerpt for debugging
            stderr_excerpt = '\n'.join(stderr_lines[-5:]) if len(stderr_lines) > 0 else "No stderr output"
            logger.warning(f"GPU processing failed with codec error (exit code {rc}) for {video_file}; will hand off to CPU worker")
            logger.debug(f"FFmpeg stderr excerpt (last 5 lines): {stderr_excerpt}")
            # Clean up any partial files from GPU attempts
            for img in glob.glob(os.path.join(output_folder, '*.jpg')):
                try:
                    os.remove(img)
                except Exception:
                    pass
            # Raise exception to signal worker pool to re-queue for CPU worker
            raise CodecNotSupportedError(f"Codec not supported by GPU for {video_file} (exit code {rc})")

    # CPU fallback: Only perform CPU fallback when not in GPU context (e.g., when gpu is None)
    # This preserves the existing fallback behavior for non-GPU processing paths
    did_cpu_fallback = False
    if rc != 0 and image_count == 0 and gpu is None:
        # Detect if failure is due to unsupported codec (even on CPU, for edge cases)
        if _detect_codec_error(rc, stderr_lines):
            logger.warning(f"Processing failed with codec error (exit code {rc}) for {video_file}; file may be corrupted or unsupported")
            # For CPU context, we can't fallback further, so just log and continue
            # The function will return failure status

    # Rename images only after all retries and error checks are complete
    # This ensures we don't rename images that will be cleaned up due to errors
    if image_count > 0:
        # Rename images from img-*.jpg format to timestamp-based names
        for image in glob.glob(f'{output_folder}/img*.jpg'):
            frame_no = int(os.path.basename(image).strip('-img').strip('.jpg')) - 1
            frame_second = frame_no * config.plex_bif_frame_interval
            os.rename(image, os.path.join(output_folder, f'{frame_second:010d}.jpg'))
        # Re-count after renaming to get final count (includes both renamed and any existing timestamped images)
        image_count = len(glob.glob(os.path.join(output_folder, '*.jpg')))

    hw = (gpu is not None and not did_cpu_fallback)
    success = image_count > 0

    if success:
        fallback_suffix = " (CPU fallback)" if did_cpu_fallback else (" (retry no-skip)" if did_retry else "")
        logger.info(f'Generated Video Preview for {video_file} HW={hw} TIME={seconds}seconds SPEED={speed} IMAGES={image_count}{fallback_suffix}')
    else:
        fallback_suffix = " (after CPU fallback)" if did_cpu_fallback else (" after retry" if did_retry else "")
        logger.error(f'Failed to generate thumbnails for {video_file}; 0 images produced{fallback_suffix}')

    return success, image_count, hw, seconds, speed


def _setup_bundle_paths(bundle_hash: str, config: Config) -> Tuple[str, str, str, str]:
    """
    Set up all bundle-related paths.

    Args:
        bundle_hash: Bundle hash from Plex
        config: Configuration object

    Returns:
        Tuple of (indexes_path, index_bif, tmp_path, chapters_path)
    """
    bundle_file = sanitize_path(f'{bundle_hash[0]}/{bundle_hash[1::1]}.bundle')
    bundle_path = sanitize_path(os.path.join(config.plex_config_folder, 'Media', 'localhost', bundle_file))
    indexes_path = sanitize_path(os.path.join(bundle_path, 'Contents', 'Indexes'))
    index_bif = sanitize_path(os.path.join(indexes_path, 'index-sd.bif'))
    tmp_path = sanitize_path(os.path.join(config.working_tmp_folder, bundle_hash))
    chapters_path = sanitize_path(os.path.join(bundle_path, 'Contents', 'Chapters'))
    return indexes_path, index_bif, tmp_path, chapters_path


def _ensure_directories(indexes_path: str, tmp_path: str, media_file: str) -> bool:
    """
    Ensure required directories exist.

    Args:
        indexes_path: Path to indexes directory
        tmp_path: Path to temporary directory
        media_file: Media file path for error messages

    Returns:
        True if directories are ready, False if creation failed
    """
    if not os.path.isdir(indexes_path):
        try:
            os.makedirs(indexes_path)
        except PermissionError as e:
            logger.error(f'Permission denied creating index path {indexes_path} for {media_file}: {e}')
            logger.info(f'Please check directory permissions for: {os.path.dirname(indexes_path)}')
            return False
        except OSError as e:
            logger.error(f'Error generating images for {media_file}. `{type(e).__name__}:{str(e)}` error when creating index path {indexes_path}')
            return False

    if not os.path.isdir(tmp_path):
        try:
            os.makedirs(tmp_path)
        except PermissionError as e:
            logger.error(f'Permission denied creating tmp path {tmp_path} for {media_file}: {e}')
            logger.info(f'Please check directory permissions for: {os.path.dirname(tmp_path)}')
            return False
        except OSError as e:
            logger.error(f'Error generating images for {media_file}. `{type(e).__name__}:{str(e)}` error when creating tmp path {tmp_path}')
            return False

    return True


def _cleanup_temp_directory(tmp_path: str) -> None:
    """
    Clean up temporary directory, logging warnings on failure.

    Args:
        tmp_path: Path to temporary directory
    """
    try:
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
    except Exception as cleanup_error:
        logger.warning(f"Failed to clean up temp directory {tmp_path}: {cleanup_error}")


def _generate_and_save_bif(media_file: str, tmp_path: str, index_bif: str,
                           gpu: Optional[str], gpu_device_path: Optional[str],
                           config: Config, progress_callback=None,
                           media_info: Optional['MediaInfo'] = None) -> None:
    """
    Generate images and create BIF file.

    Args:
        media_file: Path to media file
        tmp_path: Temporary directory for images
        index_bif: Path to output BIF file
        gpu: GPU type for acceleration
        gpu_device_path: GPU device path
        config: Configuration object
        progress_callback: Callback function for progress updates

    Raises:
        CodecNotSupportedError: If codec is not supported by GPU
        RuntimeError: If thumbnail generation produced 0 images
    """
    try:
        gen_result = generate_images(media_file, tmp_path, gpu, gpu_device_path, config, progress_callback, media_info=media_info)
    except CodecNotSupportedError:
        # Clean up temp directory before re-raising
        _cleanup_temp_directory(tmp_path)
        raise
    except Exception as e:
        logger.error(f'Error generating images for {media_file}. `{type(e).__name__}:{str(e)}` error when generating images')
        _cleanup_temp_directory(tmp_path)
        raise RuntimeError(f"Failed to generate images: {e}")

    # Determine image count from result or by scanning
    image_count = 0
    if isinstance(gen_result, tuple) and len(gen_result) >= 2:
        success, image_count = bool(gen_result[0]), int(gen_result[1])
    else:
        if os.path.isdir(tmp_path):
            image_count = len(glob.glob(os.path.join(tmp_path, '*.jpg')))

    if image_count == 0:
        logger.error(f'No thumbnails generated for {media_file}; skipping BIF creation')
        _cleanup_temp_directory(tmp_path)
        raise RuntimeError(f"Thumbnail generation produced 0 images for {media_file}")

    # Generate BIF file
    try:
        generate_bif(index_bif, tmp_path, config)
    except PermissionError as e:
        # Remove BIF if generation failed
        try:
            if os.path.exists(index_bif):
                os.remove(index_bif)
        except Exception as remove_error:
            logger.warning(f"Failed to remove failed BIF file {index_bif}: {remove_error}")
        logger.error(f'Permission denied generating BIF file {index_bif} for {media_file}: {e}')
        logger.info(f'Please check write permissions for: {os.path.dirname(index_bif)}')
        raise
    except Exception as e:
        # PermissionError is already handled above, so this catches other exceptions
        logger.error(f'Error generating images for {media_file}. `{type(e).__name__}:{str(e)}` error when generating bif')
        # Remove BIF if generation failed
        try:
            if os.path.exists(index_bif):
                os.remove(index_bif)
        except Exception as remove_error:
            logger.warning(f"Failed to remove failed BIF file {index_bif}: {remove_error}")
        raise


def generate_bif(bif_filename: str, images_path: str, config: Config) -> None:
    """
    Build a .bif file from thumbnail images.

    Args:
        bif_filename: Path to output .bif file
        images_path: Directory containing .jpg thumbnail images
        config: Configuration object

    Raises:
        PermissionError: If permission denied accessing files or directories
    """
    magic = [0x89, 0x42, 0x49, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]
    version = 0

    try:
        images = [img for img in os.listdir(images_path) if os.path.splitext(img)[1] == '.jpg']
    except PermissionError as e:
        logger.error(f'Permission denied reading images directory {images_path}: {e}')
        logger.info(f'Please check read permissions for: {images_path}')
        raise
    images.sort()

    try:
        f = open(bif_filename, "wb")
    except PermissionError as e:
        logger.error(f'Permission denied writing BIF file {bif_filename}: {e}')
        logger.info(f'Please check write permissions for: {os.path.dirname(bif_filename)}')
        raise

    try:
        with f:
            array.array('B', magic).tofile(f)
            f.write(struct.pack("<I", version))
            f.write(struct.pack("<I", len(images)))
            f.write(struct.pack("<I", 1000 * config.plex_bif_frame_interval))
            array.array('B', [0x00 for x in range(20, 64)]).tofile(f)

            bif_table_size = 8 + (8 * len(images))
            image_index = 64 + bif_table_size
            timestamp = 0

            # Get the length of each image
            for image in images:
                try:
                    statinfo = os.stat(os.path.join(images_path, image))
                except PermissionError as e:
                    logger.error(f'Permission denied reading image file {os.path.join(images_path, image)}: {e}')
                    logger.info(f'Please check read permissions for: {images_path}')
                    raise
                f.write(struct.pack("<I", timestamp))
                f.write(struct.pack("<I", image_index))
                timestamp += 1
                image_index += statinfo.st_size

            f.write(struct.pack("<I", 0xffffffff))
            f.write(struct.pack("<I", image_index))

            # Now copy the images
            for image in images:
                try:
                    with open(os.path.join(images_path, image), "rb") as img_file:
                        data = img_file.read()
                except PermissionError as e:
                    logger.error(f'Permission denied reading image file {os.path.join(images_path, image)}: {e}')
                    logger.info(f'Please check read permissions for: {images_path}')
                    raise
                f.write(data)
    except PermissionError:
        # Re-raise PermissionError (already logged above)
        raise
    logger.debug(f'Generated BIF file: {bif_filename}')


def generate_chapter_thumbnails(
    video_file: str,
    chapter_offsets: List[float],
    chapters_path: str,
    gpu: Optional[str],
    gpu_device_path: Optional[str],
    config: Config,
    progress_callback=None,
    media_info: Optional['MediaInfo'] = None,
) -> None:
    """
    Generate HDR-tonemapped chapter thumbnails as JPEGs in the Plex bundle.

    Args:
        video_file: Path to media file
        chapter_offsets: List of chapter start times in seconds
        chapters_path: Plex bundle 'Contents/Chapters' directory
        gpu: GPU type ('NVIDIA', 'AMD', etc.) or None
        gpu_device_path: GPU device path (e.g. /dev/dri/renderD128) or None
        config: Configuration object
        progress_callback: Optional callback for progress reporting
    """
    # Ensure directory
    try:
        os.makedirs(chapters_path, exist_ok=True)
    except PermissionError as e:
        logger.error(f"Permission denied creating chapters directory {chapters_path}: {e}")
        return

    # Regeneration policy:
    # - If --regenerate-thumbnails is set, remove existing chapter images for this item.
    # - Otherwise, keep existing chapter images and only generate missing ones.
    if config.regenerate_thumbnails:
        try:
            removed = 0
            # Plex may store chapter thumbs as .jpg or .jpeg depending on version/platform
            patterns = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
            for pat in patterns:
                for p in glob.glob(os.path.join(chapters_path, pat)):
                    try:
                        os.remove(p)
                        removed += 1
                    except Exception:
                        pass
        except Exception as e:
            logger.error(
                f"Chapter thumbnails: error {type(e).__name__} while deleting existing thumbnails for {video_file}: {e}"
            )

        # Also remove the no-chapters marker so future runs will re-check chapters.
        marker = os.path.join(chapters_path, ".no_chapters")
        try:
            if os.path.exists(marker):
                os.remove(marker)
        except Exception:
            pass

        # Remove any chapter-count marker(s) so we can recompute expected chapter count.
        try:
            for p in glob.glob(os.path.join(chapters_path, ".chapters_*")):
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception:
            pass

    if not chapter_offsets:
        logger.debug(f"No chapters found for {video_file}; skipping chapter thumbnails")
        return
    if media_info is None:
        media_info = MediaInfo.parse(video_file)

    # Duration (seconds) for clamping chapter thumb timestamps.
    duration_s = None
    try:
        if getattr(media_info, 'general_tracks', None) and media_info.general_tracks:
            d = getattr(media_info.general_tracks[0], 'duration', None)
            if d is not None:
                duration_s = float(d) / 1000.0
        if duration_s is None and getattr(media_info, 'video_tracks', None) and media_info.video_tracks:
            d = getattr(media_info.video_tracks[0], 'duration', None)
            if d is not None:
                duration_s = float(d) / 1000.0
    except Exception:
        duration_s = None

    # Keep a small safety margin so we don't seek past EOF.
    duration_margin_s = 0.10

    # Base scale (same as generate_images)
    base_scale = "scale=w=1280:h=720:force_original_aspect_ratio=decrease:flags=lanczos"
    vf_parameters = base_scale

    # Mirror the HDR handling used in generate_images EXACTLY (same filter chain, minus fps)
    is_hdr = False
    hdr_format = None
    if media_info.video_tracks:
        hdr_format = media_info.video_tracks[0].hdr_format
        # Sometimes this is the literal string "None"
        if hdr_format != "None" and hdr_format is not None:
            is_hdr = True
            vf_parameters = (
                "zscale=t=linear:npl=100,format=gbrpf32le,"
                "zscale=p=bt709,"
                "tonemap=tonemap=mobius:desat=0,"
                "zscale=t=bt709:m=bt709:r=tv,"
                f"format=yuv420p,{base_scale}"
            )
    logger.debug(
        f"Chapter thumbs HDR detect: is_hdr={is_hdr} hdr_format={hdr_format} file={video_file}"
    )

    quality = str(config.thumbnail_quality)

    # For single-frame extractions, the skip-frame heuristic is less useful,
    # but it can still reduce work when the seek lands on keyframes.
    allow_skip = heuristic_allows_skip(config.ffmpeg_path, video_file)

    def _build_ffmpeg_args(use_skip: bool, effective_gpu: Optional[str], effective_gpu_device_path: Optional[str], ts: float, out_file: str, use_sseof: bool = False) -> List[str]:
        # Hardware accel flags must come before -i
        args: List[str] = [
            config.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-threads:v", "1",
        ]

        # Prevent FFmpeg from prompting on existing outputs.
        # - When regenerating, overwrite outputs.
        # - Otherwise, refuse to overwrite.
        if config.regenerate_thumbnails:
            args += ["-y"]
        else:
            args += ["-n"]

        if effective_gpu is not None:
            if effective_gpu == 'NVIDIA':
                args += ["-hwaccel", "cuda"]
            elif effective_gpu == 'WINDOWS_GPU':
                args += ["-hwaccel", "d3d11va"]
            elif effective_gpu == 'APPLE':
                args += ["-hwaccel", "videotoolbox"]
            elif effective_gpu_device_path and effective_gpu_device_path.startswith('/dev/dri/'):
                args += ["-hwaccel", "vaapi", "-vaapi_device", effective_gpu_device_path]

        if use_skip:
            args += ["-skip_frame:v", "nokey"]

        # Seek strategy:
        # - Default: fast seek with -ss before -i.
        # - Near/at EOF (last chapter): use -sseof to guarantee we land on decodable frames.
        if use_sseof:
            # Half a second before EOF is usually safe even when the final chapter is exactly at the end.
            args += [
                "-sseof", "-3.0",
                "-i", video_file,
            ]
        else:
            args += [
                "-ss", f"{ts:.3f}",
                "-i", video_file,
            ]

        args += [
            "-an", "-sn", "-dn",
            "-frames:v", "1",
            "-vf", vf_parameters,
            "-q:v", quality,
            out_file,
        ]
        return args

    written = 0
    skipped_existing = 0
    failed = 0

    total = len(chapter_offsets)

    def _format_hhmmss(seconds: float) -> str:
        if seconds is None or seconds < 0:
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:05.2f}"

    def _report_progress(pct: int, current_s: float, speed_value: str = "0.0x", frame_no: int = 1) -> None:
        if not progress_callback:
            return
        total_s = duration_s if duration_s is not None else 0.0
        remaining_s = (total_s - current_s) if total_s and total_s > current_s else 0.0
        time_str = _format_hhmmss(current_s)
        try:
            progress_callback(
                pct,
                current_s,
                total_s,
                speed_value,
                remaining_s,
                frame_no,  # frame
                0.0,       # fps
                0.0,       # q
                0,         # size
                time_str,
                0.0,       # bitrate
                media_file=video_file,
            )
        except TypeError:
            # Older callback signature
            try:
                progress_callback(pct, current_s, total_s, speed_value, media_file=video_file)
            except Exception:
                pass

    for idx, t in enumerate(chapter_offsets):
        # Plex names chapter thumbnails as chapter1.jpg, chapter2.jpg, ... (1-indexed)
        out_file = os.path.join(chapters_path, f"chapter{idx + 1}.jpg")

        # When not regenerating, do not overwrite existing chapter thumbs.
        if not config.regenerate_thumbnails and os.path.exists(out_file):
            skipped_existing += 1
            continue

        # First attempt: same execution mode as the worker (GPU if provided)
        effective_gpu = gpu
        effective_gpu_device_path = gpu_device_path

        # Offset by up to +1.0s, but never past usable EOF.
        # For chapters near/at EOF, avoid seeking past the end.
        ts = t + 1.0
        use_sseof = False
        if duration_s is not None and duration_s > 0:
            usable_end = max(0.0, duration_s - duration_margin_s)
            remaining = usable_end - t
            # Only advance into the chapter if there is actually time left.
            # If the chapter starts at/after usable_end, keep ts at usable_end.
            if remaining <= 0:
                ts = usable_end
            else:
                ts = t + min(1.0, remaining)

            # Last-chapter handling:
            # - Use -sseof when the chapter is effectively at the very end, to avoid EOF seek errors.
            if idx == (total - 1):
                if remaining <= 3.0:
                    use_sseof = True
        # Progress reporting after timestamp is computed/clamped
        pct = int((idx / max(1, total)) * 100)
        _report_progress(pct, ts, "0.0x", frame_no=idx + 1)
        if duration_s is not None:
            logger.debug(f"Generating chapter thumb {idx + 1} at {ts:.3f}s (chapter {t:.3f}s +1.0s, duration {duration_s:.3f}s) -> {out_file}")
        else:
            logger.debug(f"Generating chapter thumb {idx + 1} at {ts:.3f}s (chapter {t:.3f}s +1.0s) -> {out_file}")
        cmd = _build_ffmpeg_args(allow_skip, effective_gpu, effective_gpu_device_path, ts, out_file, use_sseof=use_sseof)

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except Exception as e:
            logger.error(f"FFmpeg failed for chapter {idx} at {ts:.3f}s: {type(e).__name__}: {e}")
            failed += 1
            continue

        if result.returncode == 0:
            written += 1
            pct_done = int(((idx + 1) / max(1, total)) * 100)
            _report_progress(pct_done, ts, "1.0x", frame_no=idx + 1)
            continue

        stderr_lines = (result.stderr.decode(errors='ignore') or "").splitlines()

        # If GPU was requested and we hit a codec/hw decoder error, signal the worker pool
        # so it can re-queue this item to a CPU worker, mirroring preview generation behavior.
        if gpu is not None and _detect_codec_error(result.returncode, stderr_lines):
            logger.warning(
                f"GPU chapter thumbnail generation failed with codec error (exit code {result.returncode}) for {video_file}; will hand off to CPU worker"
            )
            raise CodecNotSupportedError(
                f"Codec not supported by GPU for chapter thumbs of {video_file} (exit code {result.returncode})"
            )

        # If we tried skip_frame and it failed, retry without skip_frame once.
        if allow_skip:
            cmd2 = _build_ffmpeg_args(False, effective_gpu, effective_gpu_device_path, ts, out_file, use_sseof=use_sseof)
            try:
                result2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            except Exception as e:
                logger.error(f"FFmpeg retry failed for chapter {idx} at {ts:.3f}s: {type(e).__name__}: {e}")
                failed += 1
                continue
            if result2.returncode == 0:
                written += 1
                pct_done = int(((idx + 1) / max(1, total)) * 100)
                _report_progress(pct_done, ts, "1.0x", frame_no=idx + 1)
                continue

            stderr_lines2 = (result2.stderr.decode(errors='ignore') or "").splitlines()
            if gpu is not None and _detect_codec_error(result2.returncode, stderr_lines2):
                logger.warning(
                    f"GPU chapter thumbnail generation failed with codec error after no-skip retry (exit code {result2.returncode}) for {video_file}; will hand off to CPU worker"
                )
                raise CodecNotSupportedError(
                    f"Codec not supported by GPU for chapter thumbs of {video_file} (exit code {result2.returncode})"
                )

            logger.error(
                f"FFmpeg returned {result2.returncode} for chapter {idx} at {ts:.3f}s (after retry): "
                f"{result2.stderr.decode(errors='ignore')}"
            )
            failed += 1
        else:
            logger.error(
                f"FFmpeg returned {result.returncode} for chapter {idx} at {ts:.3f}s: "
                f"{result.stderr.decode(errors='ignore')}"
            )
            failed += 1

    logger.info(
        f"Generated Chapter Thumbnails for {video_file} WROTE={written} SKIPPED_EXISTING={skipped_existing} FAILED={failed}"
    )

    # Final progress callback
    _report_progress(100, (duration_s if duration_s is not None else 0.0), "1.0x", frame_no=(total if total > 0 else 1))


def process_item(item_key: str, gpu: Optional[str], gpu_device_path: Optional[str],
                config: Config, plex, progress_callback=None) -> None:
    """
    Process a single media item: generate thumbnails and BIF file.

    This is the core processing function that handles:
    - Plex API queries
    - Path mapping for remote generation
    - Bundle hash generation
    - Plex directory structure creation
    - Thumbnail generation with FFmpeg
    - BIF file creation
    - Cleanup

    Args:
        item_key: Plex media item key
        gpu: GPU type for acceleration
        gpu_device_path: GPU device path
        config: Configuration object
        plex: Plex server instance
        progress_callback: Callback function for progress updates
    """
    try:
        data = retry_plex_call(plex.query, f'{item_key}/tree')
    except (Exception, http.client.BadStatusLine, xml.etree.ElementTree.ParseError) as e:
        logger.error(f"Failed to query Plex for item {item_key} after retries: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        # For connection errors, log more details
        if hasattr(e, 'request') and e.request:
            logger.error(f"Request URL: {e.request.url}")
            logger.error(f"Request method: {e.request.method}")
            logger.error(f"Request headers: {e.request.headers}")
        return
    except Exception as e:
        logger.error(f"Error querying Plex for item {item_key}: {e}")
        return

    # Lazy chapter retrieval: only query Plex for chapters if we actually need to generate chapter thumbnails.
    _chapter_offsets_cache: Optional[List[float]] = None

    def _get_chapter_offsets() -> List[float]:
        nonlocal _chapter_offsets_cache
        if _chapter_offsets_cache is not None:
            return _chapter_offsets_cache
        try:
            chapters_xml = retry_plex_call(plex.query, f'{item_key}?includeChapters=1')
        except Exception as e:
            logger.error(f"Error querying Plex for chapters of {item_key}: {e}")
            _chapter_offsets_cache = []
            return _chapter_offsets_cache

        offsets: List[float] = []
        # Plex chapter offsets are in milliseconds
        for ch in chapters_xml.findall('.//Chapter'):
            try:
                offset_ms = int(ch.attrib.get('startTimeOffset', '0'))
            except ValueError:
                continue
            offsets.append(offset_ms / 1000.0)

        _chapter_offsets_cache = offsets
        return _chapter_offsets_cache

    for media_part in data.findall('.//MediaPart'):
        if 'hash' in media_part.attrib:
            bundle_hash = media_part.attrib['hash']
            # Apply path mapping if both mapping parameters are provided (for remote generation)
            if config.plex_videos_path_mapping and config.plex_local_videos_path_mapping:
                media_file = sanitize_path(media_part.attrib['file'].replace(config.plex_videos_path_mapping, config.plex_local_videos_path_mapping))
            else:
                # Use file path directly (for local generation)
                media_file = sanitize_path(media_part.attrib['file'])

            # Validate bundle_hash has sufficient length (at least 2 characters)
            if not bundle_hash or len(bundle_hash) < 2:
                hash_value = f'"{bundle_hash}"' if bundle_hash else '(empty)'
                logger.warning(f'Skipping {media_file} due to invalid bundle hash from Plex: {hash_value} (length: {len(bundle_hash) if bundle_hash else 0}, required: >= 2)')
                continue

            if not os.path.isfile(media_file):
                logger.warning(f'Skipping as file not found {media_file}')
                continue

            try:
                indexes_path, index_bif, tmp_path, chapters_path = _setup_bundle_paths(bundle_hash, config)
            except Exception as e:
                logger.error(f'Error generating bundle_file for {media_file} due to {type(e).__name__}:{str(e)}')
                continue

            # Fast gating: decide whether we need to do any work before expensive MediaInfo parsing.
            needs_chapters = bool(config.regenerate_thumbnails)
            needs_previews = bool(config.regenerate_thumbnails)

            # Chapter thumbnails gating:
            # - If .no_chapters marker exists, assume this item has no chapters and skip further work.
            # - Prefer a cheap completeness check using a chapter-count marker file: .chapters_<N>
            #   If it exists, we only need to check for chapter<N>.* to consider the set complete.
            # - If only chapter1.* exists (legacy state), we will query Plex once to learn the chapter
            #   count, write the marker, and then fill any missing thumbnails.
            no_chapters_marker = os.path.join(chapters_path, ".no_chapters")
            no_chapters_known = False

            def _chapter_thumb_path(idx: int) -> Optional[str]:
                # Return the first matching existing chapter thumb path for chapter<idx>.*
                names = (
                    f"chapter{idx}.jpg", f"chapter{idx}.jpeg",
                    f"chapter{idx}.JPG", f"chapter{idx}.JPEG",
                )
                for n in names:
                    p = os.path.join(chapters_path, n)
                    if os.path.exists(p):
                        return p
                return None

            def _read_chapter_count_marker() -> Optional[int]:
                try:
                    markers = glob.glob(os.path.join(chapters_path, ".chapters_*"))
                except Exception:
                    markers = []
                if not markers:
                    return None
                # If multiple exist, prefer the one with the largest numeric suffix.
                best = None
                best_n = None
                for m in markers:
                    base = os.path.basename(m)
                    m2 = re.match(r"^\.chapters_(\d+)$", base)
                    if not m2:
                        continue
                    n = int(m2.group(1))
                    if best_n is None or n > best_n:
                        best_n = n
                        best = m
                return best_n

            def _write_chapter_count_marker(n: int) -> None:
                # Remove any existing markers and write a single .chapters_<n> marker.
                try:
                    for p in glob.glob(os.path.join(chapters_path, ".chapters_*")):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    os.makedirs(chapters_path, exist_ok=True)
                    marker_path = os.path.join(chapters_path, f".chapters_{n}")
                    with open(marker_path, "w", encoding="utf-8") as f:
                        f.write(f"chapters={n}\n")
                except Exception as e:
                    logger.debug(f"Failed to write chapter-count marker for {media_file}: {type(e).__name__}: {e}")

            chapter_count_marker = None
            chapter_set_complete = False
            legacy_chapter1_exists = False

            if not config.regenerate_thumbnails:
                try:
                    no_chapters_known = os.path.exists(no_chapters_marker)
                except Exception:
                    no_chapters_known = False

                if not no_chapters_known:
                    chapter_count_marker = _read_chapter_count_marker()
                    if chapter_count_marker is not None and chapter_count_marker > 0:
                        # Consider complete only if the last expected thumbnail exists.
                        chapter_set_complete = _chapter_thumb_path(chapter_count_marker) is not None
                    else:
                        # Legacy heuristic: chapter1 exists but we don't know expected count.
                        legacy_chapter1_exists = _chapter_thumb_path(1) is not None

            # If we know there are no chapters, do not attempt generation.
            if no_chapters_known:
                needs_chapters = False
            elif chapter_set_complete:
                needs_chapters = False
            else:
                # Either no thumbs, or partial/missing, or legacy state without a marker.
                needs_chapters = True

            # Preview thumbnails (BIF): skip when index-sd.bif exists (unless regenerating)
            bif_exists = False
            if not config.regenerate_thumbnails:
                try:
                    bif_exists = os.path.exists(index_bif)
                except Exception:
                    bif_exists = False
            if not bif_exists:
                needs_previews = True

            # If nothing to do, skip this media part without MediaInfo parsing.
            if (not needs_chapters) and (not needs_previews):
                logger.debug(f"No thumbnails needed for {media_file}; skipping")
                continue

            # Parse MediaInfo once per media file, but only when generation is needed.
            media_info = None
            try:
                media_info = MediaInfo.parse(media_file)
            except Exception as e:
                logger.warning(
                    f"MediaInfo parse failed for {media_file}; proceeding without cached metadata: {type(e).__name__}: {e}"
                )
                media_info = None

            # Chapter thumbnails generation logic using needs_chapters
            if not needs_chapters:
                # Either chapter thumbs exist or we've previously determined this item has no chapters.
                if not config.regenerate_thumbnails and no_chapters_known:
                    logger.debug(f"No chapters previously recorded for {media_file}; skipping chapter thumbnails")
                else:
                    logger.debug(f"Chapter thumbnails already exist for {media_file}; skipping generation")
            else:
                # Query Plex for chapter offsets only when we actually intend to generate chapter thumbnails.
                chapter_offsets = _get_chapter_offsets()

                # If Plex reports no chapters, persist a marker to avoid repeated lookups on future runs.
                if not chapter_offsets and not config.regenerate_thumbnails:
                    try:
                        os.makedirs(chapters_path, exist_ok=True)
                        with open(no_chapters_marker, "w", encoding="utf-8") as f:
                            f.write("no_chapters\n")
                        logger.debug(f"No chapters for {media_file}; wrote marker {no_chapters_marker}")
                    except Exception as e:
                        logger.debug(f"Failed to write no-chapters marker for {media_file}: {type(e).__name__}: {e}")
                else:
                    # Persist expected chapter count so future runs can cheaply detect completeness.
                    # This also allows gap-filling when only some chapter*.jpg exist.
                    if not config.regenerate_thumbnails:
                        try:
                            _write_chapter_count_marker(len(chapter_offsets))
                        except Exception:
                            pass
                    try:
                        generate_chapter_thumbnails(
                            media_file,
                            chapter_offsets,
                            chapters_path,
                            gpu,
                            gpu_device_path,
                            config,
                            progress_callback,
                            media_info=media_info,
                        )
                    except CodecNotSupportedError:
                        raise
                    except Exception as e:
                        logger.error(f"Error generating chapter thumbnails for {media_file}: {type(e).__name__}: {e}")
                        # Continue to preview generation even if chapter thumbs fail.

            # Preview/BIF generation logic using needs_previews
            if not needs_previews:
                logger.debug(f"Preview thumbnails already exist for {media_file}; skipping generation")
            else:
                # Ensure directories before generating previews
                if not _ensure_directories(indexes_path, tmp_path, media_file):
                    continue
                # If BIF exists and not regenerating, skip (should almost never trigger here)
                if os.path.exists(index_bif) and not config.regenerate_thumbnails:
                    logger.debug(f"BIF file already exists for {media_file}; skipping preview generation")
                    continue
                try:
                    _generate_and_save_bif(
                        media_file,
                        tmp_path,
                        index_bif,
                        gpu,
                        gpu_device_path,
                        config,
                        progress_callback,
                        media_info=media_info,
                    )
                except CodecNotSupportedError:
                    raise
                except Exception as e:
                    logger.error(f"Error generating preview thumbnails for {media_file}: {type(e).__name__}: {e}")
                finally:
                    _cleanup_temp_directory(tmp_path)
