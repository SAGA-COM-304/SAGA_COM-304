#!/usr/bin/env bash
set -uo pipefail
trap '' ERR   # Don’t exit on first failure

# ------------------------------------------------------------------
# download_all_segments.sh
#
# Usage:
#   ./download_all_segments.sh <CSV_FILE> <OUT_DIR> [JOBS]
#
#   CSV_FILE   : annotation CSV (header; clip names in first column)
#   OUT_DIR    : output directory for segments
#   JOBS       : number of parallel jobs (default: 4)
# ------------------------------------------------------------------

CSV_FILE="${1:?Please provide path to CSV_FILE}"
OUT_DIR="${2:?Please provide an output directory}"
JOBS="${3:-4}"
DURATION=5  # seconds
LOG_FILE="$OUT_DIR/log_file.log"

VIDEO_DIR="$OUT_DIR/video"
AUDIO_DIR="$OUT_DIR/audio"

# Check dependencies
for cmd in yt-dlp ffmpeg awk parallel; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: '$cmd' is not installed." >&2
    exit 1
  fi
done

export LC_NUMERIC=C

# Prepare directories and log
mkdir -p "$VIDEO_DIR" "$AUDIO_DIR"
: > "$LOG_FILE"

TOTAL=$(awk -F, 'NR>1 && $1!~/^[[:space:]]*$/ {count++} END {print count}' "$CSV_FILE")
echo "⮞ Processing $TOTAL clips with $JOBS jobs..."

process_line() {
  clip_name="$1"
  video_id="${clip_name%_*}"
  start_sec="${clip_name##*_}"

  mp4_path="$VIDEO_DIR/${clip_name}.mp4"
  wav_path="$AUDIO_DIR/${clip_name}.wav"

  [[ -e $mp4_path ]] && return

  COOKIE_OPTS="--cookies cookies.txt"
  url=$(yt-dlp $COOKIE_OPTS \
    --no-warnings --ignore-errors \
    -f "best[height<=360][ext=mp4]/best[height<=360]/best" \
    -g "https://youtu.be/$video_id" 2>>"$LOG_FILE" || echo "")

  if [[ -z $url ]]; then
    printf '%s [%s] ERROR: failed to get video URL\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "https://youtu.be/$video_id" \
      >> "$LOG_FILE"
    return
  fi

  ffmpeg -loglevel error -ss "$start_sec" -i "$url" -t "$DURATION" \
    -c:v h264_videotoolbox -b:v 1800k -an "$mp4_path"

  ffmpeg -loglevel error -ss "$start_sec" -i "$url" -t "$DURATION" \
    -ar 16000 -c:a pcm_s16le -ac 1 "$wav_path"
}

export -f process_line
export DURATION VIDEO_DIR AUDIO_DIR LOG_FILE

awk -F, 'NR>1 && $1!~/^[[:space:]]*$/ {print $1}' "$CSV_FILE" \
  | parallel --bar -j "$JOBS" process_line {}

echo "All segments are available in $OUT_DIR"
