#!/usr/bin/env bash
set -uo pipefail
trap '' ERR   # Don’t exit on first failure

# ------------------------------------------------------------------
# download_videos_rr.sh
#
# Usage:
#   ./download_videos_rr.sh <CSV_FILE> <OUT_DIR> [JOBS] [BATCH_IDX]
#
#   CSV_FILE   : annotation CSV (header; clip names in first column)
#   OUT_DIR    : output directory for segments
#   JOBS       : number of parallel jobs (default: 4)
#   BATCH_IDX  : optional batch number (1,2,…) to create sub-folder 01/, 02/, …
# ------------------------------------------------------------------

# ==== Round-Robin Cookie Settings ====
COOKIE_COUNT=4                                                     # Number of cookie files to rotate
COOKIE_PREFIX="dataset_module/src/downloader/cookies/cookies"      # Base name (e.g. cookies1.txt, cookies2.txt, …)
SEGMENTS_PER_COOKIE=500


# PO TOKEN
PO_TOKEN= ...
# ====================================

CSV_FILE="${1:?Please provide path to CSV_FILE}"
OUT_DIR="${2:?Please provide an output directory}"
JOBS="${3:-4}"
BATCH_IDX="${4:-}"       

#if BATCH_IDX given, add subdirectory 01, 02 ...
if [[ -n "$BATCH_IDX" ]]; then
  BATCH_DIR=$(printf "%02d" "$BATCH_IDX")
  OUT_DIR="$OUT_DIR/$BATCH_DIR"
fi

DURATION=5                   # seconds per segment
LOG_FILE="$OUT_DIR/log_file.log"

VIDEO_DIR="$OUT_DIR/video"
AUDIO_DIR="$OUT_DIR/audio"

# Check dependencies
for cmd in yt-dlp ffmpeg awk parallel bc; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: '$cmd' is not installed." >&2
    exit 1
  fi
done

export LC_NUMERIC=C

# Prepare output dirs and log
mkdir -p "$VIDEO_DIR" "$AUDIO_DIR"
: > "$LOG_FILE"

TOTAL=$(awk -F, 'NR>1 && $1!~/^[[:space:]]*$/ {count++} END {print count}' "$CSV_FILE")
echo "⮞ Processing $TOTAL clips with $JOBS jobs and $COOKIE_COUNT cookies…"

process_line() {
  local clip_name="$1"      # video ID
  local start_sec="$2"      # start time in seconds
  local segment_idx="$3"    # global segment counter

  mp4_path="$VIDEO_DIR/${clip_name}_${start_sec}.mp4"
  wav_path="$AUDIO_DIR/${clip_name}_${start_sec}.wav"

  # skip if already exists
  [[ -e $mp4_path ]] && return

  # Calculate which cookie file to use:
  #   group = (segment_idx - 1) / SEGMENTS_PER_COOKIE
  #   then cycle through COOKIE_COUNT files
#   cookie_index=$(( ((segment_idx - 1) / SEGMENTS_PER_COOKIE) % COOKIE_COUNT + 1 ))
  cookie_index=$(( (segment_idx - 1) % COOKIE_COUNT + 1 ))

  cookie_file="${COOKIE_PREFIX}${cookie_index}.txt"
  COOKIE_OPTS="--cookies $cookie_file"


  # Extract direct URL ≤360p
  url=$(yt-dlp $COOKIE_OPTS \
    --extractor-args "youtube:player-client=default,mweb;po_token=$PO_TOKEN" \
    --no-warnings --ignore-errors \
    -t sleep \
    -f "best[height<=360][ext=mp4]/best[height<=360]/best" \
    -g "https://youtu.be/$clip_name" \
    2>>"$LOG_FILE" || echo "")

  if [[ -z $url ]]; then
    echo "Failed to get URL for $clip_name" >>"$LOG_FILE"
    return
  fi

  # Download video segment + audio
  ffmpeg -loglevel error \
    -ss "$start_sec" -i "$url" \
    -map 0:v -c:v h264_videotoolbox -b:v 1800k -t "$DURATION" -an "$mp4_path" \
    -map 0:a -c:a pcm_s16le -ar 16000 -ac 1 -t "$DURATION" "$wav_path"

}

export -f process_line
export DURATION VIDEO_DIR AUDIO_DIR LOG_FILE COOKIE_COUNT COOKIE_PREFIX SEGMENTS_PER_COOKIE

# feed clip names into parallel
awk -F, 'NR>1 && $1!~/^[[:space:]]*$/ { print $1, $2, ++i }' "$CSV_FILE" \
  | parallel --bar --colsep ' ' -j "$JOBS" process_line {1} {2} {3}

echo "All segments are available in $OUT_DIR"
