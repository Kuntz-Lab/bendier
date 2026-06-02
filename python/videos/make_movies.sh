# Take all PNG frames and turn them into high quality MP4s
make_vid () {
    local input_dir="$1"
    local output_base
    output_base=$(basename "$input_dir")

    ffmpeg -framerate 30 -i "$input_dir/%d.png" \
        -vf "scale=1280:-2" \
        -c:v libx264 -preset veryslow -crf 16 \
        -pix_fmt yuv420p -profile:v high -level 4.2 \
        -movflags +faststart "${output_base}.mp4"

    # ffmpeg -framerate 30 -i "$input_dir/%d.png" \
    #     -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le "${output_base}.mov"

    # ffmpeg -framerate 30 -i "$input_dir/%d.png" \
    #     "${output_base}.webm"

}

make_vid frames/both_ends_clamped