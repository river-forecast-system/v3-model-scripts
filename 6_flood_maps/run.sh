CPUS=1
MEM="30g"
TIME="00:35:00"
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

module load slurm-auto-array parallel

# Build USA DEM argument list.
ARGFILE=$(mktemp)

for lat in $(seq 24 49); do
    for lon in $(seq 66 125); do

        # Determine the 10x10 FABDEM parent directory.
        south=$(( lat / 10 * 10 ))
        north=$(( south + 10 ))

        west=$(( (lon + 9) / 10 * 10 ))
        east=$(( west - 10 ))

        printf -v dirname \
            "N%02dW%03d-N%02dW%03d_FABDEM_V1-2" \
            "$south" "$west" "$north" "$east"

        printf -v filename \
            "N%02dW%03d_FABDEM_V1-2.tif" \
            "$lat" "$lon"

        dem="${DEM_DIR}/${dirname}/${filename}"

        if [[ -f "$dem" ]]; then
            echo "$dem"
        fi

    done
done > "$ARGFILE"

echo "Found $(wc -l < "$ARGFILE") DEMs"

slurm-auto-array \
    -n "$CPUS" \
    --mem "$MEM" \
    -t "$TIME" \
    -T "$CPUS,0,$MEM,00:40:00" \
    -J usa \
    -- /home/lrr43/scripts/compute.sh \
    :::: "$ARGFILE"

rm "$ARGFILE"