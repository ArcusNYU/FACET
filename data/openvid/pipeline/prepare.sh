cd /nvmedata/workspace2/users/Arcus/Facet

python -m data.openvid.pipeline.filters \
    --csv "/mnt/highspeed/users/Arcus/OPENVID_DATA/OpenHumanVid_part_*.csv" \
    --batch 512 \
    --provider cuda