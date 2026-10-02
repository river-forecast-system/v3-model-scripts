#!/bin/bash -l

module load ecmwf-toolbox # grib_copy and grib_count

set -euo pipefail

# The extended range (sub-seasonal) ensemble, stream=eefo: a control and 100 perturbed members, run every day at
# 00 UTC to 46 days on the TCo319 grid, archived on its native O320 grid. The run is complete late in its day, so the
# default is yesterday's.
ymd=${1:-$(date -u -d '1 day ago' +%Y%m%d)}
base=00
nsteps=185 # 6-hourly steps from 0 to 1104 hours, 46 days
nmembers=100
fcdir=${SCRATCH:?names no folder: run this on the ECMWF HPC}/ifs_eefo/$ymd

if [[ ! $ymd =~ ^[0-9]{8}$ ]]; then
  echo "the date of the run must be YYYYMMDD, not $ymd" >&2
  exit 1
fi
mkdir -p "$fcdir"
cd "$fcdir"

#####################
# Retrieve the control forecast, stream=eefo, type=cf. IFS Cycle 50r1 moved only the medium range control to
# stream=oper, type=fc; the extended range runs its own control, which stays in eefo.
#####################

mars <<EOF
retrieve,
        date=$ymd,
        time=$base,
        stream=eefo,
        expver=1,
        step=0/to/1104/by/6,
        levtype=sfc,
        class=od,
        type=cf,
        param=205.128,
        grid=O320,
        target="cf_000_${ymd}.grib"
EOF

#####################
# Retrieve the 100 perturbed members, stream=eefo, type=pf, then split them into a file each
#####################

mars <<EOF
retrieve,
        date=$ymd,
        time=$base,
        stream=eefo,
        expver=1,
        step=0/to/1104/by/6,
        levtype=sfc,
        class=od,
        type=pf,
        number=1/to/$nmembers,
        param=205.128,
        grid=O320,
        target="pf.grb"
EOF

# grib_copy cannot zero pad a key in the output name, so each member is renamed to pf_001 ... pf_100 after
grib_copy pf.grb "pf_[perturbationNumber]_${ymd}.grib.tmp"
rm -f pf.grb
for ((n = 1; n <= nmembers; n++)); do
  mv "pf_${n}_${ymd}.grib.tmp" "$(printf 'pf_%03d_%s.grib' "$n" "$ymd")"
done

#####################
# Check the control and every member hold each step
#####################

members=(pf_*_${ymd}.grib)
if [[ ${#members[@]} != $nmembers ]]; then
  echo "$fcdir holds ${#members[@]} perturbed members, not $nmembers" >&2
  exit 1
fi
for file in cf_000_${ymd}.grib "${members[@]}"; do
  fields=$(grib_count "$file")
  if [[ $fields != $nsteps ]]; then
    echo "$fcdir/$file holds $fields steps, not $nsteps" >&2
    exit 1
  fi
done
echo "wrote the control and $nmembers perturbed members of the extended range run of $ymd 00 UTC to $fcdir"
