#!/bin/bash -l

module load ecmwf-toolbox # grib_copy and grib_count

set -euo pipefail

ymd=${1:-$(date -u -d '12 hours ago' +%Y%m%d)}
base=00
nsteps=145 # 91 hourly steps, 18 3-hourly, and 36 6-hourly
nmembers=50
fcdir=${SCRATCH:?names no folder: run this on the ECMWF HPC}/ifs_enfo/$ymd

if [[ ! $ymd =~ ^[0-9]{8}$ ]]; then
  echo "the date of the run must be YYYYMMDD, not $ymd" >&2
  exit 1
fi
mkdir -p "$fcdir"
cd "$fcdir"

#####################
# Retrieve the control forecast, archived as stream=oper, type=fc since IFS Cycle 50r1 (12 May 2026).
# One request for each run of steps, since MARS expands a range such as 0/to/90/by/1 only when it is the whole list.
#####################

mars <<EOF
retrieve,
        date=$ymd,
        time=$base,
        stream=oper,
        expver=1,
        step=0/to/90/by/1,
        levtype=sfc,
        class=od,
        type=fc,
        param=205.128,
        grid=O1280,
        target="cf1.grb"

retrieve,
        date=$ymd,
        time=$base,
        stream=oper,
        expver=1,
        step=93/to/144/by/3,
        levtype=sfc,
        class=od,
        type=fc,
        param=205.128,
        grid=O1280,
        target="cf2.grb"

retrieve,
        date=$ymd,
        time=$base,
        stream=oper,
        expver=1,
        step=150/to/360/by/6,
        levtype=sfc,
        class=od,
        type=fc,
        param=205.128,
        grid=O1280,
        target="cf3.grb"
EOF

cat cf1.grb cf2.grb cf3.grb > cf_00_${ymd}.grib
rm -f cf1.grb cf2.grb cf3.grb

#####################
# Retrieve the 50 perturbed members, stream=enfo, type=pf, then split them into a file each
#####################

mars <<EOF
retrieve,
        date=$ymd,
        time=$base,
        stream=enfo,
        expver=1,
        step=0/to/90/by/1,
        levtype=sfc,
        class=od,
        type=pf,
        number=1/to/$nmembers,
        param=205.128,
        grid=O1280,
        target="pf1.grb"

retrieve,
        date=$ymd,
        time=$base,
        stream=enfo,
        expver=1,
        step=93/to/144/by/3,
        levtype=sfc,
        class=od,
        type=pf,
        number=1/to/$nmembers,
        param=205.128,
        grid=O1280,
        target="pf2.grb"

retrieve,
        date=$ymd,
        time=$base,
        stream=enfo,
        expver=1,
        step=150/to/360/by/6,
        levtype=sfc,
        class=od,
        type=pf,
        number=1/to/$nmembers,
        param=205.128,
        grid=O1280,
        target="pf3.grb"
EOF

# grib_copy cannot zero pad a key in the output name, so each member is renamed to pf_01 ... pf_50 after
grib_copy pf1.grb pf2.grb pf3.grb "pf_[perturbationNumber]_${ymd}.grib.tmp"
rm -f pf1.grb pf2.grb pf3.grb
for ((n = 1; n <= nmembers; n++)); do
  mv "pf_${n}_${ymd}.grib.tmp" "$(printf 'pf_%02d_%s.grib' "$n" "$ymd")"
done

#####################
# Check the control and every member hold each step
#####################

members=(pf_*_${ymd}.grib)
if [[ ${#members[@]} != $nmembers ]]; then
  echo "$fcdir holds ${#members[@]} perturbed members, not $nmembers" >&2
  exit 1
fi
for file in cf_00_${ymd}.grib "${members[@]}"; do
  fields=$(grib_count "$file")
  if [[ $fields != $nsteps ]]; then
    echo "$fcdir/$file holds $fields steps, not $nsteps" >&2
    exit 1
  fi
done
echo "wrote the control and $nmembers perturbed members of the run of $ymd 00 UTC to $fcdir"
