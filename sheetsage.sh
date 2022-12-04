#!/bin/bash

# NOTE: Override these with local directories of your choosing
SHEETSAGE_CACHE_DIR=$(python3 -c "import pathlib; print(pathlib.Path(pathlib.Path.home(), '.sheetsage').resolve())")
SHEETSAGE_OUTPUT_DIR=$(pwd)/output

DOCKER_CPUS=$(python3 -c "import os; cpus=os.sched_getaffinity(0); print(','.join(map(str,cpus)))")
DOCKER_CPU_ARG="--cpuset-cpus ${DOCKER_CPUS}"
DOCKER_GPU_ARG=""

if [ -f "$(pwd)/setup.py" ] && [ -d "$(pwd)/sheetsage" ]; then
  DOCKER_LINK_LIB_ARG="-v $(pwd)/sheetsage:/sheetsage/sheetsage"
else
  DOCKER_LINK_LIB_ARG=""
fi

DOCKER_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    # GPU
    -j|--use_jukebox)
      DOCKER_ARGS+=("$1")
      DOCKER_GPUS=$(nvidia-smi -L | python3 -c "import sys; print(','.join([l.strip().split()[-1][:-1] for l in list(sys.stdin)]))")
      DOCKER_GPU_ARG="--gpus device=${DOCKER_GPUS}"
      shift
      ;;
    # Flags
    --segment_hints_are_downbeats|--skip_melody|--skip_harmony|--legacy_behavior)
      DOCKER_ARGS+=("$1")
      shift
      ;;
    # Optional argument
    -*|--*)
      DOCKER_ARGS+=("$1")
      DOCKER_ARGS+=("$2")
      shift
      shift
      ;;
    # Positional argument
    *)
      if [ -f "$1" ]; then
        # Input is local file that we need to mount on the container
        # TODO: Handle file paths with spaces?
        echo "Copying input file $1 to container as ./output/input"
        cp $1 ${SHEETSAGE_OUTPUT_DIR}/input
        DOCKER_ARGS+=("/sheetsage/output/input")
      else
        # Input is URL or another positional argument
        DOCKER_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

echo "Running Sheet Sage via Docker with args: ${DOCKER_ARGS[@]}"

mkdir -p $SHEETSAGE_CACHE_DIR
mkdir -p $SHEETSAGE_OUTPUT_DIR
docker run \
  -it \
  --rm \
  ${DOCKER_CPU_ARG} \
  ${DOCKER_GPU_ARG} \
  -u $(id -u) \
  ${DOCKER_LINK_LIB_ARG} \
  -v $SHEETSAGE_CACHE_DIR:/sheetsage/cache \
  -v $SHEETSAGE_OUTPUT_DIR:/sheetsage/output \
  chrisdonahue/sheetsage \
  python -m sheetsage.infer ${DOCKER_ARGS[@]}
