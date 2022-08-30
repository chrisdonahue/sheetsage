#!/bin/bash

# NOTE: Override this with a local directory of your choosing
SHEETSAGE_CACHE_DIR=$(python -c "import pathlib; print(pathlib.Path(pathlib.Path.home(), '.sheetsage').resolve())")
mkdir -p $SHEETSAGE_CACHE_DIR

DOCKER_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    # Positional argument
    *)
      if [ -f "$1" ]; then
        # Input is local file that we need to mount on the container
        # TODO: Handle file paths with spaces?
        echo "Copying input file $1 to container as ./output/input"
        cp $1 ./output/input
        DOCKER_ARGS+=("./output/input")
      else
        # Input is URL or another positional argument
        DOCKER_ARGS+=("$1")
      fi
      shift
      ;;
    # Optional argument
    -*|--*)
      DOCKER_ARGS+=("$1")
      DOCKER_ARGS+=("$2")
      shift
      shift
      ;;
  esac
done

echo "Running Sheet Sage via Docker with args: ${DOCKER_ARGS[@]}"

docker run \
  -it \
  --rm \
  -v $(pwd)/sheetsage:/sheetsage/sheetsage \
  -v $SHEETSAGE_CACHE_DIR:/sheetsage/cache \
  -v $(pwd)/output:/sheetsage/output \
  chrisdonahue/sheetsage \
  python -m sheetsage.infer ${DOCKER_ARGS[@]}
