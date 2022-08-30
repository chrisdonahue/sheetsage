#!/bin/bash

# NOTE: Override this with a local directory of your choosing
SHEETSAGE_CACHE_DIR=$(python -c "import pathlib; print(pathlib.Path(pathlib.Path.home(), '.sheetsage').resolve())")
mkdir -p $SHEETSAGE_CACHE_DIR

while [[ $# -gt 0 ]]; do
  case $1 in
    -j|--use_jukebox)
      JUKEBOX_CMD="&& python -m sheetsage.assets SHEETSAGE_V02_JUKEBOX && python -m sheetsage.assets JUKEBOX "
      shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

docker run \
  -it \
  --rm \
  -v $(pwd)/sheetsage:/sheetsage/sheetsage \
  -v $SHEETSAGE_CACHE_DIR:/sheetsage/cache \
  chrisdonahue/sheetsage \
  /bin/bash -c \
  "python -m sheetsage.assets SHEETSAGE_V02_HANDCRAFTED ${JUKEBOX_CMD}"
