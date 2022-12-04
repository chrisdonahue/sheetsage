#!/bin/bash

# NOTE: Override this with a local directory of your choosing
SHEETSAGE_CACHE_DIR=$(python3 -c "import pathlib; print(pathlib.Path(pathlib.Path.home(), '.sheetsage').resolve())")

if [ -f "$(pwd)/setup.py" ] && [ -d "$(pwd)/sheetsage" ]; then
  DOCKER_LINK_LIB_ARG="-v $(pwd)/sheetsage:/sheetsage/sheetsage"
else
  DOCKER_LINK_LIB_ARG=""
fi

while [[ $# -gt 0 ]]; do
  case $1 in
    -j|--use_jukebox)
      JUKEBOX_CMD="&& python -m sheetsage.assets SHEETSAGE_V02_JUKEBOX && python -m sheetsage.assets JUKEBOX "
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

mkdir -p $SHEETSAGE_CACHE_DIR
docker run \
  -it \
  --rm \
  -u $(id -u) \
  ${DOCKER_LINK_LIB_ARG} \
  -v $SHEETSAGE_CACHE_DIR:/sheetsage/cache \
  chrisdonahue/sheetsage \
  /bin/bash -c \
  "python -m sheetsage.assets SHEETSAGE_V02_HANDCRAFTED ${JUKEBOX_CMD}"
