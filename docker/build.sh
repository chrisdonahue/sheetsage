source env.sh

pushd ..
docker build -t ${DOCKER_NAMESPACE}/${DOCKER_TAG}-dev -f docker/Dockerfile .
popd
