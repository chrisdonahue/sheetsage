source env.sh

pushd ..
docker build -t ${DOCKER_NAMESPACE}/${DOCKER_TAG} -f docker/Dockerfile .
popd
