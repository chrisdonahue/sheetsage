source env.sh

pushd ..
docker build -t ${DOCKER_NAMESPACE}/${DOCKER_TAG} -f docker/Dockerfile-release .
docker push ${DOCKER_NAMESPACE}/${DOCKER_TAG}
popd
