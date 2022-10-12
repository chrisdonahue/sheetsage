source env.sh

docker exec -it ${DOCKER_NAME} python -m sheetsage.serve.backend.main $@
