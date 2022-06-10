source env.sh

docker exec \
  -it \
  ${DOCKER_NAME} \
  jupyter notebook \
    --ip=0.0.0.0 \
    --port 8888 \
    --no-browser \
    --allow-root \
    --notebook-dir=/sheetsage/notebooks
