docker build . -t openpi_ur5e:latest -f scripts/docker/serve_policy.Dockerfile
docker run -d --name openpi_ur5e -it --network=host -v .:/app --gpus=all openpi_ur5e:latest /bin/bash
