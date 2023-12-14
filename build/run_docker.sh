WORKSPACE=/home/sj/Desktop/WENYI/stablediffusion/stablediffusion-scripts

docker run -it --rm \
	--gpus all \
    --shm-size=64g \
	-v $WORKSPACE:/workspace/ \
	sd-scripts:latest