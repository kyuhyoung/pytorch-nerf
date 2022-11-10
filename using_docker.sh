docker_name=u_20_py_3_8_torch_1_12_1_cu_11_3
dir_cur=${PWD##*/}
##################################################################################################
cd docker_file/; sudo docker build --force-rm --shm-size=64g -t ${docker_name} -f Dockerfile_${docker_name} .; cd -
#: << 'END'
sudo docker run --gpus '"device=1"' --rm -it --shm-size=64g -e DISPLAY=$DISPLAY -w /workspace/${dir_cur} -v $PWD:/workspace/${dir_cur} -v $HOME/.Xauthority:/root/.Xauthority:rw --net=host ${docker_name} fish
#END
