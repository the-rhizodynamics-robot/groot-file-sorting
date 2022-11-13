#!/usr/bin/env bash

#install requirements
cat requirements.txt | xargs pip install

sudo apt-get update
sudo apt install build-essential

pip install keras-retinanet==0.5.1 --no-cache-dir


#install dependancies for cv2 and pyzbar
apt-get install -y python3-opencv
apt-get install -y libzbar0

#install ffmpeg 
wget https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.2.2-amd64-static.tar.xz
tar xvf ffm*
mv ffm*/ffmpeg /bin
rm -r ffmpeg*





