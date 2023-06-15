#!/bin/bash
FILE=/dev/video4
if ! test -e "$FILE"; then
	echo "Creating loopback camera!"
	sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback devices=1 video_nr=4 card_label="Schwurbler"
fi
python3 QtGui.py --device /dev/video4

