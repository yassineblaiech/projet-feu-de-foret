#!/bin/bash
while true
do
    whoami
    echo "$USER" 
    /usr/bin/python3.9 /home/username/Desktop/autoSend.py
    echo "feux_detection"
    sleep 1
done
