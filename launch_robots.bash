#!/bin/bash
python scripts/interface_robot_1.py &
python scripts/interface_robot_2.py &
python scripts/interface_robot_3.py &
wait

