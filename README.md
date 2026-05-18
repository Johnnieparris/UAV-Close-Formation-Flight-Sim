# UAV-Close-Formation-Flight-Sim

terminal command to run plane 1

```
cd ardupilot
./Tools/autotest/sim_vehicle.py -v ArduPlane --model jsbsim --aircraft Rascal110 -I0 --sysid=1 --out=udp:127.0.0.1:14550 --map --enable-fgview --out=udp:127.0.0.1:14552
```

terminal Command to run plane 2

```
cd ardupilot
./Tools/autotest/sim_vehicle.py -v ArduPlane --model jsbsim --aircraft Rascal110 -I1 --sysid=2 --out=udp:127.0.0.1:14560 --map --enable-fgview --out=udp:127.0.0.1:14562
```

cmd to run Flight Gear visualisation connection

```
fgfs \
  --fg-aircraft=/home/johnnie/ardupilot/Tools/autotest/aircraft \
  --aircraft=Rascal110-JSBSim \
  --native-fdm=socket,in,10,,5503,udp \
  --fdm=external \
  --airport=KSFO \
  --fg-root=/usr/share/games/flightgear \
  --multiplay=out,10,127.0.0.1,5002 \
  --multiplay=in,10,,5001 \
  --callsign=PLANE1 \
  2>&1 | grep -E -v "AI error|traffic record"
```

cmd to run flight gear plane 2 

```
fgfs \
  --fg-aircraft=/home/johnnie/ardupilot/Tools/autotest/aircraft \
  --aircraft=Rascal110-JSBSim \
  --native-fdm=socket,in,10,,5513,udp \
  --fdm=external \
  --airport=KSFO \
  --fg-root=/usr/share/games/flightgear \
  --multiplay=out,10,127.0.0.1,5001 \
  --multiplay=in,10,,5002 \
  --callsign=PLANE2 \
  2>&1 | grep -E -v "AI error|traffic record"
  ```
