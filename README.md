# UAV-Close-Formation-Flight-Sim

terminal command to run 1 plane

```
cd ardupilot
./Tools/autotest/sim_vehicle.py   -v ArduPlane   --model jsbsim   --aircraft Rascal110   -I0   --sysid=1   --out=udp:127.0.0.1:14550   --map --enable-fgview
```

cmd to run Flight Gear visualisation connection

```
fgfs   --fg-aircraft=/home/johnnie/ardupilot/Tools/autotest/aircraft   --aircraft=Rascal110-JSBSim   --native-fdm=socket,in,10,,5503,udp   --fdm=external   --airport=KSFO   --fg-root=/usr/share/games/flightgear
```
