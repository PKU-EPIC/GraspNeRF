#!/bin/bash

cd /data/InterNeRF/renderer/renderer_giga_GPU6-0_rand_M

# 830*6
mycount=0;
while (( $mycount < 100 )); do
    /home/xxx/blender-2.93.3-linux-x64/blender material_lib_v2.blend --background -noaudio --python render_pile_STD_rand.py -- $mycount;
((mycount=$mycount+1));
done;