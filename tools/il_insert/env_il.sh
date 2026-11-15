# Isolated 2nd sim instance for the IL work (does not touch the main sim on 555xx / DDS domain 1).
export AALTO_INSTANCE=il AALTO_DDS_DOMAIN=9 AALTO_CAM_PORT_BASE=57555 AALTO_SCENE_PORT=57580 AALTO_RGBD_PORT=57560
export AALTO_CLOCK_PORT=57581 AALTO_TELEIMAGER_REQ_PORT=62000 TASK=Aalto-Insert-G129-Dex3-Joint
export AALTO_LOG=/tmp/aalto_sim_il.log PYTHONPATH=/opt/aalto
