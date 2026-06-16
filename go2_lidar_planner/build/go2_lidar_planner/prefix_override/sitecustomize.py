import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/root/CWNU_Unitree_Go2/go2_lidar_planner/install/go2_lidar_planner'
