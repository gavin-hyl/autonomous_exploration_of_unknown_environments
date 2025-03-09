import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/tkleeneuron/cs133b-final-project/install/uav_controller_node'
