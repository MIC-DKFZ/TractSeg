import sys, os, inspect
from os.path import join
from os.path import expanduser
import getpass
import socket

def get_project_path():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
    return parent_dir

def get_HOME_path():
    pcname = socket.gethostname()
    # UBUNTU
    if pcname == "jakob-ubuntu":
        return join(expanduser("~"), "data")
    # MAC
    elif pcname == "mbimac20.inet.dkfz-heidelberg.de":
        return join(expanduser("~"), "data")
    # OPENSTACK
    elif pcname.startswith("jakob-mrtrix-"):
        return join(expanduser("~"), "data")  # actually not existend on OPENSTACK, because not needed there
    # GPU NODE
    elif pcname.startswith("e132-comp01"):  # e132-comp01.inet.dkfz-heidelberg.de
        # return join("/ssd/data_jakob")
        # return join("/net/e132-comp04/ssd/data_jakob")
        return join("/datasets/data_jakob")
    elif pcname.startswith("e132-comp02"):
        # return join("/ssd/data_jakob")
        # return join("/net/e132-comp04/ssd/data_jakob")
        return join("/datasets/data_jakob")
    elif pcname.startswith("e132-comp03"):
        # return join("/ssd/data_jakob")
        # return join("/net/e132-comp04/ssd/data_jakob")
        return join("/datasets/data_jakob")
    elif pcname.startswith("e132-comp04"):
        # return join("/ssd/data_jakob")
        return join("/datasets/data_jakob")
    elif pcname.startswith("e132-comp05"):
        # return join("/ssd/data_jakob")
        # return join("/net/e132-comp04/ssd/data_jakob")
        return join("/datasets/data_jakob")
    elif pcname.startswith("e132-comp06"):
        # return join("/ssd/data_jakob")
        # return join("/net/e132-comp04/ssd/data_jakob")
        return join("/datasets/data_jakob")
    elif pcname.startswith("e132-comp07"):
        # return join("/ssd/data_jakob")
        # return join("/net/e132-comp04/ssd/data_jakob")
        return join("/datasets/data_jakob")
    elif pcname.startswith("e132-comp08"):
        # return join("/ssd/data_jakob")
        # return join("/net/e132-comp04/ssd/data_jakob")
        return join("/datasets/data_jakob")
    elif pcname.startswith("e132-comp09"):
        # return join("/ssd/data_jakob")
        # return join("/net/e132-comp04/ssd/data_jakob")
        return join("/datasets/data_jakob")
    elif pcname.startswith("e132-comp10"):
        # return join("/ssd/data_jakob")
        # return join("/net/e132-comp04/ssd/data_jakob")
        return join("/datasets/data_jakob")
    elif pcname.startswith("e132-comp11"):
        # return join("/ssd/data_jakob")
        # return join("/net/e132-comp04/ssd/data_jakob")
        return join("/datasets/data_jakob")
    elif pcname.startswith("e132-comp12"):
        # return join("/ssd/data_jakob")
        # return join("/net/e132-comp04/ssd/data_jakob")
        return join("/datasets/data_jakob")
    else:
        print("ERROR: PCNAME NOT FOUND")
        print(pcname)
        sys.exit(2)

def get_NETWORK_DRIVE_path():
    pcname = socket.gethostname()
    # UBUNTU
    if pcname == "jakob-ubuntu":
        return "/mnt/jakob/E130-Personal/Wasserthal"
    # MAC
    elif pcname == "mbimac20.inet.dkfz-heidelberg.de":
        return "/Volumes/E130-Personal/Wasserthal"
    # OPENSTACK
    elif pcname.startswith("jakob-mrtrix-"):
        return "/home/ubuntu/E130-Personal/Wasserthal"
    # GPU NODE
    elif pcname.startswith("e132-comp"):  # e132-comp02.inet.dkfz-heidelberg.de
        return "/ad/wasserth/E130-Personal/Wasserthal"
    else:
        print("ERROR: PCNAME NOT FOUND")
        print(pcname)
        sys.exit(2)

class Config_OLD:
    '''
    Stores configuration variables
    '''
    # class variable shared by all instances:
    pcname = socket.gethostname()

    PROJECT_PATH = get_project_path()
    HOME = get_HOME_path()
    NETWORK_DRIVE = get_NETWORK_DRIVE_path()

    if pcname.startswith("e132-comp"):
        EXP_PATH = join(HOME, "hcp_exp")    #on gpu node store locally (run sync_node afterwards on mac)
        # EXP_PATH = join(NETWORK_DRIVE, "hcp_exp_nodes")
    else:
        EXP_PATH = join(NETWORK_DRIVE, "hcp_exp_nodes")

