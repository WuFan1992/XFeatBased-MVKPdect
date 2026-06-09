
from validation.hloc.utils.read_colmaps import *


def generate_infomap(imagebin_path):
    images = read_extrinsics_binary(imagebin_path)
    infomap = {}
    for image_id, image_info in images.items():
        infomap[image_info['name']] = {
            'qvec': image_info['qvec'],
            'tvec': image_info['tvec']
        }
    return infomap


def get_pose(ref_name, infomap):
    if ref_name in infomap:
        return infomap[ref_name]['qvec'], infomap[ref_name]['tvec']
    raise ValueError(f"Reference image {ref_name} not found in COLMAP data.")




    