import os
import torch
import sys
from urllib.parse import urlparse
import errno

ENABLE_LAST_RESHAPE = True

MODELS_BACKBONE_WEIGHT = ['mnasnet_050', 'mnasnet_075', 'mnasnet_small_025', 'mnasnet_small_050', 'mnasnet_small_100',
                          'mobilenet_v3_small_010_relu', 'mobilenet_v3_small_025_relu', 'mobilenet_v3_small_050_relu',
                          'semnasnet_050']


def get_ratio(name: str) -> float:
    ratio = int(name[-3:]) / 100.0
    # print 'Using ratio : ' , ratio)
    return ratio


def gcd(a, b):
    while (b != 0):
        temp = a % b
        a = b
        b = temp
    return abs(a)


def load_state_dict_from_url(url, map_location=None, progress=True):
    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        torch.hub.download_url_to_file(url, cached_file, None, progress=progress)

    return torch.load(cached_file, map_location=map_location)
