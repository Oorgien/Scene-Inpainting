import argparse

from base_model.base_blocks import test_attn
from models.MadfGAN.model import test_blocks as madf_test_blocks
from models.MadfGAN.model import test_model as madfs_test_model


def test(device_id):
    print("MADF model blocks testing")
    assert madf_test_blocks(device_id) == True
    print("MADF blocks OK")

    print("Attention testing")
    assert test_attn(device_id) == True
    print("Attention OK")

    print("MADF model testing")
    assert madfs_test_model(device_id) == True
    print("MADF model OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing blocks and models')
    parser.add_argument('device_id', default='train', type=str,
                        help='GPU id to run test on')
    args = parser.parse_args()
    test(args.device_id)
