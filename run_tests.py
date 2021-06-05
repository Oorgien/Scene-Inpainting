import argparse

from base_model.base_blocks import test_attn, test_hypergrashconv
from models.BestModel.model import test_best_model
from models.MadfGAN.model import test_blocks as madf_test_blocks
from models.MadfGAN.model import test_model as madf_test_model


def test(device_id):
    print("Best model testing")
    assert test_best_model(device_id) == True
    print("Best model OK")

    print("Hypergraph conv testing")
    assert test_hypergrashconv(device_id) == True
    print("Hypergraph conv OK")

    print("MADF model blocks testing")
    assert madf_test_blocks(device_id) == True
    print("MADF blocks OK")

    print("Attention testing")
    assert test_attn(device_id) == True
    print("Attention OK")

    print("MADF model testing")
    assert madf_test_model(device_id) == True
    print("MADF model OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing blocks and models')
    parser.add_argument('device_id', default='train', type=str,
                        help='GPU id to run test on')
    args = parser.parse_args()
    test(args.device_id)
