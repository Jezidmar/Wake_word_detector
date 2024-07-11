import argparse

import torch


def export_to_onnx(args):
    img_size = (100, 256)
    batch_size = 1
    sample_input = torch.rand((batch_size, 1, *img_size))

    if args.use_version_1:
        from models.se_res2net_1 import se_res2net50_1

        model = se_res2net50_1(num_classes=1, in_channels=1)
    else:
        from models.se_res2net_2 import se_res2net50_2

        model = se_res2net50_2(num_classes=1, in_channels=1)

    state_dict = torch.load(
        args.checkpoint_path
    )  # this should be loaded to CUDA:0 by default and but it does not matter since we transfer it to 'rank' device later on.
    model.load_state_dict(state_dict)

    torch.onnx.export(
        model,
        sample_input,
        args.save_model_path,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to trained model including checkpoint name, for example: /path/to/model/ckpt.pth",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        help="Save path for trained model including checkpoint name, for example: /path/to/save/folder/ckkpt.onnx",
        default=None,
    )
    parser.add_argument(
        "--use_version_1",
        type=bool,
        help="Version 1 is bigger model, same as in paper",
        default=False,
    )

    args = parser.parse_args()
    export_to_onnx(args)