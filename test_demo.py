import os.path
import logging
import torch
import argparse
import json
import glob
from pprint import pprint
from fvcore.nn import FlopCountAnalysis
from utils.model_summary import get_model_activation, get_model_flops
from utils import utils_logger
from utils import utils_image as util



def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 0:
        from models.team00_SPAN import SPAN
        name, data_range = f"{model_id:02}_SPAN", 1.0
        model = SPAN(3, 3, upscale=4, feature_channels=28).eval().to(device)
        model_path = os.path.join('model_zoo', f'team00_SPAN.pth')
        stat_dict = torch.load(model_path)
        model.load_state_dict(stat_dict, strict=False)
    elif model_id == 1:
        import importlib
        model_module = importlib.import_module(f'models.team01_PDS')
        name, data_range = f"{model_id:02}_PDS", 1.0
        model = getattr(model_module, f'PDS')().eval().to(device)
        model_path = os.path.join('model_zoo', f'team01_PDS.pth')
        stat_dict = torch.load(model_path)
        model.load_state_dict(stat_dict, strict=True)
    elif model_id == 4:
        from models.team04_ZenoSR import ZenoSR
        name, data_range = f"{model_id:02}_ZenoSR", 1.0
        model_path = os.path.join('model_zoo', 'team04_ZenoSR.pth')
        model = ZenoSR()
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and ('params_ema' in ckpt or 'params' in ckpt):
            state = ckpt.get('params_ema', ckpt.get('params'))
        else:
            state = ckpt
        if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', '', 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    elif model_id == 5:
        from models.team05_1_4mamba_single_arch import SPANMamba_single_1_4T05
        name, data_range = f"{model_id:02}_SPANMamba_single_1_4T05", 1.0  # You can choose either 1.0 or 255.0 based on your own model
        model_path = os.path.join('model_zoo', 'team05_1_4mamba_single_arch.pth')
        model = SPANMamba_single_1_4T05()
        ckpt = torch.load(model_path, map_location=device)
        if 'params_ema' in ckpt:
            state_dict = ckpt['params_ema']
        elif 'params' in ckpt:
            state_dict = ckpt['params']
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=True)
    elif model_id == 6:
        from models.team06_HAESR import HAESR
        name, data_range = f"{model_id:02}_HAESR", 1.0
        model = HAESR(num_feat=48, upsampling=4, window_size=8, res_num=3, block_num=1, bias=True, ffn_bias=True,
                      pe=True).eval().to(device)
        model_path = os.path.join('model_zoo', f'team06_HAESR.pth')
        # stat_dict = torch.load(model_path)['params']
        # model.load_state_dict(stat_dict, strict=False)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['params_ema'], strict=True)
    elif model_id == 9:
        from models.team09_RFDN_SPAN import RFDN_SPAN
        name, data_range = f"{model_id:03}_RFDN_SPAN", 1.0
        model_path = os.path.join('model_zoo', f'team09_RFDN_SPAN.pth')
        model = RFDN_SPAN(in_nc=3, nf=46, num_modules=4, out_nc=3, upscale=4)
        model.load_state_dict(torch.load(model_path)["params_ema"], strict=True)
    elif model_id == 10:
        from models.team10_HFENet import HFENet
        name, data_range = f"{model_id:02}_HFENet", 1.0
        model_path = os.path.join('model_zoo', 'team10_HFENet.pth')
        model = HFENet()
        state_dict = torch.load(
            model_path,
            map_location='cpu')
        if 'params' in state_dict:
            state_dict = state_dict['params']
        elif 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        model.load_state_dict(state_dict, strict=True)  # 建议先用 strict=True 检查是否完全匹配
        model.eval()
    elif model_id == 11:
        from models.team11_VSCINet import VSCINet
        name, data_range = f"{model_id:11}_VSCINet", 1.0
        model_path = os.path.join('model_zoo', 'team11_VSCINet.pth')
        model = VSCINet()
        model.load_state_dict(torch.load(model_path)['params_ema'], strict=True)
    elif model_id == 12:
        from models.team12_DWMamba import DWMamba
        name, data_range = f"{model_id:02}_DWMamba", 1.0
        model = DWMamba(
            upscale=4,
            in_chans=3,
            img_range=1.0,
            img_size=64,
            embed_dim=48,
            d_state=8,
            depths=[2, 2, 2, 2],
            num_heads=[4, 4, 4, 4],
            window_size=16,
            inner_rank=32,
            num_tokens=64,
            convffn_kernel_size=5,
            mlp_ratio=2.0,
            upsampler='pixelshuffledirect',
            resi_connection='1conv').eval().to(device)
        model_path = os.path.join('model_zoo', f'team12_DWMamba.pth')
        stat_dict = torch.load(model_path)
        if 'params' in stat_dict:
            weight_dict = stat_dict['params']
        elif 'params_ema' in stat_dict:
            weight_dict = stat_dict['params_ema']
        else:
            weight_dict = stat_dict
        model.load_state_dict(weight_dict, strict=True)
    elif model_id == 16:
        from models.team16_PKDSR import SPANFPrunedKD
        name, data_range = f"{model_id:02}_PKDSR_FINAL_CKPT", 1.0
        model = SPANFPrunedKD(3, 3, upscale=4, tail_channels=24, feature_channels=32).eval().to(device)
        model_path = os.path.join('model_zoo', f'team16_PKDSR.pth')
        stat_dict = torch.load(model_path, map_location=torch.device("cpu"))['params_ema']
        model.load_state_dict(stat_dict, strict=True)
    elif model_id == 15:
        from models.team15_DSCF_Fused import DSCF_Fused
        name, data_range = f"{model_id:02}_DSCF_Fused", 1.0
        model_path = os.path.join('model_zoo', 'team15_DSCF_Fused.pth')
        model = DSCF_Fused(num_in_ch=3, num_out_ch=3, feature_channels=26, upscale=4)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 17:
        from models.team17_AMCANet import AMCANet
        name, data_range = f"{model_id:02}_AMCANet", 1.0
        model_path = os.path.join('model_zoo', f'team17_AMCANet.pth')
        model = AMCANet(in_nc=3, out_nc=3, dim=32, n_blocks=7, upscaling_factor=4, num_heads=2).eval().to(device)
        stat_dict = torch.load(model_path)['params_ema']
        model.load_state_dict(stat_dict, strict=True)
    elif model_id == 18:
        from models.team18_DISP import DISP
        name, data_range = f"{model_id:02}_DISP", 1.0
        model_path = os.path.join('model_zoo', 'team18_DISP.pth')
        model = DISP()
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        model = model.to(device)
        inp = torch.randn(1, 3, 256, 256).to(device)
        for i in range(20):
            out = model(inp)
    elif model_id == 19:
        from models.team19_BVIESR import BVI_SRF
        name, data_range = f"{model_id:02}_BVIESR", 1.0  # You can choose either 1.0 or 255.0 based on your own model
        model_path = os.path.join('model_zoo', 'team19_BVIESR.pth')
        model = BVI_SRF()
        model.load_state_dict(torch.load(model_path)['params'], strict=True)
        model = model.cuda()
        model.eval()
    elif model_id == 20:
        from models.team20_ERRN2 import ERRN2
        name, data_range = f"{model_id:02}_ERRN2", 1.0
        model_path = os.path.join('model_zoo', 'team20_ERRN2.pth')
        model = ERRN2(in_channels=3, out_channels=3, feature_channels=32, upscale=4)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 21:
        # Load model
        from models.team21_SAFMN_Deep15 import SAFMN_Deep15
        model_path = f"./model_zoo/team21_SAFMN_Deep15.pth"
        name, data_range = f"{model_id:02}_SAFMN", 1.0
        model = SAFMN_Deep15(num_in_ch=3, num_out_ch=3, dim=40, num_blocks=15, upscale=4)
        state_dict = torch.load(model_path, map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()
    elif model_id == 22:
        # Team 22: SPANV2_ESR
        import sys, subprocess
        span_attn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'span_attention_op')
        span_attn_dir = os.path.normpath(span_attn_dir)
        if span_attn_dir not in sys.path:
            sys.path.insert(0, span_attn_dir)
        # Auto-build span_attention if not installed
        try:
            import span_attention
        except ImportError:
            print("[team22] span_attention not found, building from source ...")
            build_result = subprocess.run(
                [sys.executable, 'setup.py', 'build_ext', '--inplace'],
                cwd=span_attn_dir,
                capture_output=True, text=True
            )
            if build_result.returncode != 0:
                raise RuntimeError(
                    f"[team22] Failed to build span_attention:\n"
                    f"{build_result.stdout}\n{build_result.stderr}"
                )
            print("[team22] span_attention built successfully.")
            import span_attention  # noqa: F811
        from models.team22_SPANV2_ESR import SPANV2_ESR
        name, data_range = f"{model_id:02}_SPANV2_ESR_C2", 1.0
        model_path = os.path.join('model_zoo', 'team22_spanv2_c2.pth')
        model = SPANV2_ESR(3, 3, feature_channels=32, upscale=4, bias=False, use_span_attn=True)
        state = torch.load(model_path, map_location='cpu')
        for key in ['model', 'state_dict', 'params', 'params_ema']:
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        model.load_state_dict(state, strict=True)
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    # print(model)
    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model, name, data_range, tile


def select_dataset(data_dir, mode):
    # inference on the DIV2K_LSDIR_test set
    if mode == "test":
        path = [
            (
                p.replace("_HR", "_LR").replace(".png", "x4.png"),
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "DIV2K_LSDIR_test_HR/*.png")))
        ]

    # inference on the DIV2K_LSDIR_valid set
    elif mode == "valid":
        path = [
            (
                p.replace("_HR", "_LR").replace(".png", "x4.png"),
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "DIV2K_LSDIR_valid_HR/*.png")))
        ]
    else:
        raise NotImplementedError(f"{mode} is not implemented in select_dataset")

    return path


def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


def run(model, model_name, data_range, tile, logger, device, args, mode="test"):
    sf = 4
    border = sf
    results = dict()
    results[f"{mode}_runtime"] = []
    results[f"{mode}_psnr"] = []
    if args.ssim:
        results[f"{mode}_ssim"] = []
    # results[f"{mode}_psnr_y"] = []
    # results[f"{mode}_ssim_y"] = []

    # --------------------------------
    # dataset path
    # --------------------------------
    data_path = select_dataset(args.data_dir, mode)
    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i, (img_lr, img_hr) in enumerate(data_path):

        # --------------------------------
        # (1) img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_hr))
        img_lr = util.imread_uint(img_lr, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)

        # --------------------------------
        # (2) img_sr
        # --------------------------------
        start.record()
        img_sr = forward(img_lr, model, tile)
        end.record()
        torch.cuda.synchronize()
        results[f"{mode}_runtime"].append(start.elapsed_time(end))  # milliseconds
        img_sr = util.tensor2uint(img_sr, data_range)

        # --------------------------------
        # (3) img_hr
        # --------------------------------
        img_hr = util.imread_uint(img_hr, n_channels=3)
        img_hr = img_hr.squeeze()
        img_hr = util.modcrop(img_hr, sf)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        # print(img_sr.shape, img_hr.shape)
        psnr = util.calculate_psnr(img_sr, img_hr, border=border)
        results[f"{mode}_psnr"].append(psnr)

        if args.ssim:
            ssim = util.calculate_ssim(img_sr, img_hr, border=border)
            results[f"{mode}_ssim"].append(ssim)
            logger.info("{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.".format(img_name + ext, psnr, ssim))
        else:
            logger.info("{:s} - PSNR: {:.2f} dB".format(img_name + ext, psnr))

        # if np.ndim(img_hr) == 3:  # RGB image
        #     img_sr_y = util.rgb2ycbcr(img_sr, only_y=True)
        #     img_hr_y = util.rgb2ycbcr(img_hr, only_y=True)
        #     psnr_y = util.calculate_psnr(img_sr_y, img_hr_y, border=border)
        #     ssim_y = util.calculate_ssim(img_sr_y, img_hr_y, border=border)
        #     results[f"{mode}_psnr_y"].append(psnr_y)
        #     results[f"{mode}_ssim_y"].append(ssim_y)
        # print(os.path.join(save_path, img_name+ext))

        # --- Save Restored Images ---
        # util.imsave(img_sr, os.path.join(save_path, img_name+ext))

    results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"])  # / 1000.0
    results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])
    if args.ssim:
        results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])
    # results[f"{mode}_ave_psnr_y"] = sum(results[f"{mode}_psnr_y"]) / len(results[f"{mode}_psnr_y"])
    # results[f"{mode}_ave_ssim_y"] = sum(results[f"{mode}_ssim_y"]) / len(results[f"{mode}_ssim_y"])
    logger.info("{:>16s} : {:<.3f} [M]".format("Max Memory", results[f"{mode}_memory"]))  # Memery
    logger.info("------> Average runtime of ({}) is : {:.6f} milliseconds".format("test" if mode == "test" else "valid",
                                                                                  results[f"{mode}_ave_runtime"]))
    logger.info("------> Average PSNR of ({}) is : {:.6f} dB".format("test" if mode == "test" else "valid",
                                                                     results[f"{mode}_ave_psnr"]))

    return results


def main(args):
    utils_logger.logger_info("NTIRE2026-EfficientSR", log_path="NTIRE2026-EfficientSR.log")
    logger = logging.getLogger("NTIRE2026-EfficientSR")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model, model_name, data_range, tile = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if True:
        # --------------------------------
        # restore image
        # --------------------------------

        # inference on the DIV2K_LSDIR_valid set
        valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="valid")
        # record PSNR, runtime
        results[model_name] = valid_results

        # inference conducted by the Organizer on DIV2K_LSDIR_test set
        if args.include_test:
            test_results = run(model, model_name, data_range, tile, logger, device, args, mode="test")
            results[model_name].update(test_results)

        input_dim = (3, 256, 256)  # set the input dimension
        activations, num_conv = get_model_activation(model, input_dim)
        activations = activations / 10 ** 6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
        logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

        # The FLOPs calculation in previous NTIRE_ESR Challenge
        # flops = get_model_flops(model, input_dim, False)
        # flops = flops/10**9
        # logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        # fvcore is used in NTIRE2026_ESR for FLOPs calculation
        input_fake = torch.rand(1, 3, 256, 256).to(device)
        flops = FlopCountAnalysis(model, input_fake).total()
        flops = flops / 10 ** 9
        logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        num_parameters = num_parameters / 10 ** 6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
        results[model_name].update(
            {"activations": activations, "num_conv": num_conv, "flops": flops, "num_parameters": num_parameters})

        with open(json_dir, "w") as f:
            json.dump(results, f)
    if args.include_test:
        fmt = "{:20s}\t{:10s}\t{:10s}\t{:14s}\t{:14s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Test PSNR", "Val Time [ms]", "Test Time [ms]", "Ave Time [ms]",
                       "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    else:
        fmt = "{:20s}\t{:10s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Val Time [ms]", "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    for k, v in results.items():
        val_psnr = f"{v['valid_ave_psnr']:2.2f}"
        val_time = f"{v['valid_ave_runtime']:3.2f}"
        mem = f"{v['valid_memory']:2.2f}"

        num_param = f"{v['num_parameters']:2.3f}"
        flops = f"{v['flops']:2.2f}"
        acts = f"{v['activations']:2.2f}"
        conv = f"{v['num_conv']:4d}"
        if args.include_test:
            # from IPython import embed; embed()
            test_psnr = f"{v['test_ave_psnr']:2.2f}"
            test_time = f"{v['test_ave_runtime']:3.2f}"
            ave_time = f"{(v['valid_ave_runtime'] + v['test_ave_runtime']) / 2:3.2f}"
            s += fmt.format(k, val_psnr, test_psnr, val_time, test_time, ave_time, num_param, flops, acts, mem, conv)
        else:
            s += fmt.format(k, val_psnr, val_time, num_param, flops, acts, mem, conv)
    with open(os.path.join(os.getcwd(), 'results.txt'), "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2026-EfficientSR")
    parser.add_argument("--data_dir", default="../", type=str)
    parser.add_argument("--save_dir", default="../results", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--include_test", action="store_true", help="Inference on the `DIV2K_LSDIR_test` set")
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")

    args = parser.parse_args()
    pprint(args)

    main(args)
