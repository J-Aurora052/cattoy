# renderer.py
import torch, os, imageio, sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import raw2alpha, TensorVMSplit, AlphaGridMask
from models.tensorBase import raw2alpha, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False,
                                device='cuda'):
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray,
                                     N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)


    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None


import imageio
import subprocess
import tempfile
import shutil


def robust_video_save(frames, output_path, fps=30, is_depth=False):
    """健壮的视频保存函数，支持自动 resize 与多级回退"""
    def resize_to_multiple_of_16(frame):
        h, w = frame.shape[:2]
        new_h = (h + 15) // 16 * 16
        new_w = (w + 15) // 16 * 16
        if (new_h, new_w) != (h, w):
            from PIL import Image
            pil_img = Image.fromarray(frame)
            pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
            return np.array(pil_img)
        return frame

    try:
        temp_dir = tempfile.mkdtemp()
        for i, frame in enumerate(frames):
            frame = resize_to_multiple_of_16(frame)
            if is_depth:
                frame = (frame * 255).astype('uint8') if frame.dtype == np.float32 else frame
            imageio.imwrite(f"{temp_dir}/frame_{i:04d}.png", frame)

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', f'{temp_dir}/frame_%04d.png',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '23' if not is_depth else '28',
            output_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        return True

    except Exception as e1:
        print(f"[FFmpeg 失败] {e1}")

        try:
            writer = imageio.get_writer(output_path, format='FFMPEG', fps=fps,
                                        codec='libx264',
                                        quality=8 if not is_depth else 5,
                                        macro_block_size=None)
            for frame in frames:
                frame = resize_to_multiple_of_16(frame)
                writer.append_data(frame)
            writer.close()
            return True
        except Exception as e2:
            print(f"[ImageIO-FFmpeg 失败] {e2}")

        try:
            gif_path = output_path.replace('.mp4', '.gif')
            imageio.mimsave(gif_path, frames, duration=1000 / fps)
            print(f"[GIF 回退成功] 保存为: {gif_path}")
            return True
        except Exception as e3:
            print(f"[GIF 保存失败] {e3}")

        try:
            seq_dir = output_path + '_frames'
            os.makedirs(seq_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                imageio.imwrite(f"{seq_dir}/frame_{i:04d}.png", frame)
            print(f"[最终回退] 保存为图像序列至: {seq_dir}")
            return False
        except Exception as e4:
            print(f"[图像序列失败] {e4}")
            return False

    finally:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)



@torch.no_grad()
def evaluation(test_dataset, tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                               ndc_ray=ndc_ray, white_bg=white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)
    if len(rgb_maps) > 0:
        robust_video_save(rgb_maps, f'{savePath}/{prtx}video.mp4', fps=30)
    if len(depth_maps) > 0:
        # 深度图需要特殊处理
        depth_frames = [(d * 255).astype('uint8') for d in depth_maps]
        robust_video_save(depth_frames, f'{savePath}/{prtx}depthvideo.mp4', fps=30, is_depth=True)




    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs


@torch.no_grad()
def evaluation_path(test_dataset, tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                               ndc_ray=ndc_ray, white_bg=white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    if len(rgb_maps) > 0:
        robust_video_save(rgb_maps, f'{savePath}/{prtx}path_video.mp4', fps=30)
    if len(depth_maps) > 0:
        depth_frames = [(d * 255).astype('uint8') for d in depth_maps]
        robust_video_save(depth_frames, f'{savePath}/{prtx}path_depthvideo.mp4', fps=30, is_depth=True)


    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs
