import os
import numpy as np
import torch
import nibabel as nib
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.data.dataset import MyTestset
from torch.utils.data import DataLoader
from einops import rearrange, repeat

def maybe_mkdirs(dirs):
    if os.path.exists(dirs) == False:
        os.makedirs(dirs)

def save_nifti(data, subject_id, results_dir, target):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    data = rearrange(data, 'h w d -> d h w')
    data = data[::-1, ::-1, :]
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, os.path.join(results_dir, f"{subject_id}_{target}.nii.gz"))

def load_state_dict_with_shape(model, checkpoint_path):
    saved_state_dict = load_state_dict(resume_path, location='cpu')
    model_state_dict = model.state_dict()

    for name, param in saved_state_dict.items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)

    model.load_state_dict(model_state_dict, strict=False)


model = create_model('./models/cldm_v21.yaml').cpu()
resume_path = "./models/control_v11p_sd21_canny.ckpt"
load_state_dict_with_shape(model, resume_path)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

data_path = "BraTS2021/ValidationData"
src = "T1"
tgt = "T2"
prompt = f"Brain MR image,{tgt} modality,Denoising, "

dataset = MyTestset(data_path, src, tgt)
dataloader = DataLoader(dataset, num_workers=8, batch_size=1, shuffle=False)

results_dir = os.path.join("results", "Denoising")  # save all the results to this directory
maybe_mkdirs(results_dir)

ddim_steps = 50
num_samples = 1
eta = 0
scale = 1

# model.eval()
for i, data in enumerate(dataloader):
    subject_id = data["subject_id"][0]
    canny = data["hint"]
    source = data["source"]
    dtxt = data["dtxt"]

    print(f"Processing {subject_id}...")

    source = rearrange(source, 'b c h w d -> (b d) c h w')
    canny = rearrange(canny, 'b c h w d -> (b d) c h w')
    
    d, c, h, w = source.shape
    bs = np.array([*range(0, d, 8)] + [d])
    output = torch.zeros((d, 3, h, w))
    noise = torch.randn((4, h//8, w//8)).cuda()

    for i in range(len(bs)-1):
        shape = output.shape[1:]
        N = bs[i+1] - bs[i]

        source_i = source[bs[i]:bs[i+1]].to(torch.float32).cuda()
        canny_i = canny[bs[i]:bs[i+1]].to(torch.float32).cuda()

        dtxt = data["dtxt"]
        dtxt = dtxt * N
        prompts = [prompt] * N
        dtxt = model.get_learned_conditioning(dtxt).cuda()

        cond = dict(c_concat=[canny_i], c_crossattn=[prompts],  dtxt=[dtxt], c_source=[source_i])

        uc_cross = model.get_unconditional_conditioning(N).cuda()
        uc_cat = torch.zeros_like(canny_i).to(torch.float32).cuda()
        un_cond = {"c_concat": [uc_cat], "c_crossattn": [[""] * N], "dtxt": [uc_cross], "c_source": [uc_cat]}

        ts = torch.full((N,), ddim_sampler.ddpm_num_timesteps-1, dtype=torch.long).cuda()
        encoder_posterior = model.encode_first_stage(cond["c_concat"][0])
        z = model.get_first_stage_encoding(encoder_posterior).detach()
        noise_i = repeat(noise, 'c h w -> N c h w', N=N)
        x_T = model.q_sample(z, t=ts, noise=noise_i)
        
        # uncoditional guidance
        samples, intermediates = ddim_sampler.sample(ddim_steps, N,
                                                     shape, cond, verbose=False, eta=eta, x_T=x_T,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond, )
        
        samples = model.decode_first_stage(samples)
        samples = samples.detach().cpu()
        output[bs[i]:bs[i+1]] = samples

    output = rearrange(output, 'd c h w -> h d w c')
    output = np.dot(output, [0.2989, 0.5870, 0.1140])
    subject_dir = os.path.join(results_dir, subject_id)
    maybe_mkdirs(subject_dir)
    save_nifti(output, subject_id, subject_dir, tgt)