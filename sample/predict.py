import os
import subprocess
from typing import Any, List, Optional
from argparse import Namespace

import torch
from cog import BasePredictor, Input, Path, BaseModel

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from utils.sampler_util import ClassifierFreeSampleModel
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from visualize.motions2hik import motions2hik
from sample.generate import construct_template_variables

"""
In case of matplot lib issues it may be needed to delete model/data_loaders/humanml/utils/plot_script.py" in lines 89~92 as
suggested in https://github.com/GuyTevet/motion-diffusion-model/issues/6
"""


class ModelOutput(BaseModel):
    json_file: Optional[Any]
    animation: Optional[List[Path]]


def get_args():
    args = Namespace()
    args.fps = 30  # 초당 프레임 수(Frames Per Second), 생성되는 모션의 시간 해상도

    args.model_path = './save/my_humanml_trans_enc_512/model000475000.pt'  # 사용할 학습된 모델 가중치(.pt) 파일 경로

    args.guidance_param = 2.5  # Classifier-Free Guidance scale, 1보다 크면 텍스트 조건을 더 강하게 반영
    args.unconstrained = False  # 무조건 샘플링 여부(일반적으로 False, 조건부 생성)

    args.dataset = 'humanml'  # 사용할 데이터셋 이름(HumanML3D)

    args.cond_mask_prob = 1  # 조건 마스킹 확률(1이면 항상 조건 사용)
    args.emb_trans_dec = False  # Transformer decoder 사용 여부(일반적으로 encoder만 사용)
    args.latent_dim = 512  # 모델의 잠재 공간(latent space) 차원 수
    args.layers = 8  # 트랜스포머 레이어 개수.
    args.arch = 'trans_enc'  # 모델 아키텍처(트랜스포머 인코더)

    args.noise_schedule = 'cosine'  # 확산모델의 노이즈 스케줄 종류
    args.sigma_small = True  # 작은 시그마 사용 여부(노이즈 관련 하이퍼파라미터)
    args.lambda_vel = 0.0  # 속도(velocity) 손실 가중치
    args.lambda_rcxyz = 0.0  # root-centered xyz 손실 가중치
    args.lambda_fc = 0.0  # frame consistency 손실 가중치

    args.diffusion_steps = 1000  # 확산모델의 스텝 수(샘플링 속도와 품질에 영향)
    args.pos_embed_max_len = 5000  # 포지셔널 임베딩의 최대 길이(프레임 수 제한)
    args.mask_frames = False  # 프레임 마스킹 사용 여부
    args.text_encoder_type = 'clip'  # 텍스트 인코더 종류(CLIP 사용)
    args.num_repetitions = 196  # 샘플링 반복 횟수(여러 개의 샘플 생성)
    
    return args


class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["mkdir", "/root/.cache/clip"])
        subprocess.run(["cp", "-r", "ViT-B-32.pt", "/root/.cache/clip"])

        self.args = get_args()
        self.num_frames = self.args.fps * 6
        print('Loading dataset...')

        # temporary data
        self.data = get_dataset_loader(name=self.args.dataset,
                                  batch_size=1,
                                  num_frames=196,
                                  split='test',
                                  hml_mode='text_only')

        self.data.fixed_length = float(self.num_frames)

        print("Creating model and diffusion...")
        self.model, self.diffusion = create_model_and_diffusion(self.args, self.data)

        print(f"Loading checkpoints from... {self.args.model_path}")
        state_dict = torch.load(self.args.model_path, map_location='cpu')
        load_model_wo_clip(self.model, state_dict)

        if self.args.guidance_param != 1:
           self.model = ClassifierFreeSampleModel(self.model)   # wrapping model with the classifier-free sampler
        self.model.to(dist_util.dev())
        self.model.eval()  # disable random masking

    def predict(
            self,
            prompt: str = Input(default="the person walked forward and is picking up his toolbox."),
            num_repetitions: int = Input(default=3, description="How many"),
            output_format: str = Input(
                description='Choose the format of the output, either an animation or a json file of the animation data.\
                The json format is: {"thetas": [...], "root_translation": [...], "joint_map": [...]}, where "thetas" \
                is an [nframes x njoints x 3] array of joint rotations in degrees, "root_translation" is an [nframes x 3] \
                array of (X, Y, Z) positions of the root, and "joint_map" is a list mapping the SMPL joint index to the\
                corresponding HumanIK joint name',
                default="json_file",
                choices=["animation", "json_file"],
            ),
    ) -> ModelOutput:
        args = self.args
        args.num_repetitions = int(num_repetitions)

        if args.num_repetitions != num_repetitions:
            self.data = get_dataset_loader(name=self.args.dataset,
                                    batch_size=args.num_repetitions,
                                    num_frames=self.num_frames,
                                    split='test',
                                    hml_mode='text_only')

        collate_args = [{'inp': torch.zeros(self.num_frames), 'tokens': None, 'lengths': self.num_frames, 'text': str(prompt)}]
        _, model_kwargs = collate(collate_args)

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.num_repetitions, device=dist_util.dev()) * args.guidance_param

        sample_fn = self.diffusion.p_sample_loop
        sample = sample_fn(
            self.model,
            (args.num_repetitions, self.model.njoints, self.model.nfeats, self.num_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        if self.model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = self.data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if self.model.data_rep in ['xyz', 'hml_vec'] else self.model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.num_repetitions,
                                                                                                self.num_frames).bool()
        sample = self.model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        all_motions = sample.cpu().numpy()
        # generate.py에서 np.save로 저장하는 포맷과 동일하게 반환
        if output_format in ['json_file', 'humanml3d']:
            # float32로 변환 및 리스트로 반환 (json 직렬화 호환)
            motions_xyz = all_motions.astype('float32').tolist()
            data_dict = {"motions": motions_xyz}
        else:
            data_dict = motions2hik(all_motions)

        return ModelOutput(json_file=data_dict, animation=[])
