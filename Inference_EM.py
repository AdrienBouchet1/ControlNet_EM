

import os 
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
from cldm.model import create_model, load_state_dict
import torch
from cldm.ddim_hacked import DDIMSampler

class InferenceEM : 



    def __init__(self,checkpoint_path : str): 

        """
        
        """


        self.__check_input(checkpoint_path)
        self.__instantiate_model()
        self.__instantiate_sampler()


        


    def __check_input(self,checkpoint_path) : 


        """
        
        
        """


        assert os.path.exists(checkpoint_path),"Le fichier de checkpoint saisi n'existe pas"    
        self.checkpoint_path=checkpoint_path

      

    def __instantiate_model(self): 

        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict(self.checkpoint_path, location='cpu'))
        self.model = self.model.cuda()
        self.model.eval()


    def __instantiate_sampler(self) : 

        self.sampler = DDIMSampler(self.model)



    def __prepare_input(self,matrix_2d):
            """
            Convertit une matrice 2D en image tensor [1, 3, H, W], sans normalisation.

            Args:
                matrix_2d (np.ndarray): Image conditionnelle en 2D (H, W)

            Returns:
                torch.Tensor: Tensor [1, 3, H, W] avec les 3 canaux identiques
            """
            if not isinstance(matrix_2d, np.ndarray):

                raise TypeError("L'entrée doit être une matrice NumPy 2D. Il est du type : {}".format(type(matrix_2d)))
        
            if matrix_2d.ndim != 2:
                print(matrix_2d.shape)
                raise ValueError("L'entrée doit être une matrice 2D (grayscale).")

        
            img_3c = np.stack([matrix_2d] * 3, axis=0)  # [3, H, W]

            tensor = torch.from_numpy(img_3c).unsqueeze(0)  # [1, 3, H, W]
            tensor=tensor.to(torch.float32).to("cuda")
            
            return tensor
    
    def __prepare_conditioning(self,image: np.ndarray) : 

        """
        
        """

        #image = np.stack([image]*3, axis=-1)
        #image = Image.fromarray(image)
        control = self.__prepare_input(image)

        prompt=""

        conditioning = {
                                "c_concat": [control],
                                "c_crossattn": [self.model.get_learned_conditioning([prompt])]
                            }

        return conditioning


    
    def generate(self,condition : np.ndarray): 


        """
        
        """


        shape = (4, 64, 64)  # dimensions en latents
        n_samples = 1


        conditioning=self.__prepare_conditioning(condition)
        samples, _ = self.sampler.sample(
                    S=50,
                    conditioning=conditioning,
                    batch_size=n_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=7.5,
                    unconditional_conditioning={
                                "c_concat": [torch.zeros_like(conditioning["c_concat"][0])],
                                "c_crossattn": [self.model.get_learned_conditioning([""])]
                            },
                    eta=0.0,
                    x_T=None
                )

        # Convertir latents en image
        decoded = self.model.decode_first_stage(samples)
        decoded = (decoded.clamp(-1., 1.) + 1) / 2.0
        decoded = decoded.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
        plt.imshow(decoded.astype(np.uint8))
        plt.axis('off')  # optionnel : cache les axes
        plt.show() 
        result = Image.fromarray(decoded.astype(np.uint8))
        return result

          







if __name__=='__main__' : 


    

    ckpt_path="/home/adrienb/Documents/ControlNet_EM/Run_1_pretrained/logs/controlnet_run/version_0/checkpoints/epoch=999-step=78999.ckpt"
    #ckpt_path='./models/control_sd15_ini.ckpt'
    assert os.path.exists(ckpt_path),"non reconnu"

    IEM=InferenceEM(checkpoint_path=ckpt_path)



    mat=np.full((512,512),fill_value=2)
    mat[:,256:]=6
    #mat[:256,:]=6
    print(mat)
    IEM.generate(mat)
    IEM.generate(mat)
    IEM.generate(mat)
    IEM.generate(mat)


    # IEM.generate(np.full((512,512),fill_value=8,dtype=np.uint8))
    # IEM.generate(np.full((512,512),fill_value=8,dtype=np.uint8))
    # IEM.generate(np.full((512,512),fill_value=4,dtype=np.uint8))
    # IEM.generate(np.full((512,512),fill_value=4,dtype=np.uint8))
    # IEM.generate(np.full((512,512),fill_value=4,dtype=np.uint8))
    # IEM.generate(np.full((512,512),fill_value=6,dtype=np.uint8))
    # IEM.generate(np.full((512,512),fill_value=6,dtype=np.uint8))
    # IEM.generate(np.full((512,512),fill_value=6,dtype=np.uint8))