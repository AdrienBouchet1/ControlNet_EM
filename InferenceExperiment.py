import os 
import Inference_EM
import numpy as np 


class InferenceExperiment: 


    def __init__(self, ckpt_path,output_folder) :


        self.__check_input(checkpoint_path=ckpt_path,output_folder=output_folder)
        self.__instantiate_IEM()
        

    def __check_input(self,checkpoint_path,output_folder) : 


        """
        
        
        """


        assert os.path.exists(checkpoint_path),"Le fichier de checkpoint saisi n'existe pas"    
        self.checkpoint_path=checkpoint_path

        assert os.path.isdir(output_folder)," mauvais dossier output"
        self.output_folder=output_folder

    def __instantiate_IEM(self) :

        self.IEM=Inference_EM.InferenceEM(checkpoint_path=self.checkpoint_path)

    def run_multiple_mask(self): 


        for label in range(1,14) : 
            labFolder=os.path.join(self.output_folder,str(label))
            os.makedirs(labFolder)


            for i in range(20) : 

                mat=np.full((512,512),fill_value=label)
                im=self.IEM.generate(mat)

                im.save(os.path.join(labFolder,"run_{}.png".format(i)))







if __name__=="__main__" :

    ckpt="/home/adrienb/Documents/ControlNet_EM/logs/controlnet_run/version_0/checkpoints/epoch=999-step=78999.ckpt"
    output="/home/adrienb/Documents/ControlNet_EM/inférence"
    exp=InferenceExperiment(ckpt_path=ckpt,output_folder=output)
    exp.run_multiple_mask()








      