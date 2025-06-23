import json
import cv2
import numpy as np
import os 
from PIL import Image 
from torch.utils.data import Dataset


class MyDataset_EM(Dataset):
    def __init__(self,input_folder):


        """On doit avoir un folder d'input du type : 
        
        input_folder /

                cat_1 / 
                cat_2 / 

                        image_1.tif
                        image_2.tif
                        mask_1.tif
                        mask_2.tif
                        




        """

        assert os.path.isdir(input_folder), "aucun dossier à l'emplacement : {}".format(input_folder)
        self.input_folder=input_folder
        self.data={}
        for cat in self.__get_categories(): 

                self.__get_files(os.path.join(self.input_folder,str(cat)),cat)





    def __get_categories(self): 
         

        """

             Permet juste de récupérer les catégories. Elles sont matérialisées par un nom de dossier n'étant qu'un entier

        """

        available_cat=list()
        for file in os.listdir(self.input_folder): 
            if os.path.isdir(os.path.join(self.input_folder,file)) :
                try : 
                       int_cat=int(file)
                       available_cat.append(file)
                       

                except : 

                     print("Le dossier : {} n'est pas considéré comme une catégorie".format(file))  
                     pass                         
            else:
                 
                 pass

        print("les catégories reconnues sont : \n ____________________________________________________________________________ \n {}".format(available_cat))
        
        return available_cat
    
        


    def __get_files(self,path,cat : int) : 
         
         list_files=os.listdir(path)


         for file in list_files : 
              if file.endswith(".tif") : 
                image=Image.open(os.path.join(path,file))
                image=np.array(image)
                mask=np.full(shape=(*image.shape,3), fill_value=cat)
                image=np.stack([image] * 3, axis=0) ### Pour l'avoir sur 3 channels

                self.data[len(list(self.data.keys()))]={"image" : image, "mask" : mask}

            
    def __len__(self) :
         return len(list(self.data.keys()))
    
  
    def __len__(self) :
         return len(list(self.data.keys()))
    

    def __getitem__(self,idx) : 
         

        source = self.data[idx]["mask"].astype(np.float32)
        #source=np.expand_dims(source, axis=2)


        target = (self.data[idx]["image"].astype(np.float32) / 127.5) - 1.0
        target=np.transpose(target, (1, 2, 0))


        return dict(jpg=target, txt="", hint=source)    
            




if __name__=="__main__": 
    
    dataset=MyDataset_EM(input_folder="/home/adrienb/Documents/Data/dataset_2_ControlNet_256_256/patches")
    print("la longueur : {}".format(len(dataset)))

    a=next(iter(dataset))
    print("prochain élément txt : {}, jpg : {}, carte : {}".format(a["txt"],a["jpg"].shape,a["hint"].shape))