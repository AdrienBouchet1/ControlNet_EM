import json
import cv2
import numpy as np
import os 
from PIL import Image 
from torch.utils.data import Dataset


class MyDataset(Dataset):
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

                self.__get_files(os.path.join(self.input_folder,int(cat)))





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

        print("les catégories reconnues sont : \n __________________________________________________________________________________________ \n {}".format(available_cat))
        
        return available_cat
    
        


    def __get_files(self,path,cat : int) : 
         
         list_files=os.listdir(path)


         for file in list_files : 
              if file.endswith(".tif") : 
                image=Image.open(os.path.join(path,file))
                image=np.array(image)
                mask=np.full(shape=image.shape, fill_valu=cat)
                image=np.stack([image] * 3, axis=0) ### Pour l'avoir sur 3 channels

                self.data[len(list(self.data.keys()))]={"image" : image, "mask" : mask}

            
    def __len__(self) :
         return len(list(self.data.keys()))
    
  
    def __len__(self) :
         return len(list(self.data.keys()))
    

    def __getitem__(self,idx) : 
         

        source = self.data[idx]["mask"].astype(np.float32)
        target = (self.data[idx]["Image"].astype(np.float32) / 127.5) - 1.0


        return dict(jpg=target, txt="", hint=source)    
            
    def __getitem__(self,idx) : 
         

        source = self.data[idx]["mask"].astype(np.float32)
        target = (self.data[idx]["Image"].astype(np.float32) / 127.5) - 1.0


        return dict(jpg=target, txt="", hint=source)    
            
    #     self.data = []
    #     with open('./training/fill50k/prompt.json', 'rt') as f:
    #         for line in f:
    #             self.data.append(json.loads(line))

    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, idx):
    #     item = self.data[idx]

    #     source_filename = item['source']
    #     target_filename = item['target']
    #     prompt = item['prompt']

    #     source = cv2.imread('./training/fill50k/' + source_filename)
    #     target = cv2.imread('./training/fill50k/' + target_filename)

    #     # Do not forget that OpenCV read images in BGR order.
    #     source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    #     target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    #     # Normalize source images to [0, 1].
    #     source = source.astype(np.float32) / 255.0

    #     # Normalize target images to [-1, 1].
    #     target = (target.astype(np.float32) / 127.5) - 1.0

    #     return dict(jpg=target, txt=prompt, hint=source)

