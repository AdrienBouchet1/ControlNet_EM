import numpy as np 
import pandas as pd 
import os 
from plotnine import aes,geom_line,ggplot


class MetricProcessor:


    def __init__(self,input_folder,output_folder) : 

        """
        
        """


        self.__check_constructor_input(input_folder,output_folder)
        #print("dataframe : ", self.dataframe)
        




    def __check_constructor_input(self,input_folder,output_folder): 


            assert os.path.isdir(input_folder), "le dossier d'input sélectionné n'existe pas"
            assert os.path.isdir(output_folder), "le dossier d'output n'existe pas"

            assert os.path.isfile(os.path.join(input_folder,"controlnet_run/version_0/metrics.csv")), " le fichier métriques n'a pas été trouvé"

            self.dataframe=pd.read_csv(os.path.join(input_folder,"controlnet_run/version_0/metrics.csv"),sep=",")

            self.output_folder=output_folder
    
    

    def __process_df(self): 
         

            self.dataframe = self.dataframe.drop(columns=["global_step","train/loss_vlb_epoch","train/loss_vlb_step","train/loss_step","train/loss_simple_step"], errors="ignore")
            #self.dataframe=self.dataframe[["step","train/loss_simple_epoch","epoch"]]
            
            cols_to_normalize = [col for col in self.dataframe.columns if col not in ["epoch", "step"]]
            self.dataframe[cols_to_normalize] = self.dataframe[cols_to_normalize].apply(lambda col: col / col.mean())

            self.dataframe= self.dataframe.melt(id_vars=["step", "epoch"], 
                 var_name="metric", 
                 value_name="value")
            self.dataframe["metric"]=pd.Categorical(self.dataframe["metric"])
            self.dataframe = self.dataframe.dropna()

            print(self.dataframe)

    def __call__(self): 
         
            self.__process_df()
            self.__plot()
            pass


    def __plot(self) : 
          


          plot=(ggplot(self.dataframe,aes(x="step",y="value",color="metric"))
                #ggplot(self.dataframe,aes(x="step",y="value"))
                +geom_line()
                       
                       
                       
                       )
          plot.save(os.path.join(self.output_folder,"training_metrics.pdf"))
if __name__=="__main__" : 
     
    input_folder="/home/adrienb/Documents/ControlNet_EM/logs"
    output_folder="/home/adrienb/Documents/ControlNet_EM/Analysis"

    m=MetricProcessor(input_folder,output_folder)

    m()
