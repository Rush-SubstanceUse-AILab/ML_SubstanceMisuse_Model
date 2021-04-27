
import os
#to use GPU just commentout the lines below, and give the GPU number
# for GPU id use either "0" or "1" or "0,1" for both
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0"; 
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


class ProcessData:

    def dffiles(self, directory):
        txtfilesList = []
        patientL = []
        count = 0
        for file in os.listdir(directory):
            try:
                if file.endswith(".txt"):
                    txtfilesList.append(directory + str(file))
                    patientL.append(str(file))
                    count = count + 1
                else:
                    raise Exception("Please check the input file it must be txt files")
            except Exception as e:
                raise e
                print("No Files found here!")
         
        print("Total Files found", count)
        txtfilesTrain_df = pd.DataFrame({'Filename':patientL, 'FileList':txtfilesList})
        return txtfilesTrain_df

    def openData(self, text_df):
        #text_df = text_df.head(1000)
        text_df['FileList'] = text_df.FileList.apply(lambda x: open(x, "r").read())
        #text_df['FileList'] = text_df.FileList.apply(lambda x: ", ".join(x.split( )))
        print(text_df.head(10))
        return text_df

    def cutoff(self,x,y):
        if x < y:
            return 0
        else:
            return 1

    def predict(self, tokenizer_loc, model, text_df, outputDir):
        model = load_model(model)
        tokenizer_pkl = open(tokenizer_loc, 'rb')
        tokenizer = pickle.load(tokenizer_pkl)
        text_df = text_df.head(10)
        x_test = text_df.FileList
        #print(x_test.head(1000))
        sequences_test = tokenizer.texts_to_sequences(x_test)
        print("-----------Padding the data----------------")
        x_test = pad_sequences(sequences_test, maxlen = 35000)
        #print(x_test)
        print("----------Making Predictions---------------")
        y_prob = model.predict(np.asarray(x_test), batch_size = 1)
        
        alcohol_prob = y_prob[:,1]
        opioid_prob = y_prob[:,2]
        nonopioid_prob = y_prob[:,3]
        
        
        documents = text_df.Filename.tolist()
        prob_df = pd.DataFrame({'Filename': documents, 'Alcohol_Probability':alcohol_prob, 'Opioid_Probability':opioid_prob, "Non_Opioid_Probability":nonopioid_prob})
        prob_df["Alcohol_Prediction"] = prob_df.Alcohol_Probability.apply(lambda x: PD.cutoff(x, 0.5))
        prob_df["Opioid_Prediction"] = prob_df.Opioid_Probability.apply(lambda x: PD.cutoff(x, 0.5))
        prob_df["Non_Opioid_Prediction"] = prob_df.Non_Opioid_Probability.apply(lambda x: PD.cutoff(x, 0.5))

        prob_df.to_csv(outputDir + 'ML_SubstanceMisue_Prediction.csv', sep = ',', index = False)
        print("Prediction Completed, file is saved!")
        return prob_df
        
        
if __name__ == "__main__":
    PD = ProcessData()
    
    inputDir = "/usr/inputdata/"
    outputDir = "/usr/outputresult/"
    tokenizer_loc = "CUIS_tokenizer.pkl"
    model_loc = "CNN_Multilabel.h5"
    txtfilesTrain_df = PD.dffiles(inputDir)
    text_df = PD.openData(txtfilesTrain_df)
    data_df = PD.predict(tokenizer_loc, model_loc, text_df, outputDir)
