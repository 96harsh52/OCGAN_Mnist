import torch
from torchsummary import summary
from ocgan.model import ocgan
from ocgan.model import Encoder,Decoder, Dl,Dv,Classfier,weights_init 
import option1
from option1 import Options
from ocgan.data import create_dataloader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

def classify_image(csv_path, Main,output_csv_path):
    opt = Options().parse()
    dataloader = create_dataloader(opt)

    # Load your PyTorch model
    Classfier_model = os.path.join(Main, 'netc.pth')
    pretrained_dict = torch.load(Classfier_model)['state_dict']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        netc = Classfier(opt).to(device)
        netc.apply(weights_init)
        netc.load_state_dict(pretrained_dict)
    except IOError:
        raise IOError("netc weights not found")

    netc.eval()

    csv_path = pd.read_csv(csv_path)
    results =[]

    for i in csv_path['File Path'].index:
   
    # Load the image
        image = Image.open(csv_path['File Path'][i])
        path = csv_path['File Path'][i]
        label = csv_path['Label'][i]


        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        input_image = transform(image).unsqueeze(0)  # Add batch dimension

        # Use the model for prediction
        with torch.no_grad():
            output = netc(input_image.to(device))
            

            threshold = 0.5
            predictions = (output > threshold).float()
            predicted_class = 1 if output.item() > threshold else 0
            output = output.item()

                    # Append results to list
            results.append({ 'Path': path, 'Label': label, 'Predicted Class': predicted_class,'Output': output})

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_csv_path, index=False)


# Example usage
image_path = '/data/WORKSPACE_AA/OCGAN_FP/FEB2024OCG/OCGAN/data_ocg_csv/Test_All_Clean.csv'
Main = '/data/WORKSPACE_AA/OCGAN_FP/FEB2024OCG/OCGAN/CKPT256_OCG_Clean60K_23Feb_Model1_1/ocgan/Fingerprint/train/weights'
output_csv_path = '/data/WORKSPACE_AA/OCGAN_FP/FEB2024OCG/OCGAN/OCGAN_test_result/Test_All_Clean_DV_0.5.csv'
classify_image(image_path, Main,output_csv_path)