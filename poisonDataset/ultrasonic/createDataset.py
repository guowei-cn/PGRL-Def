import argparse
import os, shutil, sys
parent_path = os.path.join(os.getcwd().split('PGRL-Def')[0], 'PGRL-Def')
sys.path.append(parent_path)
from lib.rawDataProcessing import download_raw, upsample_raw, createBenign, createPoisonTest, createPoisonTrain

from lib.dataLoader import target_audio

def create_data(poison_ratio, target_class, poison_type):
    # download raw data
    temp_folder = 'temp'
    download_raw(temp_folder)
    # raw data upsample to 44.1 khz
    upsample_raw(temp_folder, hz=44100)
    # build the benign dataset
    train_path, test_path = createBenign(temp_folder)
    # generate the poisoned test
    createPoisonTest('Test', save_path='poisonTest', poison_type=poison_type)
    # create he poisoned dataset
    createPoisonTrain('Train', save_path='poisonTrain_pr_{}_t_{}'.format(poison_ratio, target_class), poison_ratio=poison_ratio, target_class=target_class, poison_type=poison_type)

    # remove temp and download file
    os.remove("speech_commands_v0.01.tar.gz")
    shutil.rmtree(temp_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Poisoning data generation.")

    # Adding an argument for poison_ratio
    parser.add_argument('--poison_ratio', type=float, default=0.05,
                        choices=[0.003, 0.05],
                        help="Poison ratio value (choose between 0.003 and 0.05)")

    # Parse the arguments
    args = parser.parse_args()

    poison_ratio, target_class, poison_type = args.poison_ratio, target_audio, 'ultrasonic'
    print(poison_ratio, target_class)
    create_data(poison_ratio, target_class, poison_type)