from utils.config import Config
from run.AVMNIST_main import AVMNIST_main
from run.CREMAD_main import CREMAD_main
from run.URFunny_main import URFunny_main
from run.AVE_main import AVE_main
from run.MOSEI_main import MOSEI_main


def main():
    cfgs = Config()
    if cfgs.dataset == "AV-MNIST":
        AVMNIST_main(cfgs)
    elif cfgs.dataset == "CREMAD":
        CREMAD_main(cfgs)
    elif cfgs.dataset == "URFunny":
        URFunny_main(cfgs)
    elif cfgs.dataset == "AVE":
        AVE_main(cfgs)
    elif cfgs.dataset == "MOSEI":
        MOSEI_main(cfgs)
    


if __name__ == '__main__':
    main()