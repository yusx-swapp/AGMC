from torchvision import datasets

from data import resnet
from utils.NetworkPruning import *
from utils.feedback_calculation import *

def EvalCompressedModel(args, net,val_loader,device):


    #net = torch.nn.DataParallel(net).cuda()
    net.to(device)
    path = args.model_root
    net = network_pruning(net, [1 for i in range(500)], args)
    state_dict = torch.load(path, map_location=device)
    net.load_state_dict(state_dict)
    cudnn.benchmark = True



    criterion = nn.CrossEntropyLoss().to(device)

    val_top1,val_top5 = top5validate(val_loader, device, net, criterion)
    return val_top1,val_top5
