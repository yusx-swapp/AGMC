from utils.NetworkPruning import *
from utils.FeedbackCalculation import *

def LoadCompressedModel(net,path,val_loader,device):


    net = torch.nn.DataParallel(net).cuda()
    net.to(device)
    #net = channel_pruning(net, [1 for i in range(200)])
    net = pruning_imagnet(net, [1 for i in range(500)])
    state_dict = torch.load(path, map_location=device)

    net.load_state_dict(state_dict)
    cudnn.benchmark = True



    criterion = nn.CrossEntropyLoss().to(device)

    top5validate(val_loader, device, net, criterion)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet.__dict__['resnet56']()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root="DNN/datasets", train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)