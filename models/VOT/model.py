# placeholder for VOT model definition

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

import torch.backends.cudnn as cudnn
import numpy as np
import os
from collections import OrderedDict


__all__ = ['vggm']


pretrained_settings = {
    'vggm': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/vggm-786f2434.pth',
            'input_space': 'BGR',
            'input_size': [3, 221, 221],
            'input_range': [0, 255],
            'mean': [123.68, 116.779, 103.939],
            'std': [1, 1, 1],
            'num_classes': 1000
        }
    }
}

pretrained_settings_adnet = {
        'adnet': {
        'input_space': 'BGR',
        'input_size': [3, 112, 112],
        'input_range': [0, 255],
        'mean': [123.68, 116.779, 103.939],
        'std': [1, 1, 1],
        'num_classes': 11
    }
}

class SpatialCrossMapLRN(nn.Module):
    # Lokalna normalizacji odpowiedzi w przestrzeni (LRN) - dość starożyten
    # rozwiązanie -- które było w VGG, AlexNet i ADNecie. Pewnie to wymienimi na nowsze w ulepszeniach.

    def __init__(self, local_size=5, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNEL=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNEL = ACROSS_CHANNEL
        if ACROSS_CHANNEL:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta
        self.k = k
    
    def forward(self, x):
        if self.ACROSS_CHANNEL:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
        else:
            div = self.average(x.pow(2))
        div = div.mul(self.alpha).add(self.k).pow(self.beta)
        return x.div(div)
    

class VGGM(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGGM, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            SpatialCrossMapLRN(local_size=5, alpha=0.0001, beta=0.75, k=1, ACROSS_CHANNEL=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            SpatialCrossMapLRN(local_size=5, alpha=0.0001, beta=0.75, k=1, ACROSS_CHANNEL=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vggm_create(num_classes=1000, pretrained='imagenet'):
    if pretrained:
        settings = pretrained_settings['vggm'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        model = VGGM(num_classes=num_classes)
        model.load_state_dict(torch.load('vggm-786f2434.pth'))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = VGGM(num_classes=num_classes)
    return model


def get_action_history_onehot(action_history, opts):
    ah_onehot = []
    for i in range(len(action_history)):
        onehot = np.zeros(opts['num_actions'])
        if action_history[i] >= 0 and action_history[i] < opts['num_actions']:
            onehot[action_history[i]] = 1

        ah_onehot.extend(onehot)

    return ah_onehot


class ADNetDomainSpecific(nn.Module):
    def __init__(self, num_classes, num_history):
        super(ADNetDomainSpecific, self).__init__()
        action_dynamic_size = num_classes * num_history
        self.fc6 = nn.Linear(512 + action_dynamic_size, num_classes)
        self.fc7 = nn.Linear(512 + action_dynamic_size, 2)
    
    def load_weights(self, base_file, video_index):
        other, ext = os.path.splitext(base_file)
        if ext == '.pth' or ext == '.pkl':
            print('Loading Adnet specyficzny dla video {} z {}'.format(video_index, base_file))
            checkpoint = torch.load(base_file, map_location=lambda storage, loc: storage)

            pretrained_dict = checkpoint['adnet_domain_specific_state_dict'][video_index].state_dict()
            model_dict = self.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(pretrained_dict)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        
    
    def load_weights_from_adnet(self, adnet_model):
        pretrained_dict = adnet_model.state_dict()
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)
    
class ADNet(nn.Module):
    def __init__(self, base_network, opts, num_classes=11, phase='train', num_history=10, use_gpu=True):
        super(ADNet, self).__init__()
        self.base_network = base_network
        self.domain_specific = ADNetDomainSpecific(num_classes=num_classes, num_history=num_history)
        self.phase = phase
        self.opts = opts
        self.use_gpu = use_gpu

        self.fc4_5 = nn.Sequential(
            nn.Linear(18432, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.action_history = np.full(num_history, -1)  # inicjalizowane akcją -1 co oznacza że jeszcze nie było akcji

        self.action_dynamic_size = num_classes * num_history
        self.action_dynamic = torch.Tensor(np.zeros(self.action_dynamic_size)).float()

        if self.use_gpu:
            self.action_dynamic = self.action_dynamic.cuda()
        
        self.fc6 = nn.Linear(512 + self.action_dynamic_size, num_classes)
        self.fc7 = nn.Linear(512 + self.action_dynamic_size, 2)

        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x, action_history=None, update_action_history=False):
        """
        x: wejście sieci
        action_dynamic: wcześniejszy stan, jeśli None to używamy self.action_dynamic,
        update_action_dynamic: czy zaktualizować action_dynamic na podstawie rezultatów -- nie robi się tego w SL"""

        assert x is not None, "Input cannot be None"
        x = self.base_network(x)
        x = x.view(x.size(0), -1)
        x = self.fc4_5(x)

        if action_history is None:
            x = torch.cat((x, self.action_dynamic.expand(x.shape[0], self.action_dynamic.shape[0])), 1)
        else:
            x = torch.cat((x, action_history))
        
        fc6_out = self.fc6(x)
        fc7_out = self.fc7(x)

        if update_action_history:
            selected_action = torch.argmax(fc6_out, dim=1).cpu().numpy()
            self.action_history[1:] = self.action_history[:-1]  # przesuwamy historię w prawo
            self.action_history[0] = selected_action  # dodajemy nową akcję
            self.action_dynamic(self.action_history.flatten())  # aktualizujemy action_dynamic na podstawie nowej historii

        return fc6_out, fc7_out
    

    def load_domain_specific(self, adnet_domain_specific):

        # if self.use_gpu:
        #     adnet_domain_specific_ = nn.DataParallel(adnet_domain_specific)
        #     adnet_domain_specific_ = adnet_domain_specific_.cuda()
        # else:
        #     adnet_domain_specific_ = adnet_domain_specific

        domain_specific_state_dict = adnet_domain_specific.state_dict()
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in domain_specific_state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)

        pass


    def load_weights(self, base_file, load_domain_specific=None):
        other, ext = os.path.splitext(base_file)
        if ext == '.pth' or ext == '.pkl':
            print('Loading Adnet from {}'.format(base_file))
            checkpoint = torch.load(base_file, map_location=lambda storage, loc: storage)

            pretrained_dict = checkpoint['adnet_state_dict']
            model_dict = self.state_dict()

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                if 'module' in k:
                    name = k[7:]  
                new_state_dict[name] = v
            pretrained_dict = new_state_dict

            # 1. filtrowanie niepotrzebnych kluczy
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. nadpisywanie istniejących kluczy            
            model_dict.update(pretrained_dict)
            # 3. ładowanie nowego state dict            self.load_state_dict(pretrained_dict)
            self.load_state_dict(pretrained_dict)       

            if load_domain_specific:
                self.load_domain_specific(checkpoint['adnet_domain_specific_state_dict'])
                model_dict = self.state_dict()

                #1. filtrowanie niepotrzebnych kluczy
                pretrained_dict = {k: v for k, v in checkpoint['adnet_domain_specific_state_dict'].items() if k in model_dict}
                # 2. nadpisywanie istniejących kluczy
                model_dict.update(pretrained_dict)
                # 3. ładowanie nowego state dict
                self.load_state_dict(model_dict)
            
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    
    def update_action_history(self, action_history):
        oneshot = get_action_history_onehot(action_history, self.opts)
        self.action_dynamic = torch.Tensor(oneshot).float()
        if self.use_gpu:
            self.action_dynamic = self.action_dynamic.cuda()
    
    def reset_action_history(self):
        self.action_history = np.full(self.opts['num_history'], -1)  # resetujemy historię do stanu początkowego
        self.action_dynamic = torch.Tensor(np.zeros(self.action_dynamic_size)).float()  # resetujemy action_dynamic do stanu początkowego
        if self.use_gpu:
            self.action_dynamic = self.action_dynamic.cuda()
    
    def get_action_history_onehot(self):
        return self.action_dynamic
    
    def set_phase(self, phase):
        self.phase = phase
    

def adnet_create(opts, base_network, trained_file=None, random_initialize_domain_specific=False, multidomain=True):


    assert base_network in ['vggm'], "Currently only vggm base network is supported"

    num_classes = opts['num_actions']
    num_history = opts['num_history']


    settings_adnet = pretrained_settings_adnet['adnet']

    if base_network == 'vggm':
        base_network = vggm_create()
        base_network = base_network.features[0:10]  # bierzemy tylko część cechującą, bez klasyfikatora

    else:
        base_network = None  # placeholder, jeśli będziemy chcieli dodać inne sieci bazowe w przyszłości
        base_network = base_network.features[0:10]
    
    if trained_file:
        adnet_model = ADNet(base_network=base_network, opts=opts, num_classes=num_classes, num_history=num_history)
        adnet_model.load_weights(trained_file)

        adnet_model.input_space = settings_adnet['input_space']
        adnet_model.input_size = settings_adnet['input_size']
        adnet_model.input_range = settings_adnet['input_range']
        adnet_model.mean = settings_adnet['mean']
        adnet_model.std = settings_adnet['std']
    else:
        adnet_model = ADNet(base_network=base_network, opts=opts, num_classes=num_classes, num_history=num_history)
    
    domain_nets = []
    if multidomain:
        num_vids = opts['num_videos']
    else:
        num_vids = 1
    
    for idx in range(num_vids):
            domain_nets.append(ADNetDomainSpecific(num_classes=num_classes, num_history=num_history))

            scal = torch.Tensor([0.01])

            if trained_file and not random_initialize_domain_specific:
                domain_nets[idx].load_weights(trained_file, idx)
            else:
                # fc 6
                nn.init.normal_(domain_nets[idx].fc6.weight.data)
                domain_nets[idx].fc6.weight.data = domain_nets[idx].fc6.weight.data * scal.expand_as(domain_nets[idx].fc6.weight.data)
                domain_nets[idx].fc6.bias.data.fill_(0)
                # fc 7
                nn.init.normal_(domain_nets[idx].fc7.weight.data)
                domain_nets[idx].fc7.weight.data = domain_nets[idx].fc7.weight.data * scal.expand_as(domain_nets[idx].fc7.weight.data)
                domain_nets[idx].fc7.bias.data.fill_(0)

    return adnet_model, domain_nets