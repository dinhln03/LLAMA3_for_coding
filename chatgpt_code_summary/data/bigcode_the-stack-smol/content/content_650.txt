import torch
import numpy as np
import torch.utils.data
from net import SurnameLSTM
from data import SurnameDataset

if __name__ == '__main__':
    net = SurnameLSTM()
    state_dict = torch.load('model.pth')
    net.load_state_dict(state_dict)

    dataset = SurnameDataset(subset='val')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    sample = iter(data_loader).__next__()

    pred = np.argmax(net(sample['values']).detach().numpy(), axis=1)
    gt = np.array(sample['raw_label'])

    accuracy = np.average(np.where(pred == gt, 1, 0))
    print('Accuracy on the validation data: {:.1f} %'.format(accuracy * 100))

    print('Please enter a surname to val:')
    input_name = input()
    name = input_name.lower()
    name_ascii = np.array([ord(c) for c in name])
    name_ascii = np.pad(name_ascii, ((0, 12 - name_ascii.__len__())), mode='constant', constant_values=0).astype(
        np.float32)
    name_ascii = torch.tensor([name_ascii])

    pred = np.argmax(net(name_ascii).detach().numpy(), axis=1)
    print('Mr / Ms. {}, I guess you are {}!'.format(input_name, ['English', 'Chinese', 'Japanese'][pred[0]]))
