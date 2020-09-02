import argparse
import sys
from data import *
import time
from model import *


def train(model, dl_train, dl_val, args):
    best_loss = 1.
    best_epoch = 0
    for epoch in range(args.max_epoch):
        print('Started Epoch: {}'.format(epoch))
        model.train()
        run_epoch(model, dl_train, args)

        args.scheduler.step()

        model.eval()
        loss = validation(model, dl_val, args)

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            name = 'best_model' + time.strftime("%m-%d_%H-%M", time.localtime()) + '.pth'
            torch.save(model.state_dict(),
                       os.path.join(args.savedir, name))

    print('Best Val Loss: {:.2f}|Best epoch: {}'
          .format(best_loss, best_epoch))


def run_epoch(model, dl_train, args):
    """
    inp.shape= (time, batch, channel, height, width)
    out.shape= (batch, num_class)
    """
    running_loss = 0.
    start_time = time.time()
    for i, (inp, label) in enumerate(dl_train):
        inp, label = inp.to(args.device), label.to(args.device)

        # print('inp= ', inp.shape)
        # print('label= ', label)

        out = model(inp)
        # print('out=', out)
        # print('out.shape=\n', out.shape)

        loss = args.criterion(out, label)

        args.optim.zero_grad()
        loss.backward()
        args.optim.step()

        running_loss += loss.item()

        # if i % 100 == 99:
        #     print('Time: {:02}:{:02}| Progress: {:.2f}%| Loss: {:.8f}'
        #           .format(int((time.time() - start_time) // 60),
        #                   int((time.time() - start_time) % 60),
        #                   i / len(dl_train) * 100,
        #                   running_loss / (i + 1)), flush=True)
        #     print('out: {} | label: {}'.format(out, label))

    print('Time: {:02}:{:02}|Train Loss: {:.8f}'
          .format(int((time.time() - start_time) // 60),
                  int((time.time() - start_time) % 60),
                  running_loss / (i + 1), flush=True))


def validation(model, dl_val, args):
    print('Started Validation')

    acc = 0.
    start_time = time.time()

    with torch.no_grad():
        for i, (inp, label) in enumerate(dl_val):
            inp, label = inp.to(args.device), label.to(args.device)

            out = model(inp)
            predict = torch.argmax(out).item()
            label = label.item()
            true_result = int(predict == label)
            # print('out: {}| predict: {} | label: {} | result: {}'
            #       .format(out, predict, label, predict == label))

            acc += true_result
            # if i % 100 == 99:
            #     print('Time: {:02}:{:02}| Progress: {:.2f}%| Acc: {:.8f}'
            #           .format(int((time.time() - start_time) // 60),
            #                   int((time.time() - start_time) % 60),
            #                   i / len(dl_val) * 100,
            #                   acc / (i + 1)), flush=True)

        loss = len(dl_val) - acc
        print('Time: {:02}:{:02}|Validation Loss: {:.8f}|Val_ACC: {:.8f}'
              .format(int((time.time() - start_time) // 60),
                      int((time.time() - start_time) % 60),
                      loss / len(dl_val), acc / len(dl_val), accflush=True))

    return loss / len(dl_val)


def main():
    parser = argparse.ArgumentParser(
        description='Status of Traffic Training')

    parser.add_argument('--cpu', action='store_true', default=False)

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Sets the learning rate')

    parser.add_argument('--lr_steps', type=str, default='8,15,20')

    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Relative size of the validation subset')

    parser.add_argument('--max_epoch', type=int, default=1,
                        help='Maximum amount of epochs')

    parser.add_argument('--bs_mult', type=int, default=4,
                        help='Batch size for training per gpu')

    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()

    args.num_classes = 3

    if args.cpu:
        args.device = 'cpu'
    else:
        print('Use cuda')
        args.device = 'cuda'

    args.savedir = os.path.join(os.getcwd(), 'save')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    with open(os.path.join(args.savedir, 'args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    if args.lr_steps != '':
        args.lr_steps = [int(i) for i in args.lr_steps.split(',')]
    else:
        args.lr_steps = []

    # dl_train, dl_val = load_data(args)
    dl_train, dl_val = load_smote_data(args)

    # weight = torch.tensor([0.1, 0.9, 0.2])
    args.criterion = nn.NLLLoss()

    model = LSTM()

    model.to(args.device)

    args.optim = torch.optim.Adam(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  eps=1e-8)

    args.scheduler = torch.optim.lr_scheduler.MultiStepLR(args.optim,
                                                          milestones=args.lr_steps,
                                                          gamma=0.99)

    train(model, dl_train, dl_val, args)
    # validation(model, dl_val=dl_val, args=args)


if __name__ == '__main__':
    main()
