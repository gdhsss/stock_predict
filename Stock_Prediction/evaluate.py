from LSTMModel import lstm
from dataset import getData
from parser_my import args
import torch
import matplotlib.pyplot as plt

def eval():
    # model = torch.load(args.save_file)
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    close_max, close_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)
    for idx, (x, label) in enumerate(test_loader):
        if args.useGPU:
            x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
        else:
            x = x.squeeze(1)
        pred = model(x)
        list = pred.data.squeeze(1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())
        mse=0
        mae=0
        acc=0
        acc5=0
    for i in range(len(preds)):
        a = preds[i][0] * (close_max - close_min) + close_min
        b = labels[i] * (close_max - close_min) + close_min
        print('预测值是%.2f,真实值是%.2f' % (a , b))
        mse += ((a-b)**2)
        mae += abs(a-b)
        if int(a)==int(b):
            acc+=1
        if abs(int(a)-int(b))<=int(b)*0.01:
            acc5+=1
    print('mse', mse / len(preds))
    print('mae', mae / len(preds))
    print('acc',acc / len(preds))
    print('acc in 99%',acc5 / len(preds))
    from sklearn.metrics import r2_score
    print('r2',r2_score(labels,preds))
    # Convert preds and labels to their original scales
    # scaled_preds = [pred[:][0] * int(close_max - close_min) + close_min for pred in preds]
    # scaled_labels = [label * int(close_max - close_min) + close_min for label in labels]
    #
    # # Create a range of indices for x-axis
    # indices = range(len(preds))
    #
    # # Plot the predicted values and true values
    # plt.plot(indices, scaled_preds, label='Predicted')
    # plt.plot(indices, scaled_labels, label='True')
    #
    # # Set labels and title
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.title('Predicted vs. True Values')
    #
    # # Show legend
    # plt.legend()

    # Display the plot
    plt.show()

eval()
