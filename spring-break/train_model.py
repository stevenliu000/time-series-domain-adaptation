
for epoch in range(args.epochs):
    """Training"""
    correct_train = 0
    total_bs_train = 0 # total batch size
    train_loss = 0
    start_train_time = time.time()
    for batch_id, (x, y) in enumerate(tqdm(train_loader)):
        if torch.cuda.is_available():
            x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]
        #normalize data
        x = (x - x_mean_tr) / x_std_tr
        # take the real and imaginary part out
        real = x[:,:,0].reshape(batch_size, seq_len, feature_dim).float()
        imag = x[:,:,1].reshape(batch_size, seq_len, feature_dim).float()
        if torch.cuda.is_available():
            real.to(device)
            imag.to(device)
        real, imag = encoder(real, imag)
        pred = CNet(torch.cat((real, imag), -1).reshape(x.shape[0], -1))
        loss = criterion(pred, y.argmax(-1))
        #print(pred.argmax(-1), y.argmax(-1))
        correct_train += (pred.argmax(-1) == y.argmax(-1)).sum().item()
        total_bs_train += y.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.shape[0]
    train_acc = float(correct_train) / total_bs_train
    train_loss = train_loss / total_bs_train
    best_acc_train = max(best_acc_train, train_acc)

