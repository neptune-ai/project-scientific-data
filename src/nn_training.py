import torch


def train(run, base_namespace, model, trainloader, device, optimizer, loss_mse):
    model.train()
    train_loss = 0
    for batch_idx, (inp, y_true) in enumerate(trainloader):
        data = inp.to(device).double().unsqueeze(dim=0).permute(1, 0, 2)
        optimizer.zero_grad()
        out, _ = model(data)
        data = data.squeeze()
        loss = loss_mse(out, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        run[f"{base_namespace}/train/epoch/loss"].log(
            train_loss / len(trainloader.dataset)
        )


def test(run, base_namespace, model, testloader, device, optimizer, loss_mse):
    with torch.no_grad():
        test_loss = 0
        for batch_idx, (inp, y_true) in enumerate(testloader):
            data = inp.to(device).double().unsqueeze(dim=0).permute(1, 0, 2)
            optimizer.zero_grad()
            out, _ = model(data)
            data = data.squeeze()
            loss = loss_mse(out, data)
            test_loss += loss.item()
            run[f"{base_namespace}/test/epoch/loss"].log(
                test_loss / len(testloader.dataset)
            )
